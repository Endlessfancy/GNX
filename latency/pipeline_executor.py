"""
Pipeline Executor - Industrial-grade Pipeline Benchmark Framework

Implements true pipeline execution where different stages process different batches
in the same cycle, with cross-cycle buffer for intermediate results.

Key Components:
- PipelineBuffer: Cross-cycle data buffer
- StageStats: Per-stage timing statistics
- DataParallelStage: Split -> Parallel Exec -> Merge
- SingleDeviceStage: Simple single-device execution
- PipelineBenchmark: Pipeline orchestrator
"""

# Path setup for running as script or module
import sys
from pathlib import Path
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from copy import deepcopy

try:
    import openvino as ov
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

from latency.stage_executor import StageExecutor, ProfilingResult
from latency.graph_partitioner import HaloPartitioner, PartitionData
from latency.profiler import PipelineProfiler


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class StageStats:
    """Statistics for a single stage execution."""
    stage_name: str
    batch_id: Optional[int]       # None if pipeline filling/draining
    wall_time_ms: float
    split_time_ms: float          # CPU data splitting overhead
    merge_time_ms: float          # CPU result merging overhead
    device_times: Dict[str, float]  # {device: compute_ms}

    @property
    def total_compute_ms(self) -> float:
        """Sum of all device compute times."""
        return sum(self.device_times.values())

    @property
    def max_device_time_ms(self) -> float:
        """Max device time (parallel execution bottleneck)."""
        return max(self.device_times.values()) if self.device_times else 0.0


@dataclass
class CycleStats:
    """Statistics for a single pipeline cycle."""
    cycle_id: int
    stage_stats: List[StageStats]

    @property
    def total_wall_time_ms(self) -> float:
        """Total wall time for this cycle."""
        return sum(s.wall_time_ms for s in self.stage_stats if s.batch_id is not None)


# ============================================================================
# Pipeline Buffer
# ============================================================================

class PipelineBuffer:
    """
    Cross-cycle data buffer for pipeline execution.

    In a true pipeline, stage N processes batch B while stage N+1 processes
    batch B-1 (from the previous cycle's stage N output).

    Buffer Layout:
        buffers[stage_id] = output from this stage in the previous cycle
    """

    def __init__(self, num_stages: int):
        self.num_stages = num_stages
        # buffers[i] holds the output of stage i from the previous cycle
        self.buffers: List[Optional[Dict[str, np.ndarray]]] = [None] * num_stages

    def get_input(self,
                  stage_id: int,
                  cycle_id: int,
                  raw_inputs: List[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
        """
        Get input for a stage in a given cycle.

        Args:
            stage_id: Stage index (0 = first stage)
            cycle_id: Current cycle index
            raw_inputs: List of raw input batches

        Returns:
            Input data for this stage, or None if pipeline is filling/draining
        """
        if stage_id == 0:
            # First stage gets input directly from raw_inputs
            batch_id = cycle_id
            if batch_id < len(raw_inputs):
                return raw_inputs[batch_id]
            else:
                return None  # Pipeline draining
        else:
            # Other stages get input from previous stage's buffer
            # batch_id for this stage = cycle_id - stage_id
            batch_id = cycle_id - stage_id
            if batch_id < 0:
                return None  # Pipeline filling
            return self.buffers[stage_id - 1]

    def set_output(self, stage_id: int, output: Dict[str, np.ndarray]):
        """Save stage output for next cycle."""
        self.buffers[stage_id] = output

    def get_batch_id(self, stage_id: int, cycle_id: int, num_batches: int) -> Optional[int]:
        """Get the batch ID being processed by a stage in a cycle."""
        batch_id = cycle_id - stage_id
        if batch_id < 0 or batch_id >= num_batches:
            return None
        return batch_id

    def clear(self):
        """Clear all buffers."""
        self.buffers = [None] * self.num_stages


# ============================================================================
# Stage Executors
# ============================================================================

class SingleDeviceStage:
    """Single-device stage executor (no data parallel)."""

    def __init__(self,
                 name: str,
                 core: Core,
                 model_path: Path,
                 device: str,
                 profiler: PipelineProfiler,
                 npu_static_nodes: Optional[int] = None,
                 npu_static_edges: Optional[int] = None):
        self.name = name
        self.device = device
        self.executor = StageExecutor(
            core=core,
            model_path=model_path,
            device=device,
            stage_name=name,
            profiler=profiler,
            npu_static_nodes=npu_static_nodes if device == 'NPU' else None,
            npu_static_edges=npu_static_edges if device == 'NPU' else None
        )

    def run(self,
            inputs: Dict[str, np.ndarray],
            batch_id: int) -> Tuple[Dict[str, np.ndarray], StageStats]:
        """Execute stage on single device."""
        wall_start = time.perf_counter()

        outputs, profiling = self.executor.run(inputs, batch_id)

        wall_time = (time.perf_counter() - wall_start) * 1000

        stats = StageStats(
            stage_name=self.name,
            batch_id=batch_id,
            wall_time_ms=wall_time,
            split_time_ms=0.0,
            merge_time_ms=0.0,
            device_times={self.device: profiling.device_time_ms}
        )

        return outputs, stats


class DataParallelStage:
    """
    Data-parallel stage executor with halo expansion for graphs.

    Execution flow:
    1. Split: Partition data using HaloPartitioner
    2. Parallel Exec: Run on multiple devices async
    3. Merge: Combine results (owned nodes only)
    """

    def __init__(self,
                 name: str,
                 core: Core,
                 model_paths: Dict[str, Path],
                 devices: List[str],
                 ratios: List[float],
                 profiler: PipelineProfiler,
                 is_graph_stage: bool = True,
                 npu_static_nodes: Optional[int] = None,
                 npu_static_edges: Optional[int] = None):
        """
        Initialize data-parallel stage.

        Args:
            name: Stage name
            core: OpenVINO Core
            model_paths: {device: model_path} mapping
            devices: List of devices
            ratios: Data split ratios
            profiler: PipelineProfiler
            is_graph_stage: Whether this stage processes graph data
            npu_static_nodes: Static node count for NPU
            npu_static_edges: Static edge count for NPU
        """
        self.name = name
        self.devices = devices
        self.ratios = ratios
        self.profiler = profiler
        self.is_graph_stage = is_graph_stage
        self.npu_static_nodes = npu_static_nodes
        self.npu_static_edges = npu_static_edges

        # Create executors for each device
        self.executors: Dict[str, StageExecutor] = {}
        for i, device in enumerate(devices):
            if device not in model_paths:
                print(f"  Warning: No model for {device}")
                continue

            self.executors[device] = StageExecutor(
                core=core,
                model_path=model_paths[device],
                device=device,
                stage_name=f"{name}_{device}",
                profiler=profiler,
                stream_id=i,
                npu_static_nodes=npu_static_nodes if device == 'NPU' else None,
                npu_static_edges=npu_static_edges if device == 'NPU' else None
            )

        self.active_devices = list(self.executors.keys())

        # Graph partitioner (if needed)
        if is_graph_stage:
            self.partitioner = HaloPartitioner(
                num_partitions=len(self.active_devices),
                ratios=ratios[:len(self.active_devices)]
            )
        else:
            self.partitioner = None

    def run(self,
            inputs: Dict[str, np.ndarray],
            batch_id: int) -> Tuple[Dict[str, np.ndarray], StageStats]:
        """
        Execute stage with data parallelism.

        Args:
            inputs: Input tensors {name: data}
            batch_id: Batch identifier

        Returns:
            (outputs, stats)
        """
        wall_start = time.perf_counter()

        # 1. Split data
        split_start = time.perf_counter()
        if self.is_graph_stage and 'edge_index' in inputs:
            partitioned_inputs, partitions = self._split_graph_data(inputs)
        else:
            partitioned_inputs, partitions = self._split_dense_data(inputs)
        split_time = (time.perf_counter() - split_start) * 1000

        # 2. Parallel execution
        for device in self.active_devices:
            self.executors[device].start_async(partitioned_inputs[device])

        device_times = {}
        results = {}
        for device in self.active_devices:
            out, prof = self.executors[device].wait(batch_id)
            device_times[device] = prof.device_time_ms
            results[device] = out

        # 3. Merge results
        merge_start = time.perf_counter()
        if self.is_graph_stage and partitions:
            merged = self._merge_graph_outputs(results, partitions, inputs)
        else:
            merged = self._merge_dense_outputs(results)
        merge_time = (time.perf_counter() - merge_start) * 1000

        wall_time = (time.perf_counter() - wall_start) * 1000

        stats = StageStats(
            stage_name=self.name,
            batch_id=batch_id,
            wall_time_ms=wall_time,
            split_time_ms=split_time,
            merge_time_ms=merge_time,
            device_times=device_times
        )

        return merged, stats

    def _split_graph_data(self,
                          inputs: Dict[str, np.ndarray]
                          ) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[PartitionData]]:
        """Split graph data using halo partitioner."""
        x = inputs['x']
        edge_index = inputs['edge_index']

        partitions = self.partitioner.partition(x, edge_index)

        partitioned = {}
        for device, part in zip(self.active_devices, partitions):
            partitioned[device] = {
                'x': part.x_local.astype(np.float32),
                'edge_index': part.edge_index_local.astype(np.int64)
            }

        return partitioned, partitions

    def _split_dense_data(self,
                          inputs: Dict[str, np.ndarray]
                          ) -> Tuple[Dict[str, Dict[str, np.ndarray]], None]:
        """Split dense data (non-graph) by first dimension."""
        # Get main data array
        main_key = list(inputs.keys())[0]
        main_data = inputs[main_key]
        n_total = main_data.shape[0]

        partitioned = {}
        start = 0
        for i, device in enumerate(self.active_devices):
            if i == len(self.active_devices) - 1:
                end = n_total
            else:
                end = start + int(n_total * self.ratios[i])

            device_inputs = {}
            for key, data in inputs.items():
                if data.shape[0] == n_total:
                    device_inputs[key] = data[start:end]
                else:
                    # Broadcast smaller arrays
                    device_inputs[key] = data

            partitioned[device] = device_inputs
            start = end

        return partitioned, None

    def _merge_graph_outputs(self,
                             results: Dict[str, Dict[str, np.ndarray]],
                             partitions: List[PartitionData],
                             original_inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Merge graph outputs using partition info."""
        total_nodes = original_inputs['x'].shape[0]

        # Get outputs from each device
        outputs_list = []
        for device in self.active_devices:
            outputs_list.append(results[device]['output'])

        merged_output = self.partitioner.merge_outputs(outputs_list, partitions, total_nodes)

        return {'output': merged_output}

    def _merge_dense_outputs(self,
                             results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Merge dense outputs by concatenation."""
        # Concatenate outputs in device order
        output_arrays = [results[device]['output'] for device in self.active_devices]
        merged = np.concatenate(output_arrays, axis=0)
        return {'output': merged}


# ============================================================================
# Pipeline Benchmark
# ============================================================================

class PipelineBenchmark:
    """
    Industrial-grade pipeline benchmark framework.

    Implements true pipeline execution where different stages process
    different batches in the same cycle.
    """

    def __init__(self, profiler: PipelineProfiler):
        self.profiler = profiler
        self.core = Core() if OPENVINO_AVAILABLE else None
        self.stages: List[Union[SingleDeviceStage, DataParallelStage]] = []

    def add_single_stage(self,
                         name: str,
                         model_path: Path,
                         device: str,
                         **kwargs):
        """Add a single-device stage."""
        stage = SingleDeviceStage(
            name=name,
            core=self.core,
            model_path=model_path,
            device=device,
            profiler=self.profiler,
            **kwargs
        )
        self.stages.append(stage)
        print(f"Added single-device stage: {name} on {device}")

    def add_dp_stage(self,
                     name: str,
                     model_paths: Dict[str, Path],
                     devices: List[str],
                     ratios: List[float],
                     is_graph_stage: bool = True,
                     **kwargs):
        """Add a data-parallel stage."""
        stage = DataParallelStage(
            name=name,
            core=self.core,
            model_paths=model_paths,
            devices=devices,
            ratios=ratios,
            profiler=self.profiler,
            is_graph_stage=is_graph_stage,
            **kwargs
        )
        self.stages.append(stage)
        print(f"Added DP stage: {name} on {devices} (ratios={ratios})")

    def run_pipeline(self,
                     batches: List[Dict[str, np.ndarray]],
                     iterations: int = 10,
                     warmup: int = 3) -> Dict[str, Any]:
        """
        Run pipeline benchmark.

        Args:
            batches: List of input batches (each is {name: data})
            iterations: Number of iterations per batch
            warmup: Warmup iterations (not counted)

        Returns:
            Benchmark results with per-cycle and aggregate statistics
        """
        num_batches = len(batches)
        num_stages = len(self.stages)
        num_cycles = num_batches + num_stages - 1

        print(f"\n{'='*70}")
        print(f"Pipeline Benchmark")
        print(f"{'='*70}")
        print(f"Batches: {num_batches}, Stages: {num_stages}, Cycles: {num_cycles}")
        print(f"Iterations: {iterations} (warmup: {warmup})")

        # Warmup
        print(f"\nWarming up ({warmup} iterations)...")
        for _ in range(warmup):
            self._run_single_iteration(batches)

        # Measurement
        print(f"Running {iterations} iterations...")
        all_iteration_stats: List[List[CycleStats]] = []

        for iter_id in range(iterations):
            cycle_stats_list = self._run_single_iteration(batches)
            all_iteration_stats.append(cycle_stats_list)

            if iter_id % max(1, iterations // 5) == 0:
                print(f"  Iteration {iter_id + 1}/{iterations} complete")

        # Aggregate statistics
        results = self._aggregate_results(all_iteration_stats, num_batches, num_stages)

        return results

    def _run_single_iteration(self,
                              batches: List[Dict[str, np.ndarray]]) -> List[CycleStats]:
        """Run one complete pipeline iteration."""
        num_batches = len(batches)
        num_stages = len(self.stages)
        num_cycles = num_batches + num_stages - 1

        buffer = PipelineBuffer(num_stages)
        cycle_stats_list = []

        for cycle_id in range(num_cycles):
            stage_stats_list = []

            for stage_id, stage in enumerate(self.stages):
                # Get input for this stage
                inputs = buffer.get_input(stage_id, cycle_id, batches)
                batch_id = buffer.get_batch_id(stage_id, cycle_id, num_batches)

                if inputs is not None:
                    # Execute stage
                    outputs, stats = stage.run(inputs, batch_id)
                    buffer.set_output(stage_id, outputs)
                else:
                    # Pipeline filling/draining - create empty stats
                    stats = StageStats(
                        stage_name=stage.name,
                        batch_id=None,
                        wall_time_ms=0.0,
                        split_time_ms=0.0,
                        merge_time_ms=0.0,
                        device_times={}
                    )

                stage_stats_list.append(stats)

            cycle_stats = CycleStats(cycle_id=cycle_id, stage_stats=stage_stats_list)
            cycle_stats_list.append(cycle_stats)

        return cycle_stats_list

    def _aggregate_results(self,
                           all_stats: List[List[CycleStats]],
                           num_batches: int,
                           num_stages: int) -> Dict[str, Any]:
        """Aggregate statistics across iterations."""
        num_iterations = len(all_stats)

        # Per-stage aggregates
        stage_results = {}
        for stage_id, stage in enumerate(self.stages):
            # Collect all valid (non-None batch_id) executions
            wall_times = []
            split_times = []
            merge_times = []
            device_times_all: Dict[str, List[float]] = {}

            for iter_stats in all_stats:
                for cycle_stats in iter_stats:
                    stage_stats = cycle_stats.stage_stats[stage_id]
                    if stage_stats.batch_id is not None:
                        wall_times.append(stage_stats.wall_time_ms)
                        split_times.append(stage_stats.split_time_ms)
                        merge_times.append(stage_stats.merge_time_ms)

                        for device, t in stage_stats.device_times.items():
                            if device not in device_times_all:
                                device_times_all[device] = []
                            device_times_all[device].append(t)

            stage_results[stage.name] = {
                'wall_time_ms': {
                    'mean': np.mean(wall_times) if wall_times else 0,
                    'std': np.std(wall_times) if wall_times else 0,
                    'min': np.min(wall_times) if wall_times else 0,
                    'max': np.max(wall_times) if wall_times else 0,
                },
                'split_time_ms': {
                    'mean': np.mean(split_times) if split_times else 0,
                },
                'merge_time_ms': {
                    'mean': np.mean(merge_times) if merge_times else 0,
                },
                'device_times_ms': {
                    device: {
                        'mean': np.mean(times),
                        'std': np.std(times),
                    }
                    for device, times in device_times_all.items()
                }
            }

        # Total pipeline throughput
        # Time for a full pipeline = sum of all stage wall times in steady state
        total_times = []
        for iter_stats in all_stats:
            # Consider cycles where all stages are active (steady state)
            for cycle_stats in iter_stats:
                cycle_wall = sum(
                    s.wall_time_ms for s in cycle_stats.stage_stats
                    if s.batch_id is not None
                )
                if cycle_wall > 0:
                    total_times.append(cycle_wall)

        avg_cycle_time = np.mean(total_times) if total_times else 0
        throughput = 1000.0 / avg_cycle_time if avg_cycle_time > 0 else 0

        return {
            'config': {
                'num_batches': num_batches,
                'num_stages': num_stages,
                'num_cycles': num_batches + num_stages - 1,
                'iterations': num_iterations,
            },
            'stage_results': stage_results,
            'pipeline': {
                'avg_cycle_time_ms': avg_cycle_time,
                'throughput_batches_per_sec': throughput,
            }
        }

    def print_report(self, results: Dict[str, Any]):
        """Print formatted benchmark report."""
        print(f"\n{'='*70}")
        print("Pipeline Benchmark Report")
        print(f"{'='*70}")

        config = results['config']
        print(f"Config: {config['num_batches']} batches, {config['num_stages']} stages, "
              f"{config['num_cycles']} cycles")
        print(f"Iterations: {config['iterations']}")

        print(f"\n[Per-Stage Results]")
        print("-" * 70)

        for stage_name, stats in results['stage_results'].items():
            print(f"\n  {stage_name}:")
            wall = stats['wall_time_ms']
            print(f"    Wall Time: {wall['mean']:.2f} +/- {wall['std']:.2f} ms "
                  f"(min={wall['min']:.2f}, max={wall['max']:.2f})")

            if stats['split_time_ms']['mean'] > 0:
                print(f"    Split Time: {stats['split_time_ms']['mean']:.3f} ms")
            if stats['merge_time_ms']['mean'] > 0:
                print(f"    Merge Time: {stats['merge_time_ms']['mean']:.3f} ms")

            if stats['device_times_ms']:
                print(f"    Device Breakdown:")
                for device, dt in stats['device_times_ms'].items():
                    print(f"      {device}: {dt['mean']:.2f} +/- {dt['std']:.2f} ms")

        pipeline = results['pipeline']
        print(f"\n[Pipeline Summary]")
        print("-" * 70)
        print(f"  Avg Cycle Time: {pipeline['avg_cycle_time_ms']:.2f} ms")
        print(f"  Throughput: {pipeline['throughput_batches_per_sec']:.1f} batches/sec")
        print(f"{'='*70}")


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Pipeline Executor module loaded successfully")
    print(f"OpenVINO available: {OPENVINO_AVAILABLE}")

    if not OPENVINO_AVAILABLE:
        print("Skipping test - OpenVINO not available")
    else:
        print("\nTo test, run test_pep_latency.py with the new architecture")
