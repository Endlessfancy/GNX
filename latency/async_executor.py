"""
Async Executor - Multi-device Parallel Execution

Provides async parallel execution across CPU/GPU/NPU with:
- Data partitioning based on ratios
- Async inference on all devices simultaneously
- Result merging
- Precise timing for each phase
- NPU static shape padding support
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

try:
    import openvino as ov
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

try:
    from profiler import PipelineProfiler
    from stage_executor import StageExecutor, ProfilingResult
    from npu_utils import NPUPaddingInfo
except ImportError:
    from .profiler import PipelineProfiler
    from .stage_executor import StageExecutor, ProfilingResult
    from .npu_utils import NPUPaddingInfo


@dataclass
class BlockExecutionResult:
    """Results from block execution."""
    outputs: Dict[str, np.ndarray]
    wall_time_ms: float
    partition_time_ms: float
    device_times: Dict[str, float]
    merge_time_ms: float
    profiling_results: Dict[str, ProfilingResult] = field(default_factory=dict)


class AsyncBlockExecutor:
    """
    Async block executor for PEP blocks.

    Supports:
    - Single device execution
    - Data parallel execution with multiple devices
    - Async inference for overlapping compute
    """

    def __init__(self,
                 core: 'Core',
                 model_paths: Dict[str, Path],
                 devices: List[str],
                 ratios: List[float],
                 stage_name: str,
                 profiler: PipelineProfiler,
                 npu_static_nodes: Optional[int] = None,
                 npu_static_edges: Optional[int] = None):
        """
        Initialize async block executor.

        Args:
            core: OpenVINO Core instance
            model_paths: {device: model_path} mapping
            devices: List of devices to use
            ratios: Data split ratios per device
            stage_name: Block/stage name for logging
            profiler: PipelineProfiler instance
            npu_static_nodes: Static node count for NPU padding
            npu_static_edges: Static edge count for NPU padding
        """
        self.core = core
        self.devices = devices
        self.ratios = ratios
        self.stage_name = stage_name
        self.profiler = profiler
        self.npu_static_nodes = npu_static_nodes
        self.npu_static_edges = npu_static_edges

        # Normalize ratios
        total = sum(ratios)
        self.ratios = [r / total for r in ratios]

        # Create executors for each device
        self.executors: Dict[str, StageExecutor] = {}
        for i, device in enumerate(devices):
            if device not in model_paths:
                print(f"  Warning: No model path for {device}")
                continue

            try:
                # Pass NPU static size for NPU devices
                executor = StageExecutor(
                    core=core,
                    model_path=model_paths[device],
                    device=device,
                    stage_name=f"{stage_name}_{device}",
                    profiler=profiler,
                    stream_id=i,
                    npu_static_nodes=npu_static_nodes if device == 'NPU' else None,
                    npu_static_edges=npu_static_edges if device == 'NPU' else None
                )
                self.executors[device] = executor
            except Exception as e:
                print(f"  Warning: Failed to create executor for {device}: {e}")

        if len(self.executors) == 0:
            raise RuntimeError("No executors created successfully")

        self.active_devices = list(self.executors.keys())
        print(f"  AsyncBlockExecutor '{stage_name}' initialized:")
        print(f"    Devices: {self.active_devices}")
        print(f"    Ratios: {self.ratios[:len(self.active_devices)]}")

    def _partition_data(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Partition data across devices based on ratios.

        Args:
            data: Input array [N, ...]

        Returns:
            {device: partition_data}
        """
        n_total = data.shape[0]
        partitions = {}

        start_idx = 0
        for i, device in enumerate(self.active_devices):
            if i == len(self.active_devices) - 1:
                # Last device gets remaining
                end_idx = n_total
            else:
                ratio = self.ratios[i] if i < len(self.ratios) else 1.0 / len(self.active_devices)
                end_idx = start_idx + int(n_total * ratio)

            partitions[device] = data[start_idx:end_idx]
            start_idx = end_idx

        return partitions

    def _merge_outputs(self, outputs_per_device: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Merge outputs from all devices.

        Args:
            outputs_per_device: {device: output_array}

        Returns:
            Merged output array
        """
        # Concatenate in device order
        arrays = [outputs_per_device[device] for device in self.active_devices
                  if device in outputs_per_device]
        return np.concatenate(arrays, axis=0)

    def run(self,
            inputs: Dict[str, np.ndarray],
            batch_id: int) -> BlockExecutionResult:
        """
        Execute block with data parallel if multiple devices.

        Args:
            inputs: Input tensors {name: data}
            batch_id: Batch identifier

        Returns:
            BlockExecutionResult with outputs and timing
        """
        wall_start = time.perf_counter()

        if len(self.active_devices) == 1:
            # Single device - no partitioning needed
            return self._run_single_device(inputs, batch_id, wall_start)
        else:
            # Data parallel across devices
            return self._run_data_parallel(inputs, batch_id, wall_start)

    def _run_single_device(self,
                           inputs: Dict[str, np.ndarray],
                           batch_id: int,
                           wall_start: float) -> BlockExecutionResult:
        """Run on single device."""
        device = self.active_devices[0]
        executor = self.executors[device]

        outputs, profiling = executor.run(inputs, batch_id)

        wall_time = (time.perf_counter() - wall_start) * 1000

        return BlockExecutionResult(
            outputs=outputs,
            wall_time_ms=wall_time,
            partition_time_ms=0.0,
            device_times={device: profiling.device_time_ms},
            merge_time_ms=0.0,
            profiling_results={device: profiling}
        )

    def _run_data_parallel(self,
                           inputs: Dict[str, np.ndarray],
                           batch_id: int,
                           wall_start: float) -> BlockExecutionResult:
        """Run data parallel across multiple devices."""

        # 1. Partition input data
        partition_start = time.perf_counter()

        # Get the main data array to partition (usually 'x' or first input)
        main_key = list(inputs.keys())[0]
        main_data = inputs[main_key]

        partitions = self._partition_data(main_data)

        # Handle other inputs (like edge_index) - might need special handling
        # For now, broadcast other inputs to all devices
        other_inputs = {k: v for k, v in inputs.items() if k != main_key}

        partition_time = (time.perf_counter() - partition_start) * 1000

        # 2. Start async inference on all devices
        async_start = time.perf_counter()

        for device in self.active_devices:
            device_inputs = {main_key: partitions[device]}
            device_inputs.update(other_inputs)
            self.executors[device].start_async(device_inputs)

        # 3. Wait for all devices and collect results
        device_times = {}
        profiling_results = {}
        outputs_per_device = {}

        for device in self.active_devices:
            outputs, profiling = self.executors[device].wait(batch_id)
            device_time = (time.perf_counter() - async_start) * 1000
            device_times[device] = device_time
            profiling_results[device] = profiling
            outputs_per_device[device] = outputs['output']

        # 4. Merge outputs
        merge_start = time.perf_counter()
        merged_output = self._merge_outputs(outputs_per_device)
        merge_time = (time.perf_counter() - merge_start) * 1000

        wall_time = (time.perf_counter() - wall_start) * 1000

        # Log partition and merge timing
        self.profiler.log_execution(
            stage_name=f"{self.stage_name}_partition",
            device="CPU",
            batch_id=batch_id,
            wall_start_ns=int(partition_start * 1e9),
            wall_end_ns=int((partition_start + partition_time / 1000) * 1e9),
            hw_duration_ms=partition_time,
            stream_id=0
        )

        self.profiler.log_execution(
            stage_name=f"{self.stage_name}_merge",
            device="CPU",
            batch_id=batch_id,
            wall_start_ns=int(merge_start * 1e9),
            wall_end_ns=int((merge_start + merge_time / 1000) * 1e9),
            hw_duration_ms=merge_time,
            stream_id=0
        )

        return BlockExecutionResult(
            outputs={'output': merged_output},
            wall_time_ms=wall_time,
            partition_time_ms=partition_time,
            device_times=device_times,
            merge_time_ms=merge_time,
            profiling_results=profiling_results
        )


class PipelineExecutor:
    """
    Multi-block pipeline executor.

    Executes a sequence of blocks according to PEP configuration.
    """

    def __init__(self,
                 pep: List,
                 model_paths: Dict[str, Dict[str, Path]],
                 profiler: PipelineProfiler,
                 npu_static_nodes: Optional[int] = None,
                 npu_static_edges: Optional[int] = None):
        """
        Initialize pipeline executor.

        Args:
            pep: PEP configuration [[devices, stages, ratios], ...]
            model_paths: {block_id: {device: model_path}}
            profiler: PipelineProfiler instance
            npu_static_nodes: Static node count for NPU padding
            npu_static_edges: Static edge count for NPU padding
        """
        self.pep = pep
        self.profiler = profiler
        self.core = Core()
        self.npu_static_nodes = npu_static_nodes
        self.npu_static_edges = npu_static_edges

        print(f"\nInitializing Pipeline Executor")
        print(f"Available devices: {self.core.available_devices}")

        # Create block executors
        self.block_executors: List[AsyncBlockExecutor] = []

        for block_id, block in enumerate(pep):
            devices = block[0]
            stages = block[1]
            ratios = block[2] if len(block) > 2 else [1.0 / len(devices)] * len(devices)

            stage_name = f"Block{block_id}_S{'_'.join(map(str, stages))}"

            if block_id not in model_paths:
                raise ValueError(f"No model paths for block {block_id}")

            executor = AsyncBlockExecutor(
                core=self.core,
                model_paths=model_paths[block_id],
                devices=devices,
                ratios=ratios,
                stage_name=stage_name,
                profiler=profiler,
                npu_static_nodes=npu_static_nodes,
                npu_static_edges=npu_static_edges
            )
            self.block_executors.append(executor)

    def run(self,
            inputs: Dict[str, np.ndarray],
            batch_id: int) -> Tuple[Dict[str, np.ndarray], List[BlockExecutionResult]]:
        """
        Execute full pipeline.

        Args:
            inputs: Initial input tensors
            batch_id: Batch identifier

        Returns:
            (final_outputs, block_results)
        """
        current_data = inputs
        block_results = []

        for block_id, executor in enumerate(self.block_executors):
            result = executor.run(current_data, batch_id)
            block_results.append(result)

            # Pass output to next block
            current_data = result.outputs

        return current_data, block_results

    def warmup(self, inputs: Dict[str, np.ndarray], iterations: int = 3):
        """Warmup all blocks."""
        print(f"\nWarming up ({iterations} iterations)...")
        for i in range(iterations):
            self.run(inputs, batch_id=-1)
        print("  Warmup complete")


if __name__ == "__main__":
    if not OPENVINO_AVAILABLE:
        print("OpenVINO not available")
        exit(0)

    from pep_config import PEP1
    from model_exporter import GNNModelExporter

    print("Testing AsyncBlockExecutor...")

    # Setup
    profiler = PipelineProfiler("AsyncTest")
    core = Core()

    # Export models for PEP1
    exporter = GNNModelExporter()
    exporter.export_for_pep(PEP1, max_nodes=12000, max_edges=120000)

    # Create model paths mapping
    model_paths = {}
    for block_id, block in enumerate(PEP1):
        devices = block[0]
        stages = block[1]
        model_paths[block_id] = {}
        for device in devices:
            path = exporter.get_model_path(stages, device)
            if path:
                model_paths[block_id][device] = path

    print(f"\nModel paths: {model_paths}")

    # Create pipeline executor
    pipeline = PipelineExecutor(PEP1, model_paths, profiler)

    # Test data
    x = np.random.randn(10000, 500).astype(np.float32)
    edge_index = np.random.randint(0, 10000, (2, 100000)).astype(np.int64)

    inputs = {'x': x, 'edge_index': edge_index}

    # Warmup
    pipeline.warmup(inputs)

    # Run test
    print("\nRunning 5 batches...")
    for batch_id in range(5):
        outputs, results = pipeline.run(inputs, batch_id)
        total_time = sum(r.wall_time_ms for r in results)
        print(f"  Batch {batch_id}: total={total_time:.2f}ms, output shape={outputs['output'].shape}")

        for i, r in enumerate(results):
            print(f"    Block {i}: wall={r.wall_time_ms:.2f}ms, devices={r.device_times}")

    # Export trace
    profiler.export_chrome_trace("test_async_executor.json")
    profiler.analyze_metrics()
