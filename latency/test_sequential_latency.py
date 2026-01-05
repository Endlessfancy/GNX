"""
Sequential Profiling for Cost Modeling

Profiles each subgraph linearly to get precise execution costs:
- Outer loop: iterate through subgraphs
- Inner loop: iterate through PEP stages sequentially
- Blocking execution (no overlap)
- Output: cost table (CSV) for offline scheduling analysis

Key: Edge Index is STATIC and must be re-attached before every stage execution,
as the previous stage's output only contains the new x.

Usage:
    python test_sequential_latency.py --pep pep1
    python test_sequential_latency.py --pep pep2 --output cost_model.csv
    python test_sequential_latency.py --pep pep1 --num-iterations 5
"""

import argparse
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# For running as script: add parent dir to path
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import from latency package
from latency.profiler import PipelineProfiler
from latency.flickr_loader import FlickrSubgraphLoader
from latency.stage_executor import StageExecutor, OPENVINO_AVAILABLE
from latency.pep_config import ALL_PEPS, PEP1, PEP2, analyze_pep, print_pep
from latency.graph_partitioner import HaloPartitioner

# Model exporter requires torch_scatter - make it optional
try:
    from latency.model_exporter import GNNModelExporter
    MODEL_EXPORTER_AVAILABLE = True
except ImportError as e:
    GNNModelExporter = None
    MODEL_EXPORTER_AVAILABLE = False
    print(f"Warning: GNNModelExporter not available: {e}")

if OPENVINO_AVAILABLE:
    from openvino.runtime import Core


@dataclass
class StageResult:
    """Result from a single stage execution."""
    stage_id: int
    stage_name: str
    wall_time_ms: float
    device_time_ms: float  # PERF_COUNT device time
    compute_time_ms: float  # Pure compute time
    device: str


@dataclass
class SubgraphResult:
    """Result from profiling a single subgraph."""
    sg_id: int
    iter_id: int
    num_nodes: int
    num_edges: int
    stage_results: List[StageResult] = field(default_factory=list)
    total_ms: float = 0.0
    partition_ms: float = 0.0  # DP partition time
    merge_ms: float = 0.0  # DP merge time


class SequentialProfiler:
    """
    Sequential profiler for cost modeling.

    Profiles each subgraph by running all stages sequentially (no pipeline overlap).
    Outputs a cost table suitable for offline scheduling analysis.
    """

    def __init__(self,
                 num_subgraphs: int = 8,
                 models_dir: Path = None):
        """
        Initialize profiler.

        Args:
            num_subgraphs: Number of Flickr subgraphs to profile
            models_dir: Directory containing model files
        """
        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO is required for sequential profiling")

        self.num_subgraphs = num_subgraphs
        self.core = Core()
        self.available_devices = self.core.available_devices
        print(f"Available devices: {self.available_devices}")

        # Load Flickr data
        print("\nLoading Flickr dataset...")
        self.data_loader = FlickrSubgraphLoader(num_subgraphs=num_subgraphs)
        self.max_nodes, self.max_edges = self.data_loader.get_max_size()
        print(f"Max subgraph size: {self.max_nodes} nodes, {self.max_edges} edges")

        # Model exporter
        self.models_dir = models_dir or Path(__file__).parent / 'models'
        if MODEL_EXPORTER_AVAILABLE:
            self.exporter = GNNModelExporter(output_dir=self.models_dir)
        else:
            self.exporter = None
            print("Warning: Model exporter not available. Use pre-exported models.")

        # Profiler for getting precise timing
        self.profiler: Optional[PipelineProfiler] = None

        # Stage executors (created per PEP)
        self.stages: List[Dict] = []  # [{name, executor, device}, ...]

    def _get_available_device(self, device: str) -> str:
        """Get available device, fallback to CPU if needed."""
        if device in self.available_devices:
            return device
        print(f"  WARNING: {device} not available, using CPU")
        return "CPU"

    def _export_models_for_pep(self, pep: List, force: bool = False):
        """Export all models needed for a PEP."""
        if self.exporter is None:
            print("  Skipping model export (exporter not available)")
            return
        self.exporter.export_for_pep(pep, self.max_nodes, self.max_edges, force)

    def _get_model_path(self, stage_ids: List[int], device: str) -> Optional[Path]:
        """Get model path for a stage/device combination."""
        if self.exporter is not None:
            if device == 'NPU':
                return self.exporter.get_model_path(
                    stage_ids, device, self.max_nodes, self.max_edges)
            else:
                return self.exporter.get_model_path(stage_ids, device)
        else:
            stage_str = '_'.join(map(str, stage_ids))
            if device == "NPU":
                model_path = self.models_dir / f"stages_{stage_str}_{device}_n{self.max_nodes}_e{self.max_edges}.xml"
            else:
                model_path = self.models_dir / f"stages_{stage_str}_{device}_dynamic.xml"
            return model_path if model_path.exists() else None

    def _create_stages(self, pep: List, profiler: PipelineProfiler) -> List[Dict]:
        """Create stage executors from PEP configuration.

        For DP blocks: creates multiple executors (one per device)
        For single-device blocks: creates one executor
        """
        stages = []

        for block_id, block in enumerate(pep):
            devices = block[0]
            stage_ids = block[1]
            ratios = block[2] if len(block) > 2 else [1.0 / len(devices)] * len(devices)

            stage_name = f"Block{block_id}_S{'_'.join(map(str, stage_ids))}"

            # Map to available devices
            actual_devices = [self._get_available_device(d) for d in devices]

            # Create executors for each device
            executors = {}
            for i, device in enumerate(actual_devices):
                model_path = self._get_model_path(stage_ids, device)
                if model_path is None:
                    print(f"  WARNING: No model for {device}, skipping")
                    continue

                executors[device] = StageExecutor(
                    core=self.core,
                    model_path=model_path,
                    device=device,
                    stage_name=f"{stage_name}_{device}",
                    profiler=profiler,
                    stream_id=i,
                    npu_static_nodes=self.max_nodes if device == 'NPU' else None,
                    npu_static_edges=self.max_edges if device == 'NPU' else None
                )
                print(f"  Created executor: {stage_name}_{device}")

            if not executors:
                print(f"  WARNING: No executors for block {block_id}, skipping")
                continue

            # Determine if this is a graph stage (needs HaloPartitioner)
            is_graph_stage = any(s <= 5 for s in stage_ids)

            stages.append({
                'id': block_id,
                'name': stage_name,
                'executors': executors,  # Dict of device -> executor
                'devices': list(executors.keys()),
                'ratios': ratios[:len(executors)],
                'stage_ids': stage_ids,
                'is_graph_stage': is_graph_stage
            })

            print(f"  Created stage: {stage_name} with devices {list(executors.keys())}")

        return stages

    def profile_subgraph(self,
                         sg_id: int,
                         x: np.ndarray,
                         edge_index: np.ndarray,
                         iter_id: int = 0) -> SubgraphResult:
        """
        Profile single subgraph through all stages sequentially.

        For DP blocks: Split data -> Run each device sequentially -> Merge

        Args:
            sg_id: Subgraph ID
            x: Node features [num_nodes, feat_dim]
            edge_index: Edge indices [2, num_edges]
            iter_id: Iteration ID for this subgraph

        Data flow:
            - Stages 1-5 (Graph ops): inputs=(x, edge_index), output=mean_agg
            - Stages 6-7 (Dense ops): inputs=(mean_agg, x_original), output=final

        Returns:
            SubgraphResult with timing for each stage/device
        """
        result = SubgraphResult(
            sg_id=sg_id,
            iter_id=iter_id,
            num_nodes=x.shape[0],
            num_edges=edge_index.shape[1]
        )

        x_original = x  # Keep original x for Stage 6-7
        current_x = x
        current_edge_index = edge_index
        total_time = 0.0

        # Variables for stages 1-4 output (used by stages 5-7)
        sum_agg = None
        count = None

        for stage in self.stages:
            stage_ids = stage['stage_ids']
            first_stage = stage_ids[0]
            devices = stage['devices']
            executors = stage['executors']
            ratios = stage['ratios']
            is_graph_stage = stage['is_graph_stage']

            stage_start = time.perf_counter()

            if len(devices) == 1:
                # Single device - simple execution
                device = devices[0]
                executor = executors[device]
                last_stage = stage_ids[-1]

                if first_stage <= 4:
                    # Stages 1-4: input is x, edge_index
                    inputs = {'x': current_x, 'edge_index': current_edge_index}
                elif first_stage == 5:
                    # Stages 5-7: input is sum_agg, count, x
                    inputs = {'sum_agg': sum_agg, 'count': count, 'x': x_original}
                else:
                    # Stages 6-7: input is mean_agg, x
                    inputs = {'mean_agg': current_x, 'x': x_original}

                outputs, profiling = executor.run(inputs, batch_id=sg_id)
                wall_time_ms = (time.perf_counter() - stage_start) * 1000

                # Handle output based on last stage
                if last_stage == 4:
                    # Stages 1-4 output sum_agg and count separately
                    sum_agg = outputs.get('sum_agg')
                    count = outputs.get('count')
                else:
                    current_x = outputs.get('output', outputs.get('x', current_x))

                result.stage_results.append(StageResult(
                    stage_id=stage['id'],
                    stage_name=stage['name'],
                    wall_time_ms=wall_time_ms,
                    device_time_ms=profiling.device_time_ms,
                    compute_time_ms=profiling.compute_time_ms,
                    device=device
                ))
            else:
                # Data Parallel - Split, Run each device sequentially, Merge
                if is_graph_stage:
                    # Graph stage: use HaloPartitioner
                    partition_start = time.perf_counter()
                    partitioner = HaloPartitioner(num_partitions=len(devices), ratios=ratios)
                    partitions = partitioner.partition(current_x, current_edge_index)
                    result.partition_ms += (time.perf_counter() - partition_start) * 1000

                    device_outputs = []
                    for i, device in enumerate(devices):
                        part = partitions[i]
                        inputs = {
                            'x': part.x_local.astype(np.float32),
                            'edge_index': part.edge_index_local.astype(np.int64)
                        }

                        dev_start = time.perf_counter()
                        outputs, profiling = executors[device].run(inputs, batch_id=sg_id)
                        dev_wall_ms = (time.perf_counter() - dev_start) * 1000

                        device_outputs.append((outputs, part))

                        result.stage_results.append(StageResult(
                            stage_id=stage['id'],
                            stage_name=f"{stage['name']}_{device}",
                            wall_time_ms=dev_wall_ms,
                            device_time_ms=profiling.device_time_ms,
                            compute_time_ms=profiling.compute_time_ms,
                            device=device
                        ))

                    # Merge outputs using partitioner's merge_outputs method
                    merge_start = time.perf_counter()
                    last_stage = stage_ids[-1]

                    if last_stage == 4:
                        # Stages 1-4 输出 sum_agg 和 count，需要分别合并
                        sum_agg_list = [out.get('sum_agg') for out, _ in device_outputs]
                        count_list = [out.get('count') for out, _ in device_outputs]
                        sum_agg = partitioner.merge_outputs(sum_agg_list, partitions, x.shape[0])
                        count = partitioner.merge_outputs(count_list, partitions, x.shape[0])
                    else:
                        outputs_list = [out.get('output', out.get('x')) for out, _ in device_outputs]
                        current_x = partitioner.merge_outputs(outputs_list, partitions, x.shape[0])

                    result.merge_ms += (time.perf_counter() - merge_start) * 1000

                else:
                    # Dense stage: simple split by node count
                    last_stage = stage_ids[-1]

                    # Determine which tensor to use for size calculation
                    if first_stage == 5:
                        n_total = sum_agg.shape[0]
                    else:
                        n_total = current_x.shape[0]

                    merged_parts = []
                    node_start = 0

                    for i, device in enumerate(devices):
                        if i == len(devices) - 1:
                            n_part = n_total - node_start
                        else:
                            n_part = int(n_total * ratios[i])

                        # Build inputs based on first_stage
                        if first_stage == 5:
                            # Stages 5-7: need sum_agg, count, x slices
                            inputs = {
                                'sum_agg': sum_agg[node_start:node_start + n_part],
                                'count': count[node_start:node_start + n_part],
                                'x': x_original[node_start:node_start + n_part]
                            }
                        else:
                            # Stages 6-7: need mean_agg slice and x_original slice
                            inputs = {
                                'mean_agg': current_x[node_start:node_start + n_part],
                                'x': x_original[node_start:node_start + n_part]
                            }

                        dev_start = time.perf_counter()
                        outputs, profiling = executors[device].run(inputs, batch_id=sg_id)
                        dev_wall_ms = (time.perf_counter() - dev_start) * 1000

                        merged_parts.append(outputs.get('output', outputs.get('x')))

                        result.stage_results.append(StageResult(
                            stage_id=stage['id'],
                            stage_name=f"{stage['name']}_{device}",
                            wall_time_ms=dev_wall_ms,
                            device_time_ms=profiling.device_time_ms,
                            compute_time_ms=profiling.compute_time_ms,
                            device=device
                        ))

                        node_start += n_part

                    merge_start = time.perf_counter()
                    current_x = np.concatenate(merged_parts, axis=0)
                    result.merge_ms += (time.perf_counter() - merge_start) * 1000

            stage_wall_ms = (time.perf_counter() - stage_start) * 1000
            total_time += stage_wall_ms

        result.total_ms = total_time
        return result

    def run_profiling(self,
                      pep: List,
                      num_iterations: int = 1,
                      warmup: int = 1,
                      pep_name: str = "PEP") -> pd.DataFrame:
        """
        Profile all subgraphs with given PEP configuration.

        Args:
            pep: PEP configuration
            num_iterations: Number of iterations per subgraph
            warmup: Warmup iterations (not recorded)
            pep_name: Name for logging

        Returns:
            DataFrame with cost table
        """
        print(f"\n{'='*70}")
        print(f"Sequential Profiling: {pep_name}")
        print(f"{'='*70}")
        print_pep(pep, name=pep_name)

        # Export models
        print("\nExporting models...")
        self._export_models_for_pep(pep)

        # Create profiler for precise timing
        self.profiler = PipelineProfiler(f"{pep_name}_Sequential")

        # Create stages with profiler
        print("\nCreating stage executors...")
        self.stages = self._create_stages(pep, self.profiler)

        if len(self.stages) == 0:
            print("ERROR: No stages created")
            return pd.DataFrame()

        print(f"\nProfiling {self.num_subgraphs} subgraphs x {num_iterations} iterations")
        print(f"Warmup: {warmup} iterations")

        results = []

        for sg_id in range(self.num_subgraphs):
            # Load subgraph data
            sg = self.data_loader.get_subgraph(sg_id)
            x = sg['x'].numpy().astype(np.float32)
            edge_index = sg['edge_index'].numpy().astype(np.int64)

            # Warmup runs (not recorded)
            for _ in range(warmup):
                self.profile_subgraph(sg_id, x, edge_index, iter_id=-1)

            # Measurement runs
            for iter_id in range(num_iterations):
                result = self.profile_subgraph(sg_id, x, edge_index, iter_id)
                results.append(result)

                # Progress output
                if iter_id == 0:
                    stage_times = ", ".join([f"S{r.stage_id}:{r.wall_time_ms:.2f}ms"
                                             for r in result.stage_results])
                    print(f"  Subgraph {sg_id}: {result.num_nodes} nodes, "
                          f"{result.num_edges} edges -> total={result.total_ms:.2f}ms ({stage_times})")

        # Convert to DataFrame
        df = self._results_to_dataframe(results)
        return df

    def _results_to_dataframe(self, results: List[SubgraphResult]) -> pd.DataFrame:
        """Convert results to DataFrame cost table.

        For DP stages, creates separate columns for each device:
        - stage0_CPU_wall_ms, stage0_GPU_wall_ms, etc.
        """
        rows = []

        for result in results:
            row = {
                'sg_id': result.sg_id,
                'iter': result.iter_id,
                'num_nodes': result.num_nodes,
                'num_edges': result.num_edges,
            }

            # Add per-stage/device times (wall, device, compute)
            for sr in result.stage_results:
                # Use stage_id + device for column naming
                prefix = f"stage{sr.stage_id}_{sr.device}"
                row[f'{prefix}_wall_ms'] = sr.wall_time_ms
                row[f'{prefix}_device_ms'] = sr.device_time_ms
                row[f'{prefix}_compute_ms'] = sr.compute_time_ms

            row['partition_ms'] = result.partition_ms
            row['merge_ms'] = result.merge_ms
            row['total_ms'] = result.total_ms
            rows.append(row)

        return pd.DataFrame(rows)

    def print_statistics(self, df: pd.DataFrame):
        """Print statistics from cost table."""
        print(f"\n{'='*70}")
        print("Statistics (PERF_COUNT Profiling Results)")
        print(f"{'='*70}")

        # Find unique stage_device combinations (e.g., stage0_CPU, stage0_GPU, stage1_NPU)
        stage_devices = set()
        for col in df.columns:
            if col.startswith('stage') and '_wall_ms' in col:
                # Extract "stage0_CPU" from "stage0_CPU_wall_ms"
                prefix = col.replace('_wall_ms', '')
                stage_devices.add(prefix)

        # Per-stage/device statistics
        for prefix in sorted(stage_devices):
            # Extract stage_id and device from prefix like "stage0_CPU"
            parts = prefix.split('_', 1)
            stage_id = parts[0].replace('stage', '')
            device = parts[1] if len(parts) > 1 else 'Unknown'

            print(f"\n  {prefix} ({device}):")

            for time_type in ['wall', 'device', 'compute']:
                col = f'{prefix}_{time_type}_ms'
                if col in df.columns:
                    avg = df[col].mean()
                    std = df[col].std()
                    print(f"    {time_type:8}: avg={avg:7.2f}ms, std={std:5.2f}ms")

        # Partition and Merge statistics (DP overhead)
        if 'partition_ms' in df.columns and df['partition_ms'].sum() > 0:
            print(f"\n  DP Overhead:")
            print(f"    partition: avg={df['partition_ms'].mean():.2f}ms, std={df['partition_ms'].std():.2f}ms")
            print(f"    merge    : avg={df['merge_ms'].mean():.2f}ms, std={df['merge_ms'].std():.2f}ms")

        # Total statistics
        print(f"\n  Total (wall time):")
        print(f"    avg={df['total_ms'].mean():.2f}ms, "
              f"std={df['total_ms'].std():.2f}ms, "
              f"min={df['total_ms'].min():.2f}ms, "
              f"max={df['total_ms'].max():.2f}ms")

        # Throughput
        avg_total = df['total_ms'].mean()
        if avg_total > 0:
            throughput = 1000.0 / avg_total
            print(f"\n  Throughput: {throughput:.2f} subgraphs/sec")


def main():
    parser = argparse.ArgumentParser(description="Sequential Profiling for Cost Modeling")
    parser.add_argument("--pep", type=str, default="pep1",
                        choices=list(ALL_PEPS.keys()),
                        help="PEP configuration to profile")
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="Number of iterations per subgraph")
    parser.add_argument("--num-subgraphs", type=int, default=8,
                        help="Number of Flickr subgraphs to profile")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup iterations")
    parser.add_argument("--output", type=str, default="cost_model.csv",
                        help="Output CSV file")

    args = parser.parse_args()

    print("=" * 70)
    print("Sequential Profiling for Cost Modeling")
    print("=" * 70)
    print(f"PEP: {args.pep}")
    print(f"Subgraphs: {args.num_subgraphs}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"Output: {args.output}")

    if not OPENVINO_AVAILABLE:
        print("\nERROR: OpenVINO is required for profiling")
        return

    # Create profiler
    profiler = SequentialProfiler(num_subgraphs=args.num_subgraphs)

    # Run profiling
    pep = ALL_PEPS[args.pep]
    df = profiler.run_profiling(
        pep=pep,
        num_iterations=args.num_iterations,
        warmup=args.warmup,
        pep_name=args.pep.upper()
    )

    if df.empty:
        print("\nERROR: No results generated")
        return

    # Print statistics
    profiler.print_statistics(df)

    # Save to CSV
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / args.output

    df.to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print(f"Cost table saved to: {output_path}")
    print(f"{'='*70}")

    # Also print the table
    print("\n[Cost Table Preview]")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
