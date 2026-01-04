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

    def _create_stages(self, pep: List) -> List[Dict]:
        """Create stage executors from PEP configuration."""
        stages = []

        for block_id, block in enumerate(pep):
            devices = block[0]
            stage_ids = block[1]

            # For sequential profiling, use first device only (no data parallel)
            device = self._get_available_device(devices[0])

            # Get model path
            if self.exporter is not None:
                model_path = self.exporter.get_model_path(stage_ids, device)
            else:
                stage_str = '_'.join(map(str, stage_ids))
                shape_type = "static" if device == "NPU" else "dynamic"
                model_path = self.models_dir / f"stages_{stage_str}_{shape_type}.xml"
                if not model_path.exists():
                    model_path = None

            if model_path is None:
                print(f"  WARNING: No model found for block {block_id}, skipping")
                continue

            stage_name = f"Block{block_id}_S{'_'.join(map(str, stage_ids))}"

            # Create executor
            executor = StageExecutor(
                core=self.core,
                model_path=model_path,
                device=device,
                stage_name=stage_name,
                profiler=None,  # No profiling trace needed
                npu_static_nodes=self.max_nodes if device == 'NPU' else None,
                npu_static_edges=self.max_edges if device == 'NPU' else None
            )

            stages.append({
                'id': block_id,
                'name': stage_name,
                'executor': executor,
                'device': device,
                'stage_ids': stage_ids
            })

            print(f"  Created stage: {stage_name} on {device}")

        return stages

    def profile_subgraph(self,
                         sg_id: int,
                         x: np.ndarray,
                         edge_index: np.ndarray,
                         iter_id: int = 0) -> SubgraphResult:
        """
        Profile single subgraph through all stages sequentially.

        Args:
            sg_id: Subgraph ID
            x: Node features [num_nodes, feat_dim]
            edge_index: Edge indices [2, num_edges] - STATIC, reused every stage
            iter_id: Iteration ID for this subgraph

        Returns:
            SubgraphResult with timing for each stage
        """
        result = SubgraphResult(
            sg_id=sg_id,
            iter_id=iter_id,
            num_nodes=x.shape[0],
            num_edges=edge_index.shape[1]
        )

        current_x = x
        total_time = 0.0

        for stage in self.stages:
            # CRITICAL: Re-attach edge_index before every stage
            # Previous stage output only contains new x, not edge_index
            inputs = {
                'x': current_x,
                'edge_index': edge_index  # Static, re-attached every time
            }

            # Synchronous execution for pure latency measurement
            start = time.perf_counter()
            outputs, _ = stage['executor'].run(inputs, batch_id=sg_id)
            wall_time_ms = (time.perf_counter() - start) * 1000

            # Update x for next stage (edge_index stays the same)
            current_x = outputs.get('output', outputs.get('x', current_x))

            # Record stage timing
            stage_result = StageResult(
                stage_id=stage['id'],
                stage_name=stage['name'],
                wall_time_ms=wall_time_ms,
                device=stage['device']
            )
            result.stage_results.append(stage_result)
            total_time += wall_time_ms

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

        # Create stages
        print("\nCreating stage executors...")
        self.stages = self._create_stages(pep)

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
        """Convert results to DataFrame cost table."""
        rows = []

        for result in results:
            row = {
                'sg_id': result.sg_id,
                'iter': result.iter_id,
                'num_nodes': result.num_nodes,
                'num_edges': result.num_edges,
            }

            # Add per-stage times
            for sr in result.stage_results:
                row[f'stage{sr.stage_id}_ms'] = sr.wall_time_ms
                row[f'stage{sr.stage_id}_device'] = sr.device

            row['total_ms'] = result.total_ms
            rows.append(row)

        return pd.DataFrame(rows)

    def print_statistics(self, df: pd.DataFrame):
        """Print statistics from cost table."""
        print(f"\n{'='*70}")
        print("Statistics")
        print(f"{'='*70}")

        # Per-stage statistics
        stage_cols = [c for c in df.columns if c.endswith('_ms') and c != 'total_ms']

        for col in stage_cols:
            avg = df[col].mean()
            std = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"  {col}: avg={avg:.2f}ms, std={std:.2f}ms, min={min_val:.2f}ms, max={max_val:.2f}ms")

        # Total statistics
        print(f"\n  Total: avg={df['total_ms'].mean():.2f}ms, "
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
