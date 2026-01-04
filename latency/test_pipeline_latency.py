"""
Pipeline Latency Testing with Flickr Dataset

Industrial-grade pipeline benchmark using:
- Flickr dataset partitioned into subgraphs (batches)
- Real 7-Stage GraphSAGE models
- True pipeline execution with cross-cycle buffer
- 1-hop Halo Node graph partitioning for data parallel
- PERF_COUNT precise timing

Usage:
    python test_pipeline_latency.py                    # Run with default PEP1
    python test_pipeline_latency.py --pep pep2         # Run with PEP2
    python test_pipeline_latency.py --export-only      # Only export models
"""

import argparse
import time
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# For running as script: add parent dir to path so 'latency' package is importable
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import from latency package
from latency.profiler import PipelineProfiler
from latency.flickr_loader import FlickrSubgraphLoader
from latency.pipeline_executor import (
    PipelineBenchmark,
    SingleDeviceStage,
    DataParallelStage,
    OPENVINO_AVAILABLE
)
from latency.pep_config import (
    ALL_PEPS, PEP1, PEP2,
    analyze_pep, print_pep
)

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


class PEPPipelineTester:
    """
    Pipeline tester for PEP configurations using Flickr data.

    Features:
    - True pipeline execution with cross-cycle buffer
    - 1-hop halo node expansion for graph data parallel
    - Per-stage, per-device timing breakdown
    """

    def __init__(self,
                 num_subgraphs: int = 8,
                 models_dir: Path = None):
        """
        Initialize tester.

        Args:
            num_subgraphs: Number of Flickr subgraphs (batches)
            models_dir: Directory for model files
        """
        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO is required for pipeline testing")

        self.num_subgraphs = num_subgraphs

        # Initialize OpenVINO
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

    def _get_available_device(self, device: str) -> str:
        """Get available device, fallback to CPU if needed."""
        if device in self.available_devices:
            return device
        print(f"  WARNING: {device} not available, using CPU")
        return "CPU"

    def _prepare_batches(self) -> List[Dict[str, np.ndarray]]:
        """Prepare all subgraphs as batches."""
        batches = []
        for i in range(self.num_subgraphs):
            sg = self.data_loader.get_subgraph(i)
            batches.append({
                'x': sg['x'].numpy().astype(np.float32),
                'edge_index': sg['edge_index'].numpy().astype(np.int64)
            })
        return batches

    def _export_models_for_pep(self, pep: List, force: bool = False):
        """Export all models needed for a PEP."""
        if self.exporter is None:
            print("  Skipping model export (exporter not available)")
            return
        self.exporter.export_for_pep(pep, self.max_nodes, self.max_edges, force)

    def _create_pipeline(self,
                         pep: List,
                         profiler: PipelineProfiler) -> PipelineBenchmark:
        """Create PipelineBenchmark from PEP configuration."""
        pipeline = PipelineBenchmark(profiler)

        for block_id, block in enumerate(pep):
            devices = block[0]
            stages = block[1]
            ratios = block[2] if len(block) > 2 else [1.0 / len(devices)] * len(devices)

            # Map to available devices
            actual_devices = [self._get_available_device(d) for d in devices]

            # Get model paths (use global max_nodes/max_edges for NPU)
            model_paths = {}
            for device in actual_devices:
                if self.exporter is not None:
                    if device == 'NPU':
                        path = self.exporter.get_model_path(
                            stages, device, self.max_nodes, self.max_edges)
                    else:
                        path = self.exporter.get_model_path(stages, device)
                else:
                    # Try to find pre-exported model
                    stage_str = '_'.join(map(str, stages))
                    if device == "NPU":
                        # New naming: stages_6_7_NPU_n{nodes}_e{edges}.xml
                        path = self.models_dir / f"stages_{stage_str}_{device}_n{self.max_nodes}_e{self.max_edges}.xml"
                    else:
                        path = self.models_dir / f"stages_{stage_str}_{device}_dynamic.xml"
                    if not path.exists():
                        path = None
                if path:
                    model_paths[device] = path

            stage_name = f"Block{block_id}_S{'_'.join(map(str, stages))}"

            # Determine if this is a graph stage (stages 1-5) or dense stage (6-7)
            is_graph_stage = any(s <= 5 for s in stages)

            if len(actual_devices) == 1:
                # Single device stage
                device = actual_devices[0]
                if device in model_paths:
                    pipeline.add_single_stage(
                        name=stage_name,
                        model_path=model_paths[device],
                        device=device,
                        npu_static_nodes=self.max_nodes if device == 'NPU' else None,
                        npu_static_edges=self.max_edges if device == 'NPU' else None
                    )
            else:
                # Data parallel stage
                pipeline.add_dp_stage(
                    name=stage_name,
                    model_paths=model_paths,
                    devices=list(model_paths.keys()),
                    ratios=ratios[:len(model_paths)],
                    is_graph_stage=is_graph_stage,
                    npu_static_nodes=self.max_nodes,
                    npu_static_edges=self.max_edges
                )

        return pipeline

    def run_pep(self,
                pep: List,
                num_iterations: int = 10,
                pep_name: str = "PEP",
                warmup: int = 3) -> Dict:
        """
        Run pipeline benchmark for a PEP configuration.

        Args:
            pep: PEP configuration
            num_iterations: Number of iterations
            pep_name: Name for logging
            warmup: Warmup iterations

        Returns:
            Benchmark results
        """
        print(f"\n{'='*70}")
        print(f"Testing {pep_name}")
        print(f"{'='*70}")
        print_pep(pep, name=pep_name)

        analysis = analyze_pep(pep)
        print(f"\nAnalysis:")
        print(f"  Blocks: {analysis['num_blocks']}")
        print(f"  Devices: {analysis['all_devices']}")
        print(f"  Data Parallel: {analysis['has_data_parallel']}")

        # 1. Export models
        print("\nExporting models...")
        self._export_models_for_pep(pep)

        # 2. Prepare batches
        print("\nPreparing batches...")
        batches = self._prepare_batches()
        print(f"  Prepared {len(batches)} batches")

        # 3. Create profiler and pipeline
        profiler = PipelineProfiler(f"{pep_name}_Pipeline")
        pipeline = self._create_pipeline(pep, profiler)

        # 4. Run benchmark
        results = pipeline.run_pipeline(
            batches=batches,
            iterations=num_iterations,
            warmup=warmup
        )

        # 5. Print report
        pipeline.print_report(results)

        # 6. Export trace
        trace_dir = Path(__file__).parent / 'traces'
        trace_dir.mkdir(exist_ok=True)
        trace_path = trace_dir / f"{pep_name.lower()}_pipeline_trace.json"
        profiler.export_chrome_trace(str(trace_path))
        print(f"\nTrace saved to: {trace_path}")

        return results

    def run_all_peps(self, num_iterations: int = 10) -> Dict:
        """Run all predefined PEP configurations."""
        all_results = {}

        for name, pep in ALL_PEPS.items():
            try:
                results = self.run_pep(pep, num_iterations, name.upper())
                all_results[name] = results
            except Exception as e:
                print(f"  ERROR running {name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[name] = {'error': str(e)}

        # Summary comparison
        self._print_comparison(all_results)

        return all_results

    def _print_comparison(self, all_results: Dict):
        """Print comparison table for multiple PEPs."""
        print(f"\n{'='*70}")
        print("PEP Comparison")
        print(f"{'='*70}")
        print(f"{'PEP':<15} {'Cycle Time (ms)':<18} {'Throughput (b/s)':<18}")
        print("-" * 70)

        for name, results in all_results.items():
            if 'error' in results:
                print(f"{name:<15} ERROR: {results['error']}")
            else:
                pipeline = results.get('pipeline', {})
                cycle_time = pipeline.get('avg_cycle_time_ms', 0)
                throughput = pipeline.get('throughput_batches_per_sec', 0)
                print(f"{name:<15} {cycle_time:<18.2f} {throughput:<18.1f}")


def main():
    parser = argparse.ArgumentParser(description="PEP Pipeline Latency Testing")
    parser.add_argument("--pep", type=str, default="pep1",
                        choices=list(ALL_PEPS.keys()) + ['all'],
                        help="PEP configuration to test")
    parser.add_argument("--num-iterations", type=int, default=10,
                        help="Number of iterations per batch")
    parser.add_argument("--num-subgraphs", type=int, default=8,
                        help="Number of Flickr subgraphs (batches)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations")
    parser.add_argument("--output", type=str, default="pipeline_results.json",
                        help="Output results file")
    parser.add_argument("--export-only", action="store_true",
                        help="Only export models, don't run benchmark")

    args = parser.parse_args()

    print("=" * 70)
    print("PEP Pipeline Latency Testing with Flickr Dataset")
    print("=" * 70)
    print(f"PEP: {args.pep}")
    print(f"Subgraphs: {args.num_subgraphs}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Output: {args.output}")

    if not OPENVINO_AVAILABLE:
        print("\nERROR: OpenVINO is required for pipeline testing")
        return

    # Create tester
    tester = PEPPipelineTester(num_subgraphs=args.num_subgraphs)

    if args.export_only:
        if not MODEL_EXPORTER_AVAILABLE:
            print("\nERROR: Model exporter requires torch_scatter. Cannot export models.")
            return
        print("\nExport-only mode: exporting models for all PEPs...")
        for name, pep in ALL_PEPS.items():
            print(f"\nExporting models for {name}...")
            tester._export_models_for_pep(pep, force=False)
        print("\nModel export complete!")
        return

    # Run tests
    if args.pep == 'all':
        results = tester.run_all_peps(args.num_iterations)
    else:
        pep = ALL_PEPS[args.pep]
        results = tester.run_pep(pep, args.num_iterations, args.pep.upper(), args.warmup)

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / args.output

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\n{'='*70}")
    print("Pipeline Benchmark Complete!")
    print(f"Results saved to: {results_path}")
    print("Open trace files in chrome://tracing or https://ui.perfetto.dev")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
