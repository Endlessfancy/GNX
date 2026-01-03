"""
PEP-based Latency Testing with Flickr Dataset

Test GNN pipeline latency using:
- Flickr dataset partitioned into subgraphs
- Real 7-Stage GraphSAGE models (fused kernels)
- Multi-device async parallel execution
- PERF_COUNT precise timing

Usage:
    python test_pep_latency.py                    # Run with default PEP1
    python test_pep_latency.py --pep pep2         # Run with PEP2
    python test_pep_latency.py --pep all          # Run all PEPs
    python test_pep_latency.py --export-only      # Only export models
"""

import argparse
import time
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Ensure latency directory is in path (for running as script)
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from profiler import PipelineProfiler
from flickr_loader import FlickrSubgraphLoader
from model_exporter import GNNModelExporter
from stage_executor import StageExecutor, ProfilingResult, OPENVINO_AVAILABLE
from async_executor import AsyncBlockExecutor, BlockExecutionResult
from pep_config import (
    ALL_PEPS, PEP1, PEP2,
    analyze_pep, print_pep
)

if OPENVINO_AVAILABLE:
    from openvino.runtime import Core


class PEPLatencyTester:
    """
    Test latency for PEP configurations using Flickr data.

    Features:
    - Loads real Flickr data partitioned into subgraphs
    - Uses fused 7-Stage GraphSAGE models
    - Supports single and data-parallel execution
    - Precise timing with PERF_COUNT
    """

    def __init__(self,
                 profiler: PipelineProfiler,
                 num_subgraphs: int = 8,
                 models_dir: Path = None):
        """
        Initialize tester.

        Args:
            profiler: PipelineProfiler instance
            num_subgraphs: Number of Flickr subgraphs
            models_dir: Directory for model files
        """
        self.profiler = profiler
        self.num_subgraphs = num_subgraphs

        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO is required for PEP latency testing")

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
        self.exporter = GNNModelExporter(output_dir=self.models_dir)

        # Cache for block executors
        self._executor_cache: Dict[str, AsyncBlockExecutor] = {}

    def _get_available_device(self, device: str) -> str:
        """Get available device, fallback to CPU if needed."""
        if device in self.available_devices:
            return device
        print(f"  WARNING: {device} not available, using CPU")
        return "CPU"

    def _export_models_for_pep(self, pep: List, force: bool = False) -> Dict[int, Dict[str, Path]]:
        """
        Export all models needed for a PEP.

        Returns:
            {block_id: {device: model_path}}
        """
        self.exporter.export_for_pep(pep, self.max_nodes, self.max_edges, force)

        model_paths = {}
        for block_id, block in enumerate(pep):
            devices = block[0]
            stages = block[1]
            model_paths[block_id] = {}

            for device in devices:
                actual_device = self._get_available_device(device)
                path = self.exporter.get_model_path(stages, actual_device)
                if path:
                    model_paths[block_id][actual_device] = path
                else:
                    print(f"  WARNING: Model not found for block {block_id}, device {actual_device}")

        return model_paths

    def _create_block_executor(self,
                               block_id: int,
                               block: List,
                               model_paths: Dict[str, Path]) -> AsyncBlockExecutor:
        """Create executor for a PEP block."""
        devices = block[0]
        stages = block[1]
        ratios = block[2] if len(block) > 2 else [1.0 / len(devices)] * len(devices)

        # Map to available devices
        actual_devices = [self._get_available_device(d) for d in devices]
        actual_model_paths = {
            self._get_available_device(d): model_paths.get(self._get_available_device(d))
            for d in devices
            if model_paths.get(self._get_available_device(d))
        }

        stage_name = f"Block{block_id}_S{'_'.join(map(str, stages))}"

        # Pass NPU static size for NPU padding
        executor = AsyncBlockExecutor(
            core=self.core,
            model_paths=actual_model_paths,
            devices=list(actual_model_paths.keys()),
            ratios=ratios[:len(actual_model_paths)],
            stage_name=stage_name,
            profiler=self.profiler,
            npu_static_nodes=self.max_nodes,
            npu_static_edges=self.max_edges
        )

        return executor

    def run_pep(self,
                pep: List,
                num_iterations: int = 10,
                pep_name: str = "PEP",
                warmup: int = 3) -> Dict:
        """
        Run latency test for a single PEP.

        Args:
            pep: PEP configuration
            num_iterations: Number of iterations per subgraph
            pep_name: Name for logging
            warmup: Number of warmup iterations

        Returns:
            Timing statistics
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
        model_paths = self._export_models_for_pep(pep)

        # 2. Create block executors
        print("\nCreating block executors...")
        block_executors = []
        for block_id, block in enumerate(pep):
            executor = self._create_block_executor(block_id, block, model_paths[block_id])
            block_executors.append(executor)

        # 3. Warmup
        print(f"\nWarming up ({warmup} iterations)...")
        sg0 = self.data_loader.get_subgraph(0)
        warmup_inputs = {
            'x': sg0['x'].numpy().astype(np.float32),
            'edge_index': sg0['edge_index'].numpy().astype(np.int64)
        }

        for _ in range(warmup):
            current = warmup_inputs
            for executor in block_executors:
                result = executor.run(current, batch_id=-1)
                current = result.outputs

        # 4. Run benchmark
        print(f"\nRunning {num_iterations} iterations on {self.num_subgraphs} subgraphs...")

        all_times = []
        per_block_times = [[] for _ in range(len(pep))]
        per_subgraph_times = []

        batch_id = 0
        for sg_id in range(self.num_subgraphs):
            sg_data = self.data_loader.get_subgraph(sg_id)
            inputs = {
                'x': sg_data['x'].numpy().astype(np.float32),
                'edge_index': sg_data['edge_index'].numpy().astype(np.int64)
            }

            sg_times = []

            for iter_id in range(num_iterations):
                iter_start = time.perf_counter()

                # Execute all blocks
                current = inputs
                block_times = []

                for block_id, executor in enumerate(block_executors):
                    result = executor.run(current, batch_id)
                    block_times.append(result.wall_time_ms)
                    per_block_times[block_id].append(result.wall_time_ms)
                    current = result.outputs

                iter_time = (time.perf_counter() - iter_start) * 1000
                all_times.append(iter_time)
                sg_times.append(iter_time)
                batch_id += 1

            sg_avg = np.mean(sg_times)
            per_subgraph_times.append(sg_avg)

            if sg_id % max(1, self.num_subgraphs // 4) == 0:
                print(f"  Subgraph {sg_id}: avg={sg_avg:.2f}ms, "
                      f"nodes={sg_data['num_nodes']}, edges={sg_data['num_edges']}")

        # 5. Calculate statistics
        stats = {
            'pep_name': pep_name,
            'num_subgraphs': self.num_subgraphs,
            'num_iterations': num_iterations,
            'total_batches': batch_id,
            'avg_time_ms': np.mean(all_times),
            'std_time_ms': np.std(all_times),
            'min_time_ms': np.min(all_times),
            'max_time_ms': np.max(all_times),
            'p50_time_ms': np.percentile(all_times, 50),
            'p95_time_ms': np.percentile(all_times, 95),
            'p99_time_ms': np.percentile(all_times, 99),
            'per_block_avg_ms': [np.mean(times) for times in per_block_times],
            'per_subgraph_avg_ms': per_subgraph_times,
            'throughput_batches_per_sec': 1000.0 / np.mean(all_times)
        }

        # Print results
        print(f"\n--- {pep_name} Results ---")
        print(f"  Total batches: {stats['total_batches']}")
        print(f"  Avg Time: {stats['avg_time_ms']:.2f} +/- {stats['std_time_ms']:.2f} ms")
        print(f"  Min/Max: {stats['min_time_ms']:.2f} / {stats['max_time_ms']:.2f} ms")
        print(f"  P50/P95/P99: {stats['p50_time_ms']:.2f} / {stats['p95_time_ms']:.2f} / {stats['p99_time_ms']:.2f} ms")
        print(f"  Throughput: {stats['throughput_batches_per_sec']:.1f} batches/sec")
        print(f"  Per-Block Avg: {[f'{t:.2f}ms' for t in stats['per_block_avg_ms']]}")

        return stats

    def run_all_peps(self, num_iterations: int = 10) -> Dict:
        """Run all predefined PEP configurations."""
        results = {}

        for name, pep in ALL_PEPS.items():
            try:
                stats = self.run_pep(pep, num_iterations, name.upper())
                results[name] = stats
            except Exception as e:
                print(f"  ERROR running {name}: {e}")
                import traceback
                traceback.print_exc()
                results[name] = {'error': str(e)}

        # Summary comparison
        self._print_comparison(results)

        return results

    def _print_comparison(self, results: Dict):
        """Print comparison table for multiple PEPs."""
        print(f"\n{'='*70}")
        print("PEP Comparison")
        print(f"{'='*70}")
        print(f"{'PEP':<15} {'Avg (ms)':<12} {'Std (ms)':<12} {'P95 (ms)':<12} {'Throughput':<12}")
        print("-" * 70)

        for name, stats in results.items():
            if 'error' in stats:
                print(f"{name:<15} ERROR: {stats['error']}")
            else:
                print(f"{name:<15} "
                      f"{stats['avg_time_ms']:<12.2f} "
                      f"{stats['std_time_ms']:<12.2f} "
                      f"{stats['p95_time_ms']:<12.2f} "
                      f"{stats['throughput_batches_per_sec']:<12.1f}")


def main():
    parser = argparse.ArgumentParser(description="PEP-based Latency Testing with Flickr")
    parser.add_argument("--pep", type=str, default="pep1",
                        choices=list(ALL_PEPS.keys()) + ['all'],
                        help="PEP configuration to test")
    parser.add_argument("--num-iterations", type=int, default=10,
                        help="Number of iterations per subgraph")
    parser.add_argument("--num-subgraphs", type=int, default=8,
                        help="Number of Flickr subgraphs")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations")
    parser.add_argument("--output", type=str, default="pep_latency_trace.json",
                        help="Output trace file")
    parser.add_argument("--export-only", action="store_true",
                        help="Only export models, don't run benchmark")

    args = parser.parse_args()

    print("=" * 70)
    print("PEP-based Latency Testing with Flickr Dataset")
    print("=" * 70)
    print(f"PEP: {args.pep}")
    print(f"Subgraphs: {args.num_subgraphs}")
    print(f"Iterations per subgraph: {args.num_iterations}")
    print(f"Output: {args.output}")

    if not OPENVINO_AVAILABLE:
        print("\nERROR: OpenVINO is required for PEP latency testing")
        return

    # Create profiler
    profiler = PipelineProfiler("PEP_Latency_Test")

    # Create tester
    tester = PEPLatencyTester(
        profiler=profiler,
        num_subgraphs=args.num_subgraphs
    )

    if args.export_only:
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

    # Export trace and analyze
    trace_dir = Path(__file__).parent / 'traces'
    trace_dir.mkdir(exist_ok=True)
    trace_path = trace_dir / args.output

    profiler.export_chrome_trace(str(trace_path))
    profiler.analyze_metrics()

    # Save results to JSON
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"{args.pep}_results.json"

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
    print("Test Complete!")
    print(f"Trace saved to: {trace_path}")
    print(f"Results saved to: {results_path}")
    print("Open trace in chrome://tracing or https://ui.perfetto.dev")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
