"""
PEP-based Latency Testing

Test GNN pipeline latency using the same PEP configurations as the executer.

Usage:
    python test_pep_latency.py                    # Run with default PEP1
    python test_pep_latency.py --pep pep2         # Run with PEP2
    python test_pep_latency.py --pep all          # Run all PEPs
    python test_pep_latency.py --two-pep          # Run two-PEP test plan
"""

import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from profiler import PipelineProfiler
from stage_executor import StageExecutor, create_dummy_model, OPENVINO_AVAILABLE
from pep_config import (
    ALL_PEPS, PEP1, PEP2, PEP_CPU_ONLY, PEP_GPU_ONLY,
    get_two_pep_test_plan, get_single_pep_test_plan,
    analyze_pep, print_pep, print_execution_plan
)

try:
    from data_parallel_stage import DataParallelStage, DataParallelStageAsync
    DATA_PARALLEL_AVAILABLE = True
except ImportError:
    DATA_PARALLEL_AVAILABLE = False

try:
    import openvino as ov
    from openvino.runtime import Core
except ImportError:
    pass


class PEPLatencyTester:
    """
    Test latency for PEP configurations.

    Simulates the GNN pipeline execution with precise timing for each block.
    """

    # Stage input/output dimensions
    STAGE_DIMS = {
        # stage_id: (input_features, output_features)
        1: (500, 500),   # Gather: x -> x_j
        2: (500, 500),   # Message: x_j -> messages (identity)
        3: (500, 500),   # ReduceSum: messages -> sum_agg
        4: (500, 1),     # ReduceCount: -> count
        5: (500, 500),   # Normalize: sum_agg, count -> mean_agg
        6: (500, 256),   # Transform: mean_agg, x -> out
        7: (256, 256),   # Activate: out -> activated
    }

    def __init__(self, profiler: PipelineProfiler, num_nodes: int = 1000):
        """
        Initialize tester.

        Args:
            profiler: PipelineProfiler instance
            num_nodes: Number of nodes per batch
        """
        self.profiler = profiler
        self.num_nodes = num_nodes

        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO is required for PEP latency testing")

        self.core = Core()
        self.available_devices = self.core.available_devices
        print(f"Available devices: {self.available_devices}")

        # Cache for compiled models
        self._model_cache: Dict[str, StageExecutor] = {}

    def _get_device(self, device: str) -> str:
        """Get available device, fallback to CPU if needed."""
        if device in self.available_devices:
            return device
        print(f"  WARNING: {device} not available, using CPU")
        return "CPU"

    def _create_block_model(self, stages: List[int]) -> 'ov.Model':
        """
        Create a dummy model for a block of stages.

        Args:
            stages: List of stage IDs

        Returns:
            OpenVINO Model
        """
        first_stage = stages[0]
        last_stage = stages[-1]

        # Determine input/output dimensions
        in_dim = self.STAGE_DIMS[first_stage][0]
        out_dim = self.STAGE_DIMS[last_stage][1]

        # More compute ops for more stages
        compute_ops = max(1, len(stages))

        return create_dummy_model(
            (self.num_nodes, in_dim),
            (self.num_nodes, out_dim),
            compute_ops=compute_ops
        )

    def _get_executor(self, block_id: int, device: str, stages: List[int],
                      stream_id: int = 0) -> StageExecutor:
        """
        Get or create executor for a block.

        Args:
            block_id: Block ID
            device: Target device
            stages: Stages in this block
            stream_id: Stream ID for data parallel

        Returns:
            StageExecutor instance
        """
        cache_key = f"block{block_id}_{device}_s{stream_id}_stages{'_'.join(map(str, stages))}"

        if cache_key not in self._model_cache:
            model = self._create_block_model(stages)
            actual_device = self._get_device(device)

            executor = StageExecutor(
                core=self.core,
                model=model,
                device=actual_device,
                stage_name=f"Block{block_id}_S{stages[0]}-{stages[-1]}",
                profiler=self.profiler,
                stream_id=stream_id
            )
            self._model_cache[cache_key] = executor

        return self._model_cache[cache_key]

    def run_pep(self, pep: List, num_batches: int = 10,
                pep_name: str = "PEP") -> Dict:
        """
        Run latency test for a single PEP.

        Args:
            pep: PEP configuration
            num_batches: Number of batches to run
            pep_name: Name for logging

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

        # Create executors for each block
        block_executors = []
        for block_id, block in enumerate(pep):
            devices = block[0]
            stages = block[1]
            ratios = block[2] if len(block) > 2 else [1.0 / len(devices)] * len(devices)

            if len(devices) == 1:
                # Single device
                executor = self._get_executor(block_id, devices[0], stages)
                block_executors.append({
                    'type': 'single',
                    'executor': executor,
                    'stages': stages,
                })
            else:
                # Data parallel
                if DATA_PARALLEL_AVAILABLE:
                    dp_executors = []
                    for i, device in enumerate(devices):
                        exec = self._get_executor(block_id, device, stages, stream_id=i)
                        dp_executors.append(exec)

                    dp_stage = DataParallelStageAsync(
                        name=f"Block{block_id}_DP",
                        executors=dp_executors,
                        ratios=ratios,
                        profiler=self.profiler
                    )
                    block_executors.append({
                        'type': 'data_parallel',
                        'executor': dp_stage,
                        'stages': stages,
                        'devices': devices,
                        'ratios': ratios,
                    })
                else:
                    # Fallback: use first device only
                    print(f"  WARNING: DataParallel not available, using {devices[0]} only")
                    executor = self._get_executor(block_id, devices[0], stages)
                    block_executors.append({
                        'type': 'single',
                        'executor': executor,
                        'stages': stages,
                    })

        # Run batches
        print(f"\nRunning {num_batches} batches...")
        batch_times = []
        block_times_all = [[] for _ in range(len(pep))]

        for batch_id in range(num_batches):
            batch_start = time.perf_counter()

            # Initial input
            current_data = np.random.randn(self.num_nodes, 500).astype(np.float32)

            # Execute each block
            for block_id, block_exec in enumerate(block_executors):
                if block_exec['type'] == 'single':
                    outputs, hw_time = block_exec['executor'].run(
                        {'input': current_data}, batch_id
                    )
                    current_data = outputs['output']
                    block_times_all[block_id].append(hw_time)

                elif block_exec['type'] == 'data_parallel':
                    outputs, timing = block_exec['executor'].run(
                        {'input': current_data}, batch_id
                    )
                    current_data = outputs['output']
                    block_times_all[block_id].append(timing['stage_total_ms'])

            batch_end = time.perf_counter()
            batch_time = (batch_end - batch_start) * 1000
            batch_times.append(batch_time)

            if batch_id % max(1, num_batches // 5) == 0:
                print(f"  Batch {batch_id}: {batch_time:.2f}ms")

        # Shutdown data parallel executors
        for block_exec in block_executors:
            if block_exec['type'] == 'data_parallel' and hasattr(block_exec['executor'], 'shutdown'):
                block_exec['executor'].shutdown()

        # Calculate statistics
        stats = {
            'pep_name': pep_name,
            'num_batches': num_batches,
            'avg_batch_time_ms': np.mean(batch_times),
            'min_batch_time_ms': np.min(batch_times),
            'max_batch_time_ms': np.max(batch_times),
            'std_batch_time_ms': np.std(batch_times),
            'block_avg_times_ms': [np.mean(times) for times in block_times_all],
        }

        print(f"\n--- {pep_name} Results ---")
        print(f"  Avg Batch Time: {stats['avg_batch_time_ms']:.2f} ms")
        print(f"  Min/Max: {stats['min_batch_time_ms']:.2f} / {stats['max_batch_time_ms']:.2f} ms")
        print(f"  Std Dev: {stats['std_batch_time_ms']:.2f} ms")
        print(f"  Per-Block Avg Times: {[f'{t:.2f}ms' for t in stats['block_avg_times_ms']]}")

        return stats

    def run_two_pep_test(self, num_batches: int = 10) -> Dict:
        """
        Run the standard two-PEP test plan.

        Returns:
            Combined statistics
        """
        print("\n" + "=" * 70)
        print("Two-PEP Test Plan")
        print("=" * 70)

        plan = get_two_pep_test_plan()
        print_execution_plan(plan)

        results = {}

        for cluster in plan['clusters']:
            pep_name = cluster['pep_key']
            pep = cluster['pep']
            subgraphs = cluster['subgraph_ids']

            print(f"\n>>> Testing Cluster '{pep_name}' (subgraphs {subgraphs})")

            stats = self.run_pep(pep, num_batches, pep_name)
            results[pep_name] = stats

        # Summary
        print("\n" + "=" * 70)
        print("Two-PEP Test Summary")
        print("=" * 70)

        for name, stats in results.items():
            print(f"\n{name}:")
            print(f"  Avg Batch Time: {stats['avg_batch_time_ms']:.2f} ms")

        return results

    def run_all_peps(self, num_batches: int = 10) -> Dict:
        """
        Run all predefined PEP configurations.

        Returns:
            Dictionary of results
        """
        results = {}

        for name, pep in ALL_PEPS.items():
            try:
                stats = self.run_pep(pep, num_batches, name.upper())
                results[name] = stats
            except Exception as e:
                print(f"  ERROR running {name}: {e}")
                results[name] = {'error': str(e)}

        # Summary comparison
        print("\n" + "=" * 70)
        print("All PEPs Comparison")
        print("=" * 70)
        print(f"{'PEP':<15} {'Avg Time (ms)':<15} {'Min-Max (ms)':<20}")
        print("-" * 50)

        for name, stats in results.items():
            if 'error' in stats:
                print(f"{name:<15} ERROR: {stats['error']}")
            else:
                avg = stats['avg_batch_time_ms']
                min_t = stats['min_batch_time_ms']
                max_t = stats['max_batch_time_ms']
                print(f"{name:<15} {avg:<15.2f} {min_t:.2f} - {max_t:.2f}")

        return results


def main():
    parser = argparse.ArgumentParser(description="PEP-based Latency Testing")
    parser.add_argument("--pep", type=str, default="pep1",
                        choices=list(ALL_PEPS.keys()) + ['all'],
                        help="PEP configuration to test")
    parser.add_argument("--two-pep", action="store_true",
                        help="Run two-PEP test plan")
    parser.add_argument("--num-batches", type=int, default=10,
                        help="Number of batches to run")
    parser.add_argument("--num-nodes", type=int, default=1000,
                        help="Number of nodes per batch")
    parser.add_argument("--output", type=str, default="pep_latency_trace.json",
                        help="Output trace file")

    args = parser.parse_args()

    print("=" * 70)
    print("PEP-based Latency Testing")
    print("=" * 70)
    print(f"Batches: {args.num_batches}")
    print(f"Nodes per batch: {args.num_nodes}")
    print(f"Output: {args.output}")

    if not OPENVINO_AVAILABLE:
        print("\nERROR: OpenVINO is required for PEP latency testing")
        return

    profiler = PipelineProfiler("PEP_Latency_Test")
    tester = PEPLatencyTester(profiler, num_nodes=args.num_nodes)

    if args.two_pep:
        results = tester.run_two_pep_test(args.num_batches)
    elif args.pep == 'all':
        results = tester.run_all_peps(args.num_batches)
    else:
        pep = ALL_PEPS[args.pep]
        results = tester.run_pep(pep, args.num_batches, args.pep.upper())

    # Export trace and analyze
    profiler.export_chrome_trace(args.output)
    profiler.analyze_metrics()

    print("\n" + "=" * 70)
    print("Test Complete!")
    print(f"Trace saved to: {args.output}")
    print("Open in chrome://tracing or https://ui.perfetto.dev")
    print("=" * 70)


if __name__ == "__main__":
    main()
