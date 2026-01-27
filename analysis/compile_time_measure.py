"""
Measure compilation/optimization time for pipeline scheduler.
Tests the search engine performance for different K values.
"""
import time
import json
import re
import argparse
from typing import Dict, Tuple, List
from pipeline_scheduler import PipelineScheduler, SubgraphInfo


def load_fused_block_data(filepath: str) -> Dict[Tuple[int, int], float]:
    """Load fused block profiling data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = {}
    for key, value in data.items():
        parts = key.split(',')
        if len(parts) == 4:
            n, m = int(parts[0]), int(parts[1])
            result[(n, m)] = value['mean']
    return result


def load_npu_stage_data(filepath: str, stages: List[int] = [5, 6, 7]) -> Dict[Tuple[int, int], float]:
    """Load NPU stage data (using original_mean)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    nm_times = {}
    for key, value in data.items():
        parts = key.split(',')
        if len(parts) == 4:
            n, m, device, stage = int(parts[0]), int(parts[1]), parts[2], int(parts[3])
            if device == 'NPU' and stage in stages:
                nm_key = (n, m)
                if nm_key not in nm_times:
                    nm_times[nm_key] = 0.0
                nm_times[nm_key] += value['original_mean']
    return nm_times


def parse_flickr_partition(filepath: str, k: int) -> List[Tuple[int, int, int]]:
    """Parse Flickr partition data for a specific K value."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = rf'### K={k}[^\n]*\n\n\|[^\n]+\n\|[-\s|]+\n((?:\|[^\n]+\n)+)'
    match = re.search(pattern, content)
    if not match:
        return []
    table_content = match.group(1)
    partitions = []
    row_pattern = r'\|\s*(\d+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|'
    for row_match in re.finditer(row_pattern, table_content):
        id_ = int(row_match.group(1))
        total_nodes = int(row_match.group(4).replace(',', ''))
        internal_edges = int(row_match.group(5).replace(',', ''))
        partitions.append((id_, total_nodes, internal_edges))
    return partitions


def measure_single_k(scheduler: PipelineScheduler, subgraphs: List[SubgraphInfo], k: int) -> dict:
    """Measure optimization time for a single K value."""
    # Reset subgraphs
    for sg in subgraphs:
        sg.stage1_time = 0.0
        sg.stage2_time = 0.0

    t_start = time.perf_counter()

    if k == 1:
        sg = subgraphs[0]
        sg.stage1_time, sg.stage2_time = scheduler.get_pipeline_stage_times(sg)
        makespan = sg.stage1_time + sg.stage2_time
        order = [0]
    else:
        result = scheduler.optimize(subgraphs)
        makespan = result.makespan
        order = result.order

    t_end = time.perf_counter()
    elapsed_ms = (t_end - t_start) * 1000

    return {
        'k': k,
        'num_subgraphs': len(subgraphs),
        'optimization_time_ms': elapsed_ms,
        'makespan_ms': makespan,
        'order': order,
    }


def measure_breakdown(scheduler: PipelineScheduler, subgraphs: List[SubgraphInfo]) -> dict:
    """Measure detailed breakdown of optimization time."""
    k = len(subgraphs)

    # 1. Binary Search (DP Split) for each subgraph
    bs_times = []
    for sg in subgraphs:
        t0 = time.perf_counter()
        scheduler.find_optimal_dp_split(sg.total_nodes, sg.internal_edges)
        t1 = time.perf_counter()
        bs_times.append((t1 - t0) * 1000)

    # 2. get_pipeline_stage_times
    subgraphs_copy = [SubgraphInfo(id=sg.id, total_nodes=sg.total_nodes, internal_edges=sg.internal_edges)
                      for sg in subgraphs]
    stage_times = []
    for sg in subgraphs_copy:
        t0 = time.perf_counter()
        scheduler.get_pipeline_stage_times(sg)
        t1 = time.perf_counter()
        stage_times.append((t1 - t0) * 1000)

    # 3. Johnson's Rule sorting (pre-compute stage times first)
    for sg in subgraphs_copy:
        pass  # Already computed above

    t0 = time.perf_counter()
    group_a = [sg for sg in subgraphs_copy if sg.stage1_time < sg.stage2_time]
    group_b = [sg for sg in subgraphs_copy if sg.stage1_time >= sg.stage2_time]
    group_a.sort(key=lambda x: x.stage1_time)
    group_b.sort(key=lambda x: x.stage2_time, reverse=True)
    order = [sg.id for sg in group_a] + [sg.id for sg in group_b]
    t1 = time.perf_counter()
    sorting_time = (t1 - t0) * 1000

    # 4. calculate_makespan
    t0 = time.perf_counter()
    result = scheduler.calculate_makespan(order, subgraphs_copy)
    t1 = time.perf_counter()
    makespan_calc_time = (t1 - t0) * 1000

    return {
        'k': k,
        'binary_search_per_subgraph_ms': bs_times,
        'binary_search_total_ms': sum(bs_times),
        'stage_times_per_subgraph_ms': stage_times,
        'stage_times_total_ms': sum(stage_times),
        'sorting_time_ms': sorting_time,
        'makespan_calc_time_ms': makespan_calc_time,
    }


def main():
    parser = argparse.ArgumentParser(description='Measure pipeline scheduler optimization time')
    parser.add_argument('--k', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help='K values to test (default: 1-10)')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of repetitions for averaging (default: 1)')
    parser.add_argument('--breakdown', action='store_true',
                        help='Show detailed time breakdown')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    args = parser.parse_args()

    # Paths (relative to script location)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_dir = script_dir
    cpu_file = os.path.join(analysis_dir, "fused1-4_cpu_compute_only.json")
    gpu_file = os.path.join(analysis_dir, "fused1-4_gpu_compute_only.json")
    npu_file = os.path.join(analysis_dir, "profiling_stage1-7_compute_only.json")
    bw_file = os.path.join(analysis_dir, "bandwidth_config.json")
    partition_file = os.path.join(script_dir, "..", "experiments", "graphPartition", "flickr_partition.md")

    # Load data
    print("Loading profiling data...")
    t0 = time.perf_counter()
    cpu_data = load_fused_block_data(cpu_file)
    gpu_data = load_fused_block_data(gpu_file)
    npu_data = load_npu_stage_data(npu_file)
    with open(bw_file, 'r', encoding='utf-8') as f:
        bw_config = json.load(f)
    t1 = time.perf_counter()
    print(f"Data loading: {(t1-t0)*1000:.2f} ms")

    # Create scheduler
    print("Initializing scheduler...")
    t0 = time.perf_counter()
    scheduler = PipelineScheduler(
        cpu_data=cpu_data,
        gpu_data=gpu_data,
        npu_data=npu_data,
        bandwidth_config=bw_config,
        feature_dim=500,
        gpu_edge_limit=2_500_000,
        dp_max_iterations=9,
    )
    t1 = time.perf_counter()
    print(f"Scheduler init: {(t1-t0)*1000:.2f} ms")

    # Run measurements
    print()
    print("=" * 70)
    print(f"{'K':<4} {'Subgraphs':<12} {'Opt Time (ms)':<18} {'Makespan (ms)':<15}")
    print("=" * 70)

    all_results = []

    for k in args.k:
        partitions = parse_flickr_partition(partition_file, k)
        if not partitions:
            print(f"K={k}: No partition data found")
            continue

        times = []
        for _ in range(args.repeat):
            subgraphs = [SubgraphInfo(id=id_, total_nodes=n, internal_edges=m)
                         for id_, n, m in partitions]
            result = measure_single_k(scheduler, subgraphs, k)
            times.append(result['optimization_time_ms'])

        avg_time = sum(times) / len(times)
        result['optimization_time_ms'] = avg_time
        if args.repeat > 1:
            result['optimization_time_std'] = (sum((t - avg_time)**2 for t in times) / len(times))**0.5

        all_results.append(result)

        print(f"{k:<4} {len(partitions):<12} {avg_time:<18.3f} {result['makespan_ms']:<15.2f}")

    print("=" * 70)

    # Detailed breakdown
    if args.breakdown:
        print("\nDetailed Breakdown (K=10):")
        print("-" * 70)
        partitions = parse_flickr_partition(partition_file, 10)
        if partitions:
            subgraphs = [SubgraphInfo(id=id_, total_nodes=n, internal_edges=m)
                         for id_, n, m in partitions]
            breakdown = measure_breakdown(scheduler, subgraphs)

            print(f"Binary Search (DP Split):")
            for i, t in enumerate(breakdown['binary_search_per_subgraph_ms']):
                print(f"  Subgraph {i}: {t:.3f} ms")
            print(f"  Total: {breakdown['binary_search_total_ms']:.3f} ms")
            print(f"\nJohnson's Rule Sorting: {breakdown['sorting_time_ms']:.3f} ms")
            print(f"Makespan Calculation: {breakdown['makespan_calc_time_ms']:.3f} ms")

    # Save results
    if args.output:
        output_data = {
            'config': {
                'k_values': args.k,
                'repeat': args.repeat,
                'dp_max_iterations': 9,
                'feature_dim': 500,
            },
            'results': all_results,
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
