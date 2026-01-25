"""
Test Pipeline Scheduler with Flickr Dataset
"""
import json
import re
from typing import Dict, Tuple, List
from pipeline_scheduler import PipelineScheduler, SubgraphInfo, Interpolator2D


def load_profiling_json(filepath: str, device: str, stages: List[int]) -> Dict[Tuple[int, int], float]:
    """
    Load profiling data from JSON and aggregate stages.

    Args:
        filepath: Path to JSON file
        device: Device filter (CPU/GPU/NPU)
        stages: List of stage numbers to aggregate

    Returns:
        Dictionary of {(n, m): total_time_ms}
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Group by (n, m)
    nm_times = {}

    for key, value in data.items():
        parts = key.split(',')
        if len(parts) != 4:
            continue

        n, m, dev, stage = int(parts[0]), int(parts[1]), parts[2], int(parts[3])

        if dev == device and stage in stages:
            nm_key = (n, m)
            if nm_key not in nm_times:
                nm_times[nm_key] = 0.0
            nm_times[nm_key] += value['mean']

    return nm_times


def parse_flickr_partition(filepath: str, k: int) -> List[Tuple[int, int, int]]:
    """
    Parse Flickr partition data for a specific K value.

    Args:
        filepath: Path to markdown file
        k: Number of partitions

    Returns:
        List of (id, total_nodes, internal_edges) tuples
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Find the section for K=k
    pattern = rf'### K={k}\s*\n\n\|.*\n\|[-\s|]+\n((?:\|[^\n]+\n)+)'
    match = re.search(pattern, content)

    if not match:
        print(f"Could not find K={k} section")
        return []

    table_content = match.group(1)

    # Parse table rows
    partitions = []
    row_pattern = r'\|\s*(\d+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|'

    for row_match in re.finditer(row_pattern, table_content):
        id_ = int(row_match.group(1))
        total_nodes = int(row_match.group(4).replace(',', ''))
        internal_edges = int(row_match.group(5).replace(',', ''))
        partitions.append((id_, total_nodes, internal_edges))

    return partitions


def main():
    # Paths
    profiling_dir = "/home/haoyang/private/GNX_final/profiling/results/185H"
    cpu_file = f"{profiling_dir}/checkpoint_merged.json"
    gpu_file = f"{profiling_dir}/checkpoint_gpu.json"
    npu_file = f"{profiling_dir}/checkpoint_npu.json"
    bw_file = f"{profiling_dir}/bandwidth_table.json"
    partition_file = "/home/haoyang/private/GNX_final/experiments/graphPartition/flickr_partition.md"

    # Load bandwidth data
    with open(bw_file, 'r') as f:
        bw_data = json.load(f)

    print("=" * 80)
    print("Bandwidth Table (MB/s):")
    print("=" * 80)
    for key, value in bw_data.items():
        if value != float('inf'):
            print(f"  {key}: {value:.2f} MB/s = {value/1000:.2f} GB/s")
        else:
            print(f"  {key}: Infinity")

    # Calculate effective bandwidth for each pipeline stage
    # Stage 1 (CPU): Use CPU_stage1 bandwidth
    # Stage 1 (GPU): Use GPU_stage1 bandwidth
    # Stage 2 (NPU): Use NPU_stage5 or similar
    cpu_bw_mbps = bw_data.get('CPU_stage1', 2500)  # MB/s
    gpu_bw_mbps = bw_data.get('GPU_stage1', 2500)  # MB/s
    npu_bw_mbps = bw_data.get('NPU_stage6', 3000)  # MB/s (use stage6 as representative)

    cpu_bw_gbps = cpu_bw_mbps / 1000  # Convert to GB/s
    gpu_bw_gbps = gpu_bw_mbps / 1000
    npu_bw_gbps = npu_bw_mbps / 1000

    print(f"\nUsing bandwidths:")
    print(f"  CPU: {cpu_bw_gbps:.2f} GB/s")
    print(f"  GPU: {gpu_bw_gbps:.2f} GB/s")
    print(f"  NPU: {npu_bw_gbps:.2f} GB/s")

    # Load profiling data
    print("\n" + "=" * 80)
    print("Loading profiling data...")
    print("=" * 80)

    # Stage 1-4 for CPU (fused)
    cpu_data = load_profiling_json(cpu_file, 'CPU', [1, 2, 3, 4])
    print(f"CPU Stage 1-4: {len(cpu_data)} data points")

    # Stage 1-4 for GPU (fused)
    gpu_data = load_profiling_json(gpu_file, 'GPU', [1, 2, 3, 4])
    print(f"GPU Stage 1-4: {len(gpu_data)} data points")

    # Stage 5-7 for NPU
    npu_data = load_profiling_json(npu_file, 'NPU', [5, 6, 7])
    print(f"NPU Stage 5-7: {len(npu_data)} data points")

    # Create scheduler
    scheduler = PipelineScheduler.__new__(PipelineScheduler)
    scheduler.stage1_stages = ['fused_1234']
    scheduler.stage2_stages = ['fused_567']
    scheduler.device = 'mixed'
    scheduler.cpu_bw_gbps = cpu_bw_gbps
    scheduler.gpu_bw_gbps = gpu_bw_gbps
    scheduler.npu_bw_gbps = npu_bw_gbps
    scheduler.stage1_bw_gbps = cpu_bw_gbps
    scheduler.stage2_bw_gbps = npu_bw_gbps
    scheduler.feature_dim = 500  # Flickr feature dimension
    scheduler.bytes_per_element = 4
    scheduler.gpu_edge_limit = 2_500_000
    scheduler.dp_max_iterations = 9

    # Build interpolators
    scheduler.cpu_interpolator = Interpolator2D(cpu_data) if cpu_data else None
    scheduler.gpu_interpolator = Interpolator2D(gpu_data) if gpu_data else None
    scheduler.npu_interpolator = Interpolator2D(npu_data) if npu_data else None

    # Test with different K values
    k_values = [2, 4, 6, 8, 10]

    print("\n" + "=" * 80)
    print("Flickr Dataset Pipeline Scheduling Results")
    print("=" * 80)

    # Store all results for JSON export
    all_results = {}

    for k in k_values:
        partitions = parse_flickr_partition(partition_file, k)

        if not partitions:
            print(f"\nK={k}: No partition data found")
            continue

        print(f"\n{'='*80}")
        print(f"K={k} ({len(partitions)} partitions)")
        print(f"{'='*80}")

        # Create subgraphs
        subgraphs = []
        for id_, total_nodes, internal_edges in partitions:
            sg = SubgraphInfo(id=id_, total_nodes=total_nodes, internal_edges=internal_edges)
            subgraphs.append(sg)
            print(f"  Partition {id_}: {total_nodes:,} nodes, {internal_edges:,} edges")

        # Optimize
        result = scheduler.optimize(subgraphs)

        # Print results
        print(f"\nOptimal Order: {result.order}")
        print(f"Total Makespan: {result.makespan:.2f} ms")

        # Print cycle breakdown
        print(f"\nCycle Breakdown:")
        print(f"{'Cycle':<6} {'Time':<10} {'CPU%':<8} {'GPU%':<8} {'NPU%':<8}")
        print("-" * 50)
        for i, cdt in enumerate(result.cycle_device_times):
            print(f"{i:<6} {cdt.cycle_time:<10.2f} {cdt.cpu_util*100:<8.1f} {cdt.gpu_util*100:<8.1f} {cdt.npu_util*100:<8.1f}")

        # Print average utilization
        total_cpu = sum(cdt.cpu_total for cdt in result.cycle_device_times)
        total_gpu = sum(cdt.gpu_total for cdt in result.cycle_device_times)
        total_npu = sum(cdt.npu_total for cdt in result.cycle_device_times)
        print("-" * 50)
        print(f"{'Avg':<6} {result.makespan:<10.2f} {total_cpu/result.makespan*100:<8.1f} {total_gpu/result.makespan*100:<8.1f} {total_npu/result.makespan*100:<8.1f}")

        # Print DP split info
        print(f"\nDP Split (CPU+GPU for Stage 1):")
        print(f"{'SG':<4} {'α%':<8} {'CPU Comp':<12} {'GPU Comp':<12} {'S1 Time':<12} {'S2 Time':<12}")
        print("-" * 70)
        sg_dict = {sg.id: sg for sg in subgraphs}
        for sg_id in result.order:
            sg = sg_dict[sg_id]
            print(f"{sg_id:<4} {sg.dp_split_ratio*100:<8.1f} {sg.cpu_compute_time:<12.2f} {sg.gpu_compute_time:<12.2f} {sg.stage1_time:<12.2f} {sg.stage2_time:<12.2f}")

        # Store results for JSON export
        all_results[f"K={k}"] = {
            "k": k,
            "num_partitions": len(partitions),
            "optimal_order": result.order,
            "makespan_ms": result.makespan,
            "cycle_times": result.cycle_times,
            "cycle_device_times": [cdt.to_dict() for cdt in result.cycle_device_times],
            "subgraphs": [sg.to_dict() for sg in subgraphs],
            "avg_utilization": {
                "cpu": total_cpu / result.makespan if result.makespan > 0 else 0,
                "gpu": total_gpu / result.makespan if result.makespan > 0 else 0,
                "npu": total_npu / result.makespan if result.makespan > 0 else 0
            }
        }


    # Save all results to JSON
    output_file = "/home/haoyang/private/GNX_final/experiments/pipeline_time/flickr_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
