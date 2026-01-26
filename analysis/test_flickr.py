"""
Test Pipeline Scheduler with Flickr Dataset using compute-only profiling data.
"""
import json
import re
from typing import Dict, Tuple, List
from pipeline_scheduler import PipelineScheduler, SubgraphInfo


def load_fused_block_data(filepath: str) -> Dict[Tuple[int, int], float]:
    """
    Load fused block profiling data (fused1-4_cpu/gpu_compute_only.json).

    Format: "n,m,device,block0" -> {mean, std, original_mean, transfer_time}
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    result = {}
    for key, value in data.items():
        parts = key.split(',')
        if len(parts) == 4:
            n, m = int(parts[0]), int(parts[1])
            result[(n, m)] = value['mean']
    return result


def load_npu_stage_data(filepath: str, stages: List[int] = [5, 6, 7]) -> Dict[Tuple[int, int], float]:
    """
    Load NPU stage data and aggregate specified stages.

    Format: "n,m,NPU,stage" -> {mean, std, original_mean, transfer_time}

    Note: NPU stages 5-7 are node-only operations (NORMALIZE, TRANSFORM, ACTIVATE).
    Their compute time is independent of m (edges). We use original_mean instead of
    mean because the transfer_time subtraction is not applicable to NPU profiling
    (NPU original times appear to already be compute-only based on their m-independence).
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Group by (n, m)
    nm_times = {}

    for key, value in data.items():
        parts = key.split(',')
        if len(parts) == 4:
            n, m, device, stage = int(parts[0]), int(parts[1]), parts[2], int(parts[3])
            if device == 'NPU' and stage in stages:
                nm_key = (n, m)
                if nm_key not in nm_times:
                    nm_times[nm_key] = 0.0
                # Use original_mean for NPU (transfer subtraction not applicable)
                nm_times[nm_key] += value['original_mean']

    return nm_times


def parse_flickr_partition(filepath: str, k: int) -> List[Tuple[int, int, int]]:
    """
    Parse Flickr partition data for a specific K value.

    Returns: List of (id, total_nodes, internal_edges) tuples
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Find K=k section
    pattern = rf'### K={k}[^\n]*\n\n\|[^\n]+\n\|[-\s|]+\n((?:\|[^\n]+\n)+)'
    match = re.search(pattern, content)

    if not match:
        print(f"Could not find K={k} section")
        return []

    table_content = match.group(1)

    # Parse table rows: | ID | Owned | Halo | Total | Internal | Boundary | Halo% |
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
    analysis_dir = "/home/haoyang/private/GNX_final/analysis"
    cpu_file = f"{analysis_dir}/fused1-4_cpu_compute_only.json"
    gpu_file = f"{analysis_dir}/fused1-4_gpu_compute_only.json"
    npu_file = f"{analysis_dir}/profiling_stage1-7_compute_only.json"
    bw_file = f"{analysis_dir}/bandwidth_config.json"
    partition_file = "/home/haoyang/private/GNX_final/experiments/graphPartition/flickr_partition.md"

    # Load bandwidth configuration
    with open(bw_file, 'r') as f:
        bw_config = json.load(f)

    print("=" * 80)
    print("Bandwidth Configuration (GB/s):")
    print("=" * 80)
    bw = bw_config.get('bandwidth_corrected', bw_config.get('bandwidth_original', {}))
    for device, values in bw.items():
        print(f"  {device}: input={values['input']:.3f}, output={values['output']:.3f}")

    # Load profiling data
    print("\n" + "=" * 80)
    print("Loading compute-only profiling data...")
    print("=" * 80)

    cpu_data = load_fused_block_data(cpu_file)
    print(f"CPU Stage 1-4: {len(cpu_data)} data points")

    gpu_data = load_fused_block_data(gpu_file)
    print(f"GPU Stage 1-4: {len(gpu_data)} data points")

    npu_data = load_npu_stage_data(npu_file, stages=[5, 6, 7])
    print(f"NPU Stage 5-7: {len(npu_data)} data points")

    # Create scheduler
    scheduler = PipelineScheduler(
        cpu_data=cpu_data,
        gpu_data=gpu_data,
        npu_data=npu_data,
        bandwidth_config=bw_config,
        feature_dim=500,  # Flickr
        gpu_edge_limit=2_500_000,
        dp_max_iterations=9,
    )

    print(f"\nScheduler configured:")
    print(f"  CPU bandwidth: in={scheduler.cpu_bw_in:.3f}, out={scheduler.cpu_bw_out:.3f} GB/s")
    print(f"  GPU bandwidth: in={scheduler.gpu_bw_in:.3f}, out={scheduler.gpu_bw_out:.3f} GB/s")
    print(f"  NPU bandwidth: in={scheduler.npu_bw_in:.3f}, out={scheduler.npu_bw_out:.3f} GB/s")

    # Test with different K values
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print("\n" + "=" * 80)
    print("Flickr Dataset Pipeline Scheduling Results (Compute-Only Profiling)")
    print("=" * 80)

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
            print(f"  Partition {id_}: {total_nodes:,} nodes, {internal_edges:,} edges (m/n={internal_edges/total_nodes:.1f})")

        # Special case: K=1 (no pipeline, just sequential)
        if k == 1:
            sg = subgraphs[0]
            sg.stage1_time, sg.stage2_time = scheduler.get_pipeline_stage_times(sg)
            makespan = sg.stage1_time + sg.stage2_time

            print(f"\nK=1 Baseline (no pipeline parallelism):")
            print(f"  Stage 1 (CPU+GPU DP):")
            print(f"    α = {sg.dp_split_ratio*100:.1f}% (CPU ratio)")
            print(f"    CPU: compute={sg.cpu_compute_time:.2f}ms, transfer={sg.cpu_transfer_time:.2f}ms, total={sg.cpu_total_time:.2f}ms")
            print(f"    GPU: compute={sg.gpu_compute_time:.2f}ms, transfer={sg.gpu_transfer_time:.2f}ms, total={sg.gpu_total_time:.2f}ms")
            print(f"    Stage 1 time = max(CPU, GPU) = {sg.stage1_time:.2f}ms")
            print(f"  Stage 2 (NPU):")
            print(f"    compute={sg.stage2_compute:.2f}ms, transfer={sg.stage2_transfer:.2f}ms")
            print(f"    Stage 2 time = {sg.stage2_time:.2f}ms")
            print(f"  Total Makespan = Stage1 + Stage2 = {makespan:.2f}ms")

            all_results[f"K={k}"] = {
                "k": k,
                "num_partitions": 1,
                "optimal_order": [0],
                "makespan_ms": makespan,
                "baseline": True,
                "stage1_time_ms": sg.stage1_time,
                "stage2_time_ms": sg.stage2_time,
                "subgraphs": [sg.to_dict()],
            }
            continue

        # Optimize with pipeline
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

        # Average utilization
        total_cpu = sum(cdt.cpu_total for cdt in result.cycle_device_times)
        total_gpu = sum(cdt.gpu_total for cdt in result.cycle_device_times)
        total_npu = sum(cdt.npu_total for cdt in result.cycle_device_times)
        print("-" * 50)
        print(f"{'Avg':<6} {result.makespan:<10.2f} {total_cpu/result.makespan*100:<8.1f} {total_gpu/result.makespan*100:<8.1f} {total_npu/result.makespan*100:<8.1f}")

        # DP split info
        print(f"\nDP Split (CPU+GPU for Stage 1):")
        print(f"{'SG':<4} {'α%':<8} {'CPU Comp':<12} {'GPU Comp':<12} {'S1 Time':<12} {'S2 Time':<12}")
        print("-" * 70)
        sg_dict = {sg.id: sg for sg in subgraphs}
        for sg_id in result.order:
            sg = sg_dict[sg_id]
            print(f"{sg_id:<4} {sg.dp_split_ratio*100:<8.1f} {sg.cpu_compute_time:<12.2f} {sg.gpu_compute_time:<12.2f} {sg.stage1_time:<12.2f} {sg.stage2_time:<12.2f}")

        # Store results
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

    # Calculate speedup vs K=1 baseline
    if "K=1" in all_results:
        baseline = all_results["K=1"]["makespan_ms"]
        print("\n" + "=" * 80)
        print("Summary: Speedup vs K=1 Baseline")
        print("=" * 80)
        print(f"{'K':<4} {'Makespan (ms)':<15} {'Speedup':<10}")
        print("-" * 35)
        for k in k_values:
            key = f"K={k}"
            if key in all_results:
                makespan = all_results[key]["makespan_ms"]
                speedup = baseline / makespan
                marker = " ⭐" if speedup == max(baseline / all_results[f"K={kk}"]["makespan_ms"] for kk in k_values if f"K={kk}" in all_results) else ""
                print(f"{k:<4} {makespan:<15.2f} {speedup:<10.2f}x{marker}")

    # Save results
    output_file = f"{analysis_dir}/flickr_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
