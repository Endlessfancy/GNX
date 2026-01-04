"""
Pipeline Time Analysis

按照 pipeline cycle 统计时间：
- Cycle 1: SG0 进入 Stage0
- Cycle 2: SG1 进入 Stage0, SG0 进入 Stage1
- ...
- 每个 cycle 时间 = max(stage0_time, stage1_time)

DP 模式：
- Stage0 时间 = max(CPU, GPU) + merge 时间
- 不计算 partition 时间（假设离线分割）

Usage:
    python analyze_pipeline.py
    python analyze_pipeline.py --csv results/cost_model.csv
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple


def load_cost_model(csv_path: str) -> pd.DataFrame:
    """Load cost model CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def get_stage_times(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    """
    获取每个 subgraph 的 stage 时间。

    Returns:
        (stage0_times, stage1_times) - 每个 subgraph 的时间列表

    Stage0 = max(CPU, GPU) + merge
    Stage1 = NPU
    """
    sg_ids = sorted(df['sg_id'].unique())

    stage0_times = []
    stage1_times = []

    for sg_id in sg_ids:
        sg_data = df[df['sg_id'] == sg_id]

        # Stage0: max(CPU, GPU) + merge
        cpu_time = sg_data['stage0_CPU_wall_ms'].mean() if 'stage0_CPU_wall_ms' in df.columns else 0
        gpu_time = sg_data['stage0_GPU_wall_ms'].mean() if 'stage0_GPU_wall_ms' in df.columns else 0
        merge_time = sg_data['merge_ms'].mean() if 'merge_ms' in df.columns else 0

        stage0 = max(cpu_time, gpu_time) + merge_time
        stage0_times.append(stage0)

        # Stage1: NPU
        npu_time = sg_data['stage1_NPU_wall_ms'].mean() if 'stage1_NPU_wall_ms' in df.columns else 0
        stage1_times.append(npu_time)

    return stage0_times, stage1_times


def calculate_pipeline_time(stage0_times: List[float], stage1_times: List[float]) -> Tuple[float, List[dict]]:
    """
    计算 pipeline 总时间。

    Pipeline cycles:
    - Cycle 0: SG0 Stage0
    - Cycle 1: SG1 Stage0, SG0 Stage1 -> max(stage0[1], stage1[0])
    - Cycle 2: SG2 Stage0, SG1 Stage1 -> max(stage0[2], stage1[1])
    - ...
    - Cycle N: SG(N-1) Stage1 (drain)

    Returns:
        (total_time, cycles) - 总时间和每个 cycle 的详情
    """
    n = len(stage0_times)
    cycles = []
    total_time = 0

    # Cycle 0: 只有 SG0 进入 Stage0
    cycle_time = stage0_times[0]
    cycles.append({
        'cycle': 0,
        'stage0_sg': 0,
        'stage1_sg': None,
        'stage0_time': stage0_times[0],
        'stage1_time': 0,
        'cycle_time': cycle_time
    })
    total_time += cycle_time

    # Cycle 1 to N-1: Stage0 和 Stage1 并行
    for i in range(1, n):
        s0_time = stage0_times[i]
        s1_time = stage1_times[i - 1]
        cycle_time = max(s0_time, s1_time)

        cycles.append({
            'cycle': i,
            'stage0_sg': i,
            'stage1_sg': i - 1,
            'stage0_time': s0_time,
            'stage1_time': s1_time,
            'cycle_time': cycle_time
        })
        total_time += cycle_time

    # Cycle N: 最后一个 SG 的 Stage1 (drain)
    cycle_time = stage1_times[n - 1]
    cycles.append({
        'cycle': n,
        'stage0_sg': None,
        'stage1_sg': n - 1,
        'stage0_time': 0,
        'stage1_time': stage1_times[n - 1],
        'cycle_time': cycle_time
    })
    total_time += cycle_time

    return total_time, cycles


def calculate_sequential_time(stage0_times: List[float], stage1_times: List[float]) -> float:
    """计算顺序执行时间（无重叠）。"""
    total = 0
    for i in range(len(stage0_times)):
        total += stage0_times[i] + stage1_times[i]
    return total


def analyze_csv(csv_path: str):
    """Main analysis."""
    print("=" * 60)
    print("Pipeline Time Analysis")
    print("=" * 60)

    # Load data
    df = load_cost_model(csv_path)

    # Get stage times
    stage0_times, stage1_times = get_stage_times(df)
    n = len(stage0_times)

    # Print per-subgraph times
    print(f"\nSubgraphs: {n}")
    print(f"\nPer-subgraph times (ms):")
    print("-" * 50)
    print(f"{'SG':<4} {'Stage0':>12} {'Stage1':>12}")
    print(f"{'':4} {'(max+merge)':>12} {'(NPU)':>12}")
    print("-" * 50)
    for i in range(n):
        print(f"{i:<4} {stage0_times[i]:>12.2f} {stage1_times[i]:>12.2f}")

    # Calculate times
    seq_time = calculate_sequential_time(stage0_times, stage1_times)
    pipe_time, cycles = calculate_pipeline_time(stage0_times, stage1_times)

    # Print cycle details
    print(f"\n" + "=" * 60)
    print("Pipeline Cycles")
    print("=" * 60)
    print(f"{'Cycle':<6} {'Stage0 SG':>10} {'Stage1 SG':>10} {'S0 Time':>10} {'S1 Time':>10} {'Cycle':>10}")
    print("-" * 60)
    for c in cycles:
        s0_sg = str(c['stage0_sg']) if c['stage0_sg'] is not None else '-'
        s1_sg = str(c['stage1_sg']) if c['stage1_sg'] is not None else '-'
        print(f"{c['cycle']:<6} {s0_sg:>10} {s1_sg:>10} {c['stage0_time']:>10.2f} {c['stage1_time']:>10.2f} {c['cycle_time']:>10.2f}")

    # Summary
    print(f"\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Sequential time: {seq_time:.2f} ms")
    print(f"  Pipeline time:   {pipe_time:.2f} ms")
    print(f"  Speedup:         {seq_time / pipe_time:.2f}x")
    print(f"\n  Stage0 total:    {sum(stage0_times):.2f} ms")
    print(f"  Stage1 total:    {sum(stage1_times):.2f} ms")

    return pipe_time, cycles


def main():
    parser = argparse.ArgumentParser(description="Pipeline Time Analysis")
    parser.add_argument("--csv", type=str,
                        default=str(Path(__file__).parent / "results" / "cost_model.csv"),
                        help="Path to cost_model.csv")
    args = parser.parse_args()

    analyze_csv(args.csv)


if __name__ == "__main__":
    main()
