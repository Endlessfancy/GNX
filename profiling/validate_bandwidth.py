"""
验证带宽估算：比较 fused stage 和单独 stage 的累和

方法:
1. 单独 stage (sync API): measured = input_transfer + compute + output_transfer
   → compute = measured - input_transfer - output_transfer

2. Fused stage (async API): measured = input_transfer + compute (无 output transfer)
   → compute = measured - input_transfer

验证: sum(stage_1-4_compute) ≈ fused_1-4_compute
"""

import json
from pathlib import Path

# 带宽数据 (GB/s) - 来自 bandwidth_v2.json
BANDWIDTH = {
    'GPU': {'input': 5.243, 'output': 3.305},
}

FEATURE_DIM = 500
BYTES_PER_FLOAT = 4
BYTES_PER_INT = 8


def get_io_size(stage_id, num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """获取每个 stage 的输入/输出大小 (bytes)"""
    if stage_id == 1:
        input_size = num_nodes * feature_dim * BYTES_PER_FLOAT + 2 * num_edges * BYTES_PER_INT
        output_size = num_edges * feature_dim * BYTES_PER_FLOAT
    elif stage_id == 2:
        input_size = num_edges * feature_dim * BYTES_PER_FLOAT
        output_size = num_edges * feature_dim * BYTES_PER_FLOAT
    elif stage_id == 3:
        input_size = num_edges * feature_dim * BYTES_PER_FLOAT + 2 * num_edges * BYTES_PER_INT
        output_size = num_nodes * feature_dim * BYTES_PER_FLOAT
    elif stage_id == 4:
        input_size = 2 * num_edges * BYTES_PER_INT
        output_size = num_nodes * BYTES_PER_FLOAT
    elif stage_id == 5:
        input_size = num_nodes * feature_dim * BYTES_PER_FLOAT + num_nodes * BYTES_PER_FLOAT
        output_size = num_nodes * feature_dim * BYTES_PER_FLOAT
    elif stage_id == 6:
        input_size = 2 * num_nodes * feature_dim * BYTES_PER_FLOAT
        output_size = num_nodes * 256 * BYTES_PER_FLOAT
    elif stage_id == 7:
        input_size = num_nodes * feature_dim * BYTES_PER_FLOAT
        output_size = num_nodes * feature_dim * BYTES_PER_FLOAT
    else:
        raise ValueError(f"Invalid stage_id: {stage_id}")
    return input_size, output_size


def get_fused_io_size(fused_type, num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """获取 fused stage 的输入/输出大小"""
    if fused_type == "1-4":
        # Fused 1-4: Input = Stage1 input, Output = Stage4 output
        input_size = num_nodes * feature_dim * BYTES_PER_FLOAT + 2 * num_edges * BYTES_PER_INT
        output_size = num_nodes * BYTES_PER_FLOAT
    elif fused_type == "1-7":
        # Fused 1-7: Input = Stage1 input, Output = Stage7 output (but stage 6 output is 256)
        input_size = num_nodes * feature_dim * BYTES_PER_FLOAT + 2 * num_edges * BYTES_PER_INT
        output_size = num_nodes * 256 * BYTES_PER_FLOAT  # Stage 6 output is 256-dim
    else:
        raise ValueError(f"Invalid fused_type: {fused_type}")
    return input_size, output_size


def calc_transfer_time(input_bytes, output_bytes, device, include_output=True):
    """计算传输时间 (ms)"""
    bw = BANDWIDTH.get(device)
    if not bw:
        return 0.0, 0.0

    input_ms = input_bytes / (bw['input'] * 1e6)
    output_ms = output_bytes / (bw['output'] * 1e6) if include_output else 0
    return input_ms, output_ms


def main():
    script_dir = Path(__file__).parent

    # Load data
    with open(script_dir / 'results/185H/raw_measurements.json') as f:
        raw_data = json.load(f)

    with open(script_dir / '../profiling_fused/results/185H/Test2/block0_gpu.json') as f:
        fused_1_4_data = json.load(f)

    with open(script_dir / '../experiments/baseline_fused1-7_profiling/results/185H/fused_block0_7_gpu.json') as f:
        fused_1_7_data = json.load(f)

    # Test cases
    test_cases = [
        (5000, 50000),
        (10000, 100000),
        (20000, 200000),
        (50000, 500000),
    ]

    print("=" * 90)
    print("Bandwidth Validation: Fused vs Sum of Individual Stages")
    print("=" * 90)
    print()
    print("Sync API (individual stages): measured = input_transfer + compute + output_transfer")
    print("Async API (fused stages):     measured = input_transfer + compute (no output)")
    print()

    for num_nodes, num_edges in test_cases:
        print(f"\n{'='*90}")
        print(f"Test: {num_nodes//1000}k nodes, {num_edges//1000}k edges")
        print(f"{'='*90}")

        # Get individual stage data
        stage_data = {}
        for stage_id in range(1, 8):
            key = f"{num_nodes},{num_edges},GPU,{stage_id}"
            if key in raw_data:
                stage_data[stage_id] = raw_data[key]['mean']

        if len(stage_data) != 7:
            print(f"  Missing individual stage data, skipping...")
            continue

        # Get fused data
        fused_1_4_key = f"{num_nodes},{num_edges},GPU,block0"
        fused_1_7_key = f"fused0_7,{num_nodes},{num_edges},GPU"

        fused_1_4_time = fused_1_4_data.get(fused_1_4_key, {}).get('mean', -1)
        fused_1_7_time = fused_1_7_data.get(fused_1_7_key, {}).get('mean', -1)

        if fused_1_4_time < 0 or fused_1_7_time < 0:
            print(f"  Missing fused data, skipping...")
            continue

        # Calculate compute time for each individual stage
        print(f"\n  Individual Stages (sync API, includes output transfer):")
        print(f"  {'Stage':<8} {'Measured':>10} {'In Xfer':>10} {'Out Xfer':>10} {'Compute':>10}")
        print(f"  {'-'*50}")

        individual_compute = {}
        sum_compute_1_4 = 0
        sum_compute_1_7 = 0

        for stage_id in range(1, 8):
            measured = stage_data[stage_id]
            in_bytes, out_bytes = get_io_size(stage_id, num_nodes, num_edges)
            in_xfer, out_xfer = calc_transfer_time(in_bytes, out_bytes, 'GPU')
            compute = measured - in_xfer - out_xfer
            individual_compute[stage_id] = compute

            if stage_id <= 4:
                sum_compute_1_4 += compute
            sum_compute_1_7 += compute

            print(f"  Stage {stage_id:<3} {measured:>10.2f} {in_xfer:>10.2f} {out_xfer:>10.2f} {compute:>10.2f}")

        # Calculate compute time for fused stages
        print(f"\n  Fused Stages (async API, NO output transfer):")

        # Fused 1-4
        in_bytes_1_4, out_bytes_1_4 = get_fused_io_size("1-4", num_nodes, num_edges)
        in_xfer_1_4, _ = calc_transfer_time(in_bytes_1_4, out_bytes_1_4, 'GPU', include_output=False)
        fused_compute_1_4 = fused_1_4_time - in_xfer_1_4

        # Fused 1-7
        in_bytes_1_7, out_bytes_1_7 = get_fused_io_size("1-7", num_nodes, num_edges)
        in_xfer_1_7, _ = calc_transfer_time(in_bytes_1_7, out_bytes_1_7, 'GPU', include_output=False)
        fused_compute_1_7 = fused_1_7_time - in_xfer_1_7

        print(f"  {'Fused':<8} {'Measured':>10} {'In Xfer':>10} {'Compute':>10}")
        print(f"  {'-'*40}")
        print(f"  1-4      {fused_1_4_time:>10.2f} {in_xfer_1_4:>10.2f} {fused_compute_1_4:>10.2f}")
        print(f"  1-7      {fused_1_7_time:>10.2f} {in_xfer_1_7:>10.2f} {fused_compute_1_7:>10.2f}")

        # Compare
        print(f"\n  Validation:")
        print(f"  {'Comparison':<25} {'Sum Individual':>15} {'Fused':>15} {'Diff':>10} {'Ratio':>10}")
        print(f"  {'-'*75}")

        diff_1_4 = sum_compute_1_4 - fused_compute_1_4
        ratio_1_4 = sum_compute_1_4 / fused_compute_1_4 if fused_compute_1_4 > 0 else float('inf')
        print(f"  Stage 1-4 compute        {sum_compute_1_4:>15.2f} {fused_compute_1_4:>15.2f} {diff_1_4:>10.2f} {ratio_1_4:>10.2f}x")

        diff_1_7 = sum_compute_1_7 - fused_compute_1_7
        ratio_1_7 = sum_compute_1_7 / fused_compute_1_7 if fused_compute_1_7 > 0 else float('inf')
        print(f"  Stage 1-7 compute        {sum_compute_1_7:>15.2f} {fused_compute_1_7:>15.2f} {diff_1_7:>10.2f} {ratio_1_7:>10.2f}x")

        # Also show what happens if we assume individual stages DON'T have output transfer
        print(f"\n  Alternative: If individual stages measured WITHOUT output transfer:")
        sum_no_out_1_4 = 0
        sum_no_out_1_7 = 0
        for stage_id in range(1, 8):
            measured = stage_data[stage_id]
            in_bytes, out_bytes = get_io_size(stage_id, num_nodes, num_edges)
            in_xfer, _ = calc_transfer_time(in_bytes, out_bytes, 'GPU', include_output=False)
            compute = measured - in_xfer
            if stage_id <= 4:
                sum_no_out_1_4 += compute
            sum_no_out_1_7 += compute

        diff_no_out_1_4 = sum_no_out_1_4 - fused_compute_1_4
        ratio_no_out_1_4 = sum_no_out_1_4 / fused_compute_1_4 if fused_compute_1_4 > 0 else float('inf')
        print(f"  Stage 1-4 (no out)       {sum_no_out_1_4:>15.2f} {fused_compute_1_4:>15.2f} {diff_no_out_1_4:>10.2f} {ratio_no_out_1_4:>10.2f}x")

        diff_no_out_1_7 = sum_no_out_1_7 - fused_compute_1_7
        ratio_no_out_1_7 = sum_no_out_1_7 / fused_compute_1_7 if fused_compute_1_7 > 0 else float('inf')
        print(f"  Stage 1-7 (no out)       {sum_no_out_1_7:>15.2f} {fused_compute_1_7:>15.2f} {diff_no_out_1_7:>10.2f} {ratio_no_out_1_7:>10.2f}x")


if __name__ == '__main__':
    main()
