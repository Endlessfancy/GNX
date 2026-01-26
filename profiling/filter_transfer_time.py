"""
过滤 Profiling 结果：扣除数据传输时间，保留纯计算时间

方法：
1. 读取 raw_measurements.json（总时间 = 输入传输 + 计算 + 输出传输）
2. 根据每个 stage 的输入/输出大小计算传输时间
3. 扣除传输时间 = 纯计算时间

旧 profiling 使用同步 API compiled_model(inputs)：
- 即使不重新 set tensor，数据也会传输（test_transfer_time.py 验证）
- 同步 API 返回输出，包含 device→host 传输

带宽来源：bandwidth_v2.json
- CPU: 输入 8.75 GB/s, 输出 3.43 GB/s
- GPU: 输入 5.24 GB/s, 输出 3.31 GB/s
- NPU: 输入 1.27 GB/s, 输出 2.25 GB/s

注意事项：
- 传输和计算可能有重叠（OpenVINO 内部优化）
- 当 transfer_time > total_time * 90% 时，使用限制
"""

import json
from pathlib import Path

# 带宽数据 (GB/s) - 来自 bandwidth_v2.json
BANDWIDTH = {
    'CPU': {'input': 8.754, 'output': 3.434},
    'GPU': {'input': 5.243, 'output': 3.305},
    'NPU': {'input': 1.272, 'output': 2.253},
}

# 校正系数：从 Stage 2 (identity) 验证得出
# 实际测量/估算 比例，用于减少负数
# CPU: 测量值比估算低22%，GPU: 测量值和估算接近
TRANSFER_CORRECTION = {
    'CPU': 0.78,  # 估算传输时间 * 0.78
    'GPU': 1.00,  # 不校正
    'NPU': 1.00,  # 无数据，不校正
}

# 最大传输时间占比（限制传输时间不超过总时间的这个比例）
MAX_TRANSFER_RATIO = 0.9

FEATURE_DIM = 500
BYTES_PER_FLOAT = 4
BYTES_PER_INT = 8


def get_io_size(stage_id, num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """
    获取每个 stage 的输入/输出大小 (bytes)

    Returns:
        (input_bytes, output_bytes)
    """
    if stage_id == 1:
        # Input: x[N,F] + edge_index[2,E], Output: x_j[E,F]
        input_size = num_nodes * feature_dim * BYTES_PER_FLOAT + 2 * num_edges * BYTES_PER_INT
        output_size = num_edges * feature_dim * BYTES_PER_FLOAT

    elif stage_id == 2:
        # Input: x_j[E,F], Output: messages[E,F]
        input_size = num_edges * feature_dim * BYTES_PER_FLOAT
        output_size = num_edges * feature_dim * BYTES_PER_FLOAT

    elif stage_id == 3:
        # Input: messages[E,F] + edge_index[2,E] + num_nodes, Output: sum_agg[N,F]
        input_size = num_edges * feature_dim * BYTES_PER_FLOAT + 2 * num_edges * BYTES_PER_INT
        output_size = num_nodes * feature_dim * BYTES_PER_FLOAT

    elif stage_id == 4:
        # Input: edge_index[2,E] + num_nodes + num_edges, Output: count[N]
        input_size = 2 * num_edges * BYTES_PER_INT
        output_size = num_nodes * BYTES_PER_FLOAT

    elif stage_id == 5:
        # Input: sum_agg[N,F] + count[N], Output: mean_agg[N,F]
        input_size = num_nodes * feature_dim * BYTES_PER_FLOAT + num_nodes * BYTES_PER_FLOAT
        output_size = num_nodes * feature_dim * BYTES_PER_FLOAT

    elif stage_id == 6:
        # Input: mean_agg[N,F] + x[N,F], Output: out[N,256]
        input_size = 2 * num_nodes * feature_dim * BYTES_PER_FLOAT
        output_size = num_nodes * 256 * BYTES_PER_FLOAT  # output is 256

    elif stage_id == 7:
        # Input: out[N,256], Output: activated[N,256]
        # Stage 7 (ReLU) 的输入是 Stage 6 的输出，维度是 256
        input_size = num_nodes * 256 * BYTES_PER_FLOAT
        output_size = num_nodes * 256 * BYTES_PER_FLOAT

    else:
        raise ValueError(f"Invalid stage_id: {stage_id}")

    return input_size, output_size


def calc_transfer_time(input_bytes, output_bytes, device):
    """
    计算传输时间 (ms)

    旧 profiling 使用同步 API compiled_model(inputs)：
    - 即使不重新 set tensor，数据也会传输（test_transfer_time.py 验证）
    - 同步 API 返回输出，包含 device→host 传输

    所以：total_time = 输入传输 + 计算 + 输出传输

    transfer_time = (input_bytes / input_bw + output_bytes / output_bw) * correction
    """
    bw = BANDWIDTH.get(device)
    if not bw:
        return 0.0

    # GB/s = bytes / (ms * 1e6), so ms = bytes / (GB/s * 1e6)
    input_ms = input_bytes / (bw['input'] * 1e6)
    output_ms = output_bytes / (bw['output'] * 1e6)

    # 应用校正系数（减少负数）
    correction = TRANSFER_CORRECTION.get(device, 1.0)

    return (input_ms + output_ms) * correction


def filter_profiling_results(raw_file, output_file):
    """
    过滤 profiling 结果
    """
    with open(raw_file, 'r') as f:
        raw_data = json.load(f)

    filtered = {}
    stats = {'total': 0, 'clamped': 0}

    for key, value in raw_data.items():
        # Parse key: "nodes,edges,device,stage"
        parts = key.split(',')
        if len(parts) != 4:
            continue

        num_nodes = int(parts[0])
        num_edges = int(parts[1])
        device = parts[2]
        stage_id = int(parts[3])

        # Get total time
        total_time = value.get('mean', 0)
        if total_time <= 0:
            continue

        # Calculate IO size
        input_bytes, output_bytes = get_io_size(stage_id, num_nodes, num_edges)

        # Calculate transfer time
        transfer_time = calc_transfer_time(input_bytes, output_bytes, device)

        # 限制传输时间不超过总时间的 MAX_TRANSFER_RATIO
        max_transfer = total_time * MAX_TRANSFER_RATIO
        clamped_transfer = min(transfer_time, max_transfer)

        # Pure compute time
        compute_time = total_time - clamped_transfer

        stats['total'] += 1
        if transfer_time > max_transfer:
            stats['clamped'] += 1

        filtered[key] = {
            'total_time_ms': total_time,
            'estimated_transfer_ms': transfer_time,  # 原始估算
            'clamped_transfer_ms': clamped_transfer,  # 限制后
            'compute_time_ms': compute_time,
            'input_MB': input_bytes / 1e6,
            'output_MB': output_bytes / 1e6,
            'std_ms': value.get('std', 0),
        }

    # Save
    with open(output_file, 'w') as f:
        json.dump(filtered, f, indent=2)

    return filtered, stats


def print_summary(filtered, stats):
    """打印摘要"""
    print(f"\n{'='*70}")
    print("过滤结果摘要")
    print(f"{'='*70}")
    print(f"总记录数: {stats['total']}")
    print(f"传输时间被限制 (>{MAX_TRANSFER_RATIO*100:.0f}%): {stats['clamped']}")

    # Group by device and stage
    by_device_stage = {}
    for key, value in filtered.items():
        parts = key.split(',')
        device = parts[2]
        stage = int(parts[3])

        ds_key = f"{device}_stage{stage}"
        if ds_key not in by_device_stage:
            by_device_stage[ds_key] = []
        by_device_stage[ds_key].append(value)

    print(f"\n{'设备_Stage':<15} {'平均总时间':<12} {'传输(限制后)':<14} {'计算时间':<12} {'计算占比':<10}")
    print("-" * 70)

    for ds_key in sorted(by_device_stage.keys()):
        values = by_device_stage[ds_key]
        avg_total = sum(v['total_time_ms'] for v in values) / len(values)
        avg_transfer = sum(v['clamped_transfer_ms'] for v in values) / len(values)
        avg_compute = sum(v['compute_time_ms'] for v in values) / len(values)
        compute_pct = avg_compute / avg_total * 100 if avg_total > 0 else 0

        print(f"{ds_key:<15} {avg_total:>10.2f}ms {avg_transfer:>12.2f}ms {avg_compute:>10.2f}ms {compute_pct:>8.1f}%")


def create_compute_only_lookup(filtered, output_file):
    """
    创建只有计算时间的 lookup table
    格式: "nodes,edges,device,stage" -> compute_time_ms
    """
    lookup = {}
    for key, value in filtered.items():
        lookup[key] = {
            'compute_time_ms': value['compute_time_ms'],
            'std_ms': value['std_ms'],
        }

    with open(output_file, 'w') as f:
        json.dump(lookup, f, indent=2)

    return lookup


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Filter transfer time from profiling results')
    parser.add_argument('--input', type=str, default='results/185H/raw_measurements.json',
                        help='Input raw measurements file')
    parser.add_argument('--output', type=str, default='results/185H/compute_only.json',
                        help='Output filtered file')
    parser.add_argument('--lookup', type=str, default='results/185H/lookup_compute_only.json',
                        help='Output lookup table file')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_file = script_dir / args.input
    output_file = script_dir / args.output
    lookup_file = script_dir / args.lookup

    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"Lookup 文件: {lookup_file}")

    print(f"\n带宽参数:")
    for device, bw in BANDWIDTH.items():
        print(f"  {device}: 输入 {bw['input']:.2f} GB/s, 输出 {bw['output']:.2f} GB/s")

    # Filter
    filtered, stats = filter_profiling_results(input_file, output_file)
    print(f"\n已保存过滤结果到: {output_file}")

    # Create lookup
    lookup = create_compute_only_lookup(filtered, lookup_file)
    print(f"已保存 Lookup 到: {lookup_file}")

    # Print summary
    print_summary(filtered, stats)


if __name__ == '__main__':
    main()
