"""
为三个数据集计算纯计算时间的 lookup table：

1. Profiling (raw_measurements.json): 减去输入+输出传输时间
2. Fused 1-4 (block0_gpu.json): 只减去输入传输时间（异步 API）
3. Fused 1-7 (fused_block0_7_gpu.json): 只减去输入传输时间（异步 API）
"""

import json
from pathlib import Path

# 带宽数据 (GB/s) - 来自 bandwidth_v2.json
BANDWIDTH = {
    'CPU': {'input': 8.754, 'output': 3.434},
    'GPU': {'input': 5.243, 'output': 3.305},
    'NPU': {'input': 1.272, 'output': 2.253},
}

FEATURE_DIM = 500
BYTES_PER_FLOAT = 4
BYTES_PER_INT = 8


def get_stage_io_size(stage_id, num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """获取单独 stage 的输入/输出大小 (bytes)"""
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
        # Stage 7 (ReLU) 的输入是 Stage 6 的输出，维度是 256，不是 500
        input_size = num_nodes * 256 * BYTES_PER_FLOAT
        output_size = num_nodes * 256 * BYTES_PER_FLOAT
    else:
        raise ValueError(f"Invalid stage_id: {stage_id}")
    return input_size, output_size


def get_fused_io_size(fused_type, num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """获取 fused stage 的输入/输出大小"""
    # Fused 输入都是 Stage 1 的输入
    input_size = num_nodes * feature_dim * BYTES_PER_FLOAT + 2 * num_edges * BYTES_PER_INT

    if fused_type == "1-4":
        output_size = num_nodes * BYTES_PER_FLOAT  # Stage 4 output: count[N]
    elif fused_type == "1-7":
        output_size = num_nodes * 256 * BYTES_PER_FLOAT  # Stage 6 output: out[N,256]
    else:
        raise ValueError(f"Invalid fused_type: {fused_type}")
    return input_size, output_size


def calc_transfer_time(input_bytes, output_bytes, device, include_output=True):
    """计算传输时间 (ms)"""
    bw = BANDWIDTH.get(device)
    if not bw:
        return 0.0

    input_ms = input_bytes / (bw['input'] * 1e6)
    output_ms = output_bytes / (bw['output'] * 1e6) if include_output else 0
    return input_ms + output_ms


def process_profiling(input_file, output_file):
    """处理 profiling 数据（减去输入+输出传输时间）"""
    with open(input_file, 'r') as f:
        raw_data = json.load(f)

    result = {}
    stats = {'total': 0, 'negative': 0}

    for key, value in raw_data.items():
        parts = key.split(',')
        if len(parts) != 4:
            continue

        num_nodes = int(parts[0])
        num_edges = int(parts[1])
        device = parts[2]
        stage_id = int(parts[3])

        total_time = value.get('mean', 0)
        if total_time <= 0:
            continue

        # 计算 I/O 大小和传输时间
        input_bytes, output_bytes = get_stage_io_size(stage_id, num_nodes, num_edges)
        transfer_time = calc_transfer_time(input_bytes, output_bytes, device, include_output=True)

        # 纯计算时间
        compute_time = total_time - transfer_time

        stats['total'] += 1
        if compute_time < 0:
            stats['negative'] += 1

        result[key] = {
            'mean': compute_time,
            'std': value.get('std', 0),
            'original_mean': total_time,
            'transfer_time': transfer_time,
        }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    return stats


def process_fused(input_file, output_file, fused_type, key_parser):
    """处理 fused 数据（只减去输入传输时间）"""
    with open(input_file, 'r') as f:
        raw_data = json.load(f)

    result = {}
    stats = {'total': 0, 'negative': 0, 'failed': 0}

    for key, value in raw_data.items():
        if value.get('failed', False):
            stats['failed'] += 1
            continue

        # 解析 key
        num_nodes, num_edges, device = key_parser(key)
        if num_nodes is None:
            continue

        total_time = value.get('mean', 0)
        if total_time <= 0:
            continue

        # 计算 I/O 大小和传输时间（只算输入）
        input_bytes, output_bytes = get_fused_io_size(fused_type, num_nodes, num_edges)
        transfer_time = calc_transfer_time(input_bytes, output_bytes, device, include_output=False)

        # 纯计算时间
        compute_time = total_time - transfer_time

        stats['total'] += 1
        if compute_time < 0:
            stats['negative'] += 1

        result[key] = {
            'mean': compute_time,
            'std': value.get('std', 0),
            'original_mean': total_time,
            'transfer_time': transfer_time,
        }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    return stats


def parse_fused_1_4_key(key):
    """解析 fused 1-4 的 key: "nodes,edges,device,block0" """
    parts = key.split(',')
    if len(parts) != 4:
        return None, None, None
    return int(parts[0]), int(parts[1]), parts[2]


def parse_fused_1_7_key(key):
    """解析 fused 1-7 的 key: "fused0_7,nodes,edges,device" """
    parts = key.split(',')
    if len(parts) != 4:
        return None, None, None
    return int(parts[1]), int(parts[2]), parts[3]


def main():
    base_dir = Path(__file__).parent

    print("=" * 70)
    print("计算纯计算时间 Lookup Tables")
    print("=" * 70)

    # 1. Profiling
    print("\n[1] Profiling (raw_measurements.json)")
    print("    方法: 减去输入+输出传输时间")
    profiling_input = base_dir / 'results/185H/raw_measurements.json'
    profiling_output = base_dir / 'results/185H/compute_only.json'
    stats = process_profiling(profiling_input, profiling_output)
    print(f"    总记录: {stats['total']}, 负值: {stats['negative']} ({stats['negative']/stats['total']*100:.1f}%)")
    print(f"    保存到: {profiling_output}")

    # 2. Fused 1-4
    print("\n[2] Fused 1-4 (block0_gpu.json)")
    print("    方法: 只减去输入传输时间")
    fused_1_4_input = base_dir / '../profiling_fused/results/185H/Test2/block0_gpu.json'
    fused_1_4_output = base_dir / '../profiling_fused/results/185H/Test2/block0_gpu_compute_only.json'
    stats = process_fused(fused_1_4_input, fused_1_4_output, "1-4", parse_fused_1_4_key)
    print(f"    总记录: {stats['total']}, 负值: {stats['negative']} ({stats['negative']/max(stats['total'],1)*100:.1f}%), 失败: {stats['failed']}")
    print(f"    保存到: {fused_1_4_output}")

    # 3. Fused 1-7
    print("\n[3] Fused 1-7 (fused_block0_7_gpu.json)")
    print("    方法: 只减去输入传输时间")
    fused_1_7_input = base_dir / '../experiments/baseline_fused1-7_profiling/results/185H/fused_block0_7_gpu.json'
    fused_1_7_output = base_dir / '../experiments/baseline_fused1-7_profiling/results/185H/fused_block0_7_gpu_compute_only.json'
    stats = process_fused(fused_1_7_input, fused_1_7_output, "1-7", parse_fused_1_7_key)
    print(f"    总记录: {stats['total']}, 负值: {stats['negative']} ({stats['negative']/max(stats['total'],1)*100:.1f}%), 失败: {stats['failed']}")
    print(f"    保存到: {fused_1_7_output}")

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
