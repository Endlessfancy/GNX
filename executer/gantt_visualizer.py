"""
Gantt Chart Visualizer for Pipeline Execution
甘特图可视化工具，用于展示流水线执行的时间线
"""

from typing import Dict


def generate_gantt_chart(detailed_timing: Dict, output_file: str = 'pipeline_gantt.txt') -> str:
    """
    生成ASCII格式的甘特图

    Args:
        detailed_timing: pipeline_executor的detailed_timing数据
        output_file: 输出文件名

    Returns:
        甘特图字符串
    """
    timestamps = detailed_timing['timestamps']
    thread_ids = detailed_timing['thread_ids']

    if not timestamps:
        return "No timing data available for Gantt chart"

    # 找到全局开始时间和结束时间
    all_starts = [ts['start'] for ts in timestamps.values()]
    all_ends = [ts['end'] for ts in timestamps.values()]

    global_start = min(all_starts)
    global_end = max(all_ends)
    total_duration = (global_end - global_start) * 1000  # ms

    # 时间比例尺（每个字符代表多少ms）
    # 目标：甘特图宽度约60-80个字符
    target_width = 70
    time_scale = total_duration / target_width  # ms per character

    lines = []
    lines.append("="*80)
    lines.append("Pipeline Execution Gantt Chart")
    lines.append("="*80)
    lines.append("")
    lines.append(f"Total duration: {total_duration:.0f}ms")
    lines.append(f"Time scale: ~{time_scale:.1f}ms per character")
    lines.append("")

    # 找出所有unique的block_id
    num_blocks = len(thread_ids)
    all_sg_ids = sorted(set(k[0] for k in timestamps.keys()))

    # 为每个block生成一行
    for block_id in range(num_blocks):
        thread_id = thread_ids.get(block_id, 'N/A')
        line_prefix = f"Block {block_id} (Thread-{str(thread_id)[-4:]}): "

        # 构建时间线
        timeline = []

        for sg_id in all_sg_ids:
            key = (sg_id, block_id)
            if key in timestamps:
                ts = timestamps[key]

                # 计算相对于global_start的偏移（字符位置）
                start_offset = int((ts['start'] - global_start) * 1000 / time_scale)
                duration = int((ts['end'] - ts['start']) * 1000 / time_scale)

                # 确保至少显示1个字符
                if duration < 1:
                    duration = 1

                # 填充空白（等待时间）
                while len(timeline) < start_offset:
                    timeline.append(' ')

                # 绘制执行块（用subgraph ID标记）
                sg_char = str(sg_id) if sg_id < 10 else chr(ord('A') + sg_id - 10)
                for _ in range(duration):
                    timeline.append(sg_char)

        lines.append(line_prefix + ''.join(timeline))

    # 添加时间轴标记
    lines.append("")
    time_axis = "Time axis:   "
    num_marks = int(total_duration / 1000) + 1  # 每秒一个标记

    for i in range(num_marks + 1):
        time_ms = i * 1000
        pos = int(time_ms / time_scale)

        # 填充空白到标记位置
        while len(time_axis) < pos + 13:
            time_axis += ' '

        # 添加标记
        mark = f"{time_ms/1000:.1f}s"
        time_axis = time_axis[:pos+13] + mark

    lines.append(time_axis)
    lines.append("")

    # 添加图例
    lines.append("Legend:")
    lines.append("  Each character represents a time slice of the pipeline execution")
    lines.append("  Numbers/letters indicate which subgraph is being processed")
    lines.append("  Gaps indicate waiting time or idle periods")
    lines.append("  Overlapping execution shows pipeline parallelism")
    lines.append("")

    # 分析并行度
    lines.append("Pipeline Overlap Analysis:")

    # 找到流水线重叠的时间段
    overlap_periods = _find_overlaps(timestamps, global_start, time_scale, num_blocks)

    if overlap_periods:
        total_overlap_time = sum(period['duration'] for period in overlap_periods)
        overlap_percentage = (total_overlap_time / total_duration) * 100

        lines.append(f"  Overlapping execution detected: {len(overlap_periods)} periods")
        lines.append(f"  Total overlap time: {total_overlap_time:.0f}ms ({overlap_percentage:.1f}% of total)")
        lines.append(f"  This demonstrates true pipeline parallelism!")
    else:
        lines.append(f"  No overlapping execution detected (sequential execution)")

    lines.append("")
    lines.append("="*80)

    # 组合输出
    gantt_output = '\n'.join(lines)

    # 保存到文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(gantt_output)
    except Exception as e:
        print(f"Warning: Could not save Gantt chart to file: {e}")

    return gantt_output


def _find_overlaps(timestamps: Dict, global_start: float, time_scale: float, num_blocks: int) -> list:
    """
    找到流水线重叠执行的时间段

    Args:
        timestamps: 时间戳数据
        global_start: 全局开始时间
        time_scale: 时间比例尺
        num_blocks: block数量

    Returns:
        重叠时间段列表
    """
    # 为每个block创建时间段列表 [(start_ms, end_ms, sg_id), ...]
    block_periods = {block_id: [] for block_id in range(num_blocks)}

    for (sg_id, block_id), ts in timestamps.items():
        start_ms = (ts['start'] - global_start) * 1000
        end_ms = (ts['end'] - global_start) * 1000
        block_periods[block_id].append((start_ms, end_ms, sg_id))

    # 按开始时间排序
    for block_id in block_periods:
        block_periods[block_id].sort()

    # 找到重叠时间段（至少两个block同时执行）
    overlaps = []

    # 简单算法：遍历时间轴，检查每个时间点有多少block在执行
    if num_blocks < 2:
        return overlaps

    # 创建事件列表（开始和结束事件）
    events = []
    for block_id in range(num_blocks):
        for start_ms, end_ms, sg_id in block_periods[block_id]:
            events.append(('start', start_ms, block_id, sg_id))
            events.append(('end', end_ms, block_id, sg_id))

    # 按时间排序
    events.sort(key=lambda x: (x[1], x[0] == 'end'))  # end在start之前（同时间点）

    # 扫描事件，记录并行度
    active_blocks = set()
    last_time = 0
    overlap_start = None

    for event_type, event_time, block_id, sg_id in events:
        if event_type == 'start':
            # 如果已经有其他block在执行，开始记录重叠
            if len(active_blocks) >= 1 and overlap_start is None:
                overlap_start = event_time

            active_blocks.add(block_id)
        else:  # end
            active_blocks.discard(block_id)

            # 如果重叠结束（只剩一个或零个block）
            if len(active_blocks) < 2 and overlap_start is not None:
                overlaps.append({
                    'start': overlap_start,
                    'end': event_time,
                    'duration': event_time - overlap_start
                })
                overlap_start = None

    return overlaps


def print_detailed_timing_table(detailed_timing: Dict):
    """
    打印详细的时间统计表格

    Args:
        detailed_timing: pipeline_executor的detailed_timing数据
    """
    block_times = detailed_timing['block_times']
    thread_ids = detailed_timing['thread_ids']

    if not block_times:
        print("No detailed timing data available")
        return

    # 找出所有unique的sg_id和block_id
    all_sg_ids = sorted(set(k[0] for k in block_times.keys()))
    all_block_ids = sorted(set(k[1] for k in block_times.keys()))

    print("\n" + "="*80)
    print("Detailed Timing Statistics (by Subgraph and Block)")
    print("="*80)
    print("")

    for block_id in all_block_ids:
        thread_id = thread_ids.get(block_id, 'N/A')
        print(f"\nBlock {block_id} (Thread-{str(thread_id)[-4:]}):")
        print(f"{'Subgraph':<12} {'Wait Time':<12} {'Exec Time':<12} {'Total Time':<12}")
        print("-" * 50)

        total_wait = 0.0
        total_exec = 0.0
        total_total = 0.0

        for sg_id in all_sg_ids:
            key = (sg_id, block_id)
            if key in block_times:
                timing = block_times[key]
                wait = timing['wait']
                exec_time = timing['exec']
                total = timing['total']

                total_wait += wait
                total_exec += exec_time
                total_total += total

                print(f"SG{sg_id:<10} {wait:>10.0f}ms {exec_time:>10.0f}ms {total:>10.0f}ms")

        print("-" * 50)
        print(f"{'Total':<12} {total_wait:>10.0f}ms {total_exec:>10.0f}ms {total_total:>10.0f}ms")
