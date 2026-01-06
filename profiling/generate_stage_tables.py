"""
Generate Stage Profiling Tables from lookup_table.json

按设备分组生成7个stage的profiling表格，每个设备单独一个子表。
"""

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_lookup_table(json_path: str) -> dict:
    """Load lookup_table.json"""
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_lookup_table(data: dict) -> dict:
    """
    Parse lookup table into structured format.

    Returns:
        {stage: {device: {(nodes, edges): time_ms}}}
    """
    result = defaultdict(lambda: defaultdict(dict))

    for key, value in data.items():
        parts = key.split(',')
        nodes = int(parts[0])
        edges = int(parts[1])
        device = parts[2]
        stage = int(parts[3])
        time_ms = value['total_time_ms']

        result[stage][device][(nodes, edges)] = time_ms

    return result


def get_unique_values(parsed_data: dict) -> tuple:
    """Get unique nodes and edges values."""
    all_nodes = set()
    all_edges = set()

    for stage_data in parsed_data.values():
        for device_data in stage_data.values():
            for (nodes, edges) in device_data.keys():
                all_nodes.add(nodes)
                all_edges.add(edges)

    return sorted(all_nodes), sorted(all_edges)


def format_time(time_ms: float) -> str:
    """Format time value for display."""
    if time_ms >= 1000:
        return f"{time_ms/1000:.2f}s"
    elif time_ms >= 100:
        return f"{time_ms:.1f}"
    elif time_ms >= 10:
        return f"{time_ms:.2f}"
    else:
        return f"{time_ms:.3f}"


def generate_stage_table(stage: int, stage_data: dict, all_nodes: list, all_edges: list) -> str:
    """
    Generate markdown table for a single stage, grouped by device.

    Args:
        stage: Stage number (1-7)
        stage_data: {device: {(nodes, edges): time_ms}}
        all_nodes: List of all node counts
        all_edges: List of all edge counts

    Returns:
        Markdown formatted table string
    """
    stage_names = {
        1: "GATHER (Neighbor Feature Gathering)",
        2: "MESSAGE (⚠️ Identity - Check if data is outdated)",
        3: "REDUCE_SUM (Sum Aggregation)",
        4: "REDUCE_COUNT (Count Neighbors)",
        5: "NORMALIZE (Compute Mean)",
        6: "TRANSFORM (Linear Layer)",
        7: "ACTIVATE (ReLU)"
    }

    # Check if stage depends on edges
    edge_dependent = stage <= 4

    lines = []
    lines.append(f"## Stage {stage}: {stage_names.get(stage, 'Unknown')}")
    lines.append("")

    if edge_dependent:
        lines.append("*依赖 nodes 和 edges*")
    else:
        lines.append("*只依赖 nodes (edges 无影响)*")
    lines.append("")

    devices = ['CPU', 'GPU', 'NPU']

    for device in devices:
        if device not in stage_data:
            continue

        device_data = stage_data[device]
        lines.append(f"### {device}")
        lines.append("")

        if edge_dependent:
            # Create a table with nodes as rows, edges as columns
            # Find which edges are available for this device
            available_edges = sorted(set(e for (n, e) in device_data.keys()))

            # Header
            header = "| Nodes |"
            for e in available_edges:
                if e >= 1000000:
                    header += f" {e//1000000}M |"
                elif e >= 1000:
                    header += f" {e//1000}k |"
                else:
                    header += f" {e} |"
            lines.append(header)

            # Separator
            sep = "|-------|"
            for _ in available_edges:
                sep += "--------|"
            lines.append(sep)

            # Data rows
            available_nodes = sorted(set(n for (n, e) in device_data.keys()))
            for nodes in available_nodes:
                if nodes >= 1000:
                    row = f"| {nodes//1000}k |"
                else:
                    row = f"| {nodes} |"

                for edges in available_edges:
                    if (nodes, edges) in device_data:
                        time_ms = device_data[(nodes, edges)]
                        row += f" {format_time(time_ms)} |"
                    else:
                        row += " - |"
                lines.append(row)
        else:
            # For stages 5-7, edges don't matter, so just show by nodes
            # Average across different edge counts
            node_times = defaultdict(list)
            for (nodes, edges), time_ms in device_data.items():
                node_times[nodes].append(time_ms)

            # Header
            lines.append("| Nodes | Avg Time (ms) | Min | Max |")
            lines.append("|-------|---------------|-----|-----|")

            for nodes in sorted(node_times.keys()):
                times = node_times[nodes]
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)

                if nodes >= 1000:
                    node_str = f"{nodes//1000}k"
                else:
                    node_str = str(nodes)

                lines.append(f"| {node_str} | {format_time(avg_time)} | {format_time(min_time)} | {format_time(max_time)} |")

        lines.append("")

    return "\n".join(lines)


def generate_summary_table(parsed_data: dict) -> str:
    """Generate a summary comparison table across all stages and devices."""
    lines = []
    lines.append("## Summary: 100k Nodes Performance Comparison")
    lines.append("")
    lines.append("*Using 100k nodes, 1M edges for stages 1-4; 100k nodes for stages 5-7*")
    lines.append("")
    lines.append("| Stage | Description | CPU | GPU | NPU | GPU Speedup | NPU Speedup |")
    lines.append("|-------|-------------|-----|-----|-----|-------------|-------------|")

    stage_names = {
        1: "GATHER",
        2: "MESSAGE",
        3: "REDUCE_SUM",
        4: "REDUCE_COUNT",
        5: "NORMALIZE",
        6: "TRANSFORM",
        7: "ACTIVATE"
    }

    for stage in range(1, 8):
        stage_data = parsed_data.get(stage, {})

        # Get times for 100k nodes
        cpu_time = None
        gpu_time = None
        npu_time = None

        for device in ['CPU', 'GPU', 'NPU']:
            if device not in stage_data:
                continue

            device_data = stage_data[device]

            # Find 100k node entry
            for (nodes, edges), time_ms in device_data.items():
                if nodes == 100000:
                    # Prefer 1M edges for stages 1-4
                    if stage <= 4:
                        if edges == 1000000:
                            if device == 'CPU':
                                cpu_time = time_ms
                            elif device == 'GPU':
                                gpu_time = time_ms
                            else:
                                npu_time = time_ms
                    else:
                        # For stages 5-7, any edge count works (take first)
                        if device == 'CPU' and cpu_time is None:
                            cpu_time = time_ms
                        elif device == 'GPU' and gpu_time is None:
                            gpu_time = time_ms
                        elif device == 'NPU' and npu_time is None:
                            npu_time = time_ms

        # Calculate speedups
        gpu_speedup = f"{cpu_time/gpu_time:.1f}x" if cpu_time and gpu_time else "-"
        npu_speedup = f"{cpu_time/npu_time:.1f}x" if cpu_time and npu_time else "-"

        cpu_str = format_time(cpu_time) if cpu_time else "-"
        gpu_str = format_time(gpu_time) if gpu_time else "-"
        npu_str = format_time(npu_time) if npu_time else "-"

        lines.append(f"| {stage} | {stage_names[stage]} | {cpu_str} | {gpu_str} | {npu_str} | {gpu_speedup} | {npu_speedup} |")

    lines.append("")
    return "\n".join(lines)


def main():
    # Paths
    script_dir = Path(__file__).parent
    json_path = script_dir / "results" / "lookup_table.json"
    output_path = script_dir / "results" / "stage_tables.md"

    print(f"Loading: {json_path}")
    data = load_lookup_table(json_path)
    print(f"Loaded {len(data)} entries")

    # Parse data
    parsed_data = parse_lookup_table(data)
    all_nodes, all_edges = get_unique_values(parsed_data)

    print(f"Nodes: {all_nodes}")
    print(f"Edges: {all_edges}")
    print(f"Stages: {sorted(parsed_data.keys())}")

    # Generate markdown
    lines = []
    lines.append("# Stage Profiling Tables")
    lines.append("")
    lines.append("Generated from `lookup_table.json`")
    lines.append("")
    lines.append("**注意**: Stage 2 (MESSAGE) 时间异常高，可能是旧版本代码生成的数据。")
    lines.append("当前代码中 Stage 2 应该是 identity 操作。")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Add summary table first
    lines.append(generate_summary_table(parsed_data))
    lines.append("---")
    lines.append("")

    # Generate table for each stage
    for stage in range(1, 8):
        if stage in parsed_data:
            lines.append(generate_stage_table(stage, parsed_data[stage], all_nodes, all_edges))
            lines.append("---")
            lines.append("")

    # Write output
    output_content = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_content)

    print(f"\nSaved to: {output_path}")
    print(f"Total lines: {len(lines)}")


if __name__ == "__main__":
    main()
