"""
Pipeline Scheduler using Johnson's Rule for 2-stage pipeline optimization.

This module implements Johnson's Rule to find the optimal scheduling order
for N subgraphs on a 2-stage pipeline, minimizing total makespan.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import interpolator directly to avoid circular imports
import numpy as np
from scipy.interpolate import interp2d


class Interpolator2D:
    """
    2D interpolator for profiling lookup table queries.
    Embedded version to avoid import issues.
    """

    def __init__(self, data_points: Dict[Tuple[int, int], float]):
        """
        Initialize interpolator.

        Args:
            data_points: {(n, m): time_ms} dictionary
        """
        self.data_points = data_points

        # Extract all n and m values
        points = list(data_points.keys())
        self.n_values = sorted(set(p[0] for p in points))
        self.m_values = sorted(set(p[1] for p in points))

        # Build 2D grid
        self.n_grid, self.m_grid = np.meshgrid(self.n_values, self.m_values, indexing='ij')
        self.z_grid = np.zeros_like(self.n_grid, dtype=float)

        # Fill known data points
        for i, n in enumerate(self.n_values):
            for j, m in enumerate(self.m_values):
                if (n, m) in data_points:
                    self.z_grid[i, j] = data_points[(n, m)]
                else:
                    # For missing points, use nearest neighbor
                    self.z_grid[i, j] = self._nearest_neighbor(n, m)

        # Create interpolation function
        try:
            self.interp_func = interp2d(
                self.n_values, self.m_values, self.z_grid.T,
                kind='linear', bounds_error=False, fill_value=None
            )
        except Exception:
            # If not enough data points, fallback to nearest neighbor
            self.interp_func = None

    def _nearest_neighbor(self, n: int, m: int) -> float:
        """Nearest neighbor lookup"""
        min_dist = float('inf')
        nearest_val = 0.0

        for (n_data, m_data), val in self.data_points.items():
            dist = ((n - n_data) ** 2 + (m - m_data) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_val = val

        return nearest_val

    def query(self, n: int, m: int) -> float:
        """
        Query the interpolated value for (n, m).

        Args:
            n: Number of nodes
            m: Number of edges

        Returns:
            Interpolated time (ms)
        """
        # Check if in known points
        if (n, m) in self.data_points:
            return self.data_points[(n, m)]

        # Use interpolation
        if self.interp_func is not None:
            try:
                result = self.interp_func(n, m)[0]
                # Ensure non-negative
                return max(0.0, float(result))
            except Exception:
                pass

        # Fallback to nearest neighbor
        return self._nearest_neighbor(n, m)


def parse_profiling_markdown(filepath: str) -> Dict[Tuple[int, int], float]:
    """
    Parse profiling data from markdown file.

    Supports two formats:
    1. CPU format: [5000n, 50000e]... 71.07ms
    2. NPU format: ratio= 10 (   50000 edges)... 12.30ms
       (with node count from section header like "Testing 5000 nodes")

    Args:
        filepath: Path to markdown file

    Returns:
        Dictionary of {(nodes, edges): time_ms}
    """
    import re
    data_points = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern 1: CPU format - [5000n, 50000e]... 71.07ms
    pattern_cpu = r'\[(\d+)n,\s*(\d+)e\]\.\.\.\s*([\d.]+)ms'

    for match in re.finditer(pattern_cpu, content):
        n = int(match.group(1))
        m = int(match.group(2))
        time_ms = float(match.group(3))
        data_points[(n, m)] = time_ms

    # Pattern 2: NPU format - need to track current node count from headers
    # Header: "Testing 5000 nodes" or "NPU Testing: 5000 nodes"
    # Data: ratio= 10 (   50000 edges)... 12.30ms
    if not data_points:  # Try NPU format if CPU format found nothing
        current_nodes = None
        lines = content.split('\n')

        for line in lines:
            # Check for node header
            node_match = re.search(r'Testing[:\s]+(\d+)\s+nodes', line)
            if node_match:
                current_nodes = int(node_match.group(1))
                continue

            # Check for ratio line with edges and time
            if current_nodes:
                # Pattern: ratio= 10 (   50000 edges)... 12.30ms
                ratio_match = re.search(r'ratio=\s*\d+\s*\(\s*(\d+)\s*edges\)\.\.\.\s*([\d.]+)ms', line)
                if ratio_match:
                    m = int(ratio_match.group(1))
                    time_ms = float(ratio_match.group(2))
                    data_points[(current_nodes, m)] = time_ms

    return data_points


def parse_partition_markdown(filepath: str) -> List[Tuple[int, int, int]]:
    """
    Parse partition data from markdown file.

    Expected format (table):
    | ID | Owned | Halo | Total Nodes | Internal Edges | Boundary Edges | Halo% |
    | 0 | 22,616 | 71,565 | 94,181 | 2,128,304 | 370,652 | 316.4% |

    Args:
        filepath: Path to markdown file

    Returns:
        List of (id, total_nodes, internal_edges) tuples
    """
    import re
    partitions = []

    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern for table rows: | 0 | 22,616 | 71,565 | 94,181 | 2,128,304 | ...
    pattern = r'\|\s*(\d+)\s*\|\s*[\d,]+\s*\|\s*[\d,]+\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|'

    for match in re.finditer(pattern, content):
        id_ = int(match.group(1))
        total_nodes = int(match.group(2).replace(',', ''))
        internal_edges = int(match.group(3).replace(',', ''))
        partitions.append((id_, total_nodes, internal_edges))

    return partitions


@dataclass
class SubgraphInfo:
    """Information about a subgraph"""
    id: int
    total_nodes: int
    internal_edges: int
    stage1_time: float = 0.0
    stage2_time: float = 0.0
    # Breakdown for detailed analysis
    stage1_compute: float = 0.0
    stage1_transfer: float = 0.0
    stage2_compute: float = 0.0
    stage2_transfer: float = 0.0
    # DP (Data Parallel) split info
    dp_split_ratio: float = 0.0  # CPU处理的比例 (alpha)
    cpu_compute_time: float = 0.0  # CPU部分计算时间
    gpu_compute_time: float = 0.0  # GPU部分计算时间
    # Separate transfer times for CPU and GPU
    cpu_transfer_time: float = 0.0  # CPU数据传输时间
    gpu_transfer_time: float = 0.0  # GPU数据传输时间
    cpu_total_time: float = 0.0     # CPU总时间 (compute + transfer)
    gpu_total_time: float = 0.0     # GPU总时间 (compute + transfer)
    # Detailed transfer breakdown (in/out times and sizes)
    cpu_transfer_in_time: float = 0.0    # CPU数据传入时间
    cpu_transfer_out_time: float = 0.0   # CPU数据传出时间
    gpu_transfer_in_time: float = 0.0    # GPU数据传入时间
    gpu_transfer_out_time: float = 0.0   # GPU数据传出时间
    npu_transfer_in_time: float = 0.0    # NPU数据传入时间
    npu_transfer_out_time: float = 0.0   # NPU数据传出时间
    cpu_transfer_in_size: float = 0.0    # CPU数据传入大小 (bytes)
    cpu_transfer_out_size: float = 0.0   # CPU数据传出大小 (bytes)
    gpu_transfer_in_size: float = 0.0    # GPU数据传入大小 (bytes)
    gpu_transfer_out_size: float = 0.0   # GPU数据传出大小 (bytes)
    npu_transfer_in_size: float = 0.0    # NPU数据传入大小 (bytes)
    npu_transfer_out_size: float = 0.0   # NPU数据传出大小 (bytes)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'total_nodes': self.total_nodes,
            'internal_edges': self.internal_edges,
            'stage1': {
                'time': self.stage1_time,
                'compute': self.stage1_compute,
                'transfer': self.stage1_transfer,
            },
            'stage2': {
                'time': self.stage2_time,
                'compute': self.stage2_compute,
                'transfer': self.stage2_transfer,
            },
            'dp_split': {
                'ratio': self.dp_split_ratio,
                'cpu': {
                    'compute': self.cpu_compute_time,
                    'transfer': self.cpu_transfer_time,
                    'transfer_in': {
                        'time_ms': self.cpu_transfer_in_time,
                        'size_bytes': self.cpu_transfer_in_size,
                    },
                    'transfer_out': {
                        'time_ms': self.cpu_transfer_out_time,
                        'size_bytes': self.cpu_transfer_out_size,
                    },
                    'total': self.cpu_total_time,
                },
                'gpu': {
                    'compute': self.gpu_compute_time,
                    'transfer': self.gpu_transfer_time,
                    'transfer_in': {
                        'time_ms': self.gpu_transfer_in_time,
                        'size_bytes': self.gpu_transfer_in_size,
                    },
                    'transfer_out': {
                        'time_ms': self.gpu_transfer_out_time,
                        'size_bytes': self.gpu_transfer_out_size,
                    },
                    'total': self.gpu_total_time,
                },
                'npu': {
                    'transfer_in': {
                        'time_ms': self.npu_transfer_in_time,
                        'size_bytes': self.npu_transfer_in_size,
                    },
                    'transfer_out': {
                        'time_ms': self.npu_transfer_out_time,
                        'size_bytes': self.npu_transfer_out_size,
                    },
                },
            },
        }


@dataclass
class CycleDeviceTimes:
    """Device times for a single cycle (for utilization calculation)"""
    cycle_time: float = 0.0      # Total cycle time (sync point)

    # Compute times
    cpu_compute: float = 0.0     # CPU compute time
    gpu_compute: float = 0.0     # GPU compute time
    npu_compute: float = 0.0     # NPU compute time

    # Transfer times (total)
    cpu_transfer: float = 0.0    # CPU transfer time
    gpu_transfer: float = 0.0    # GPU transfer time
    npu_transfer: float = 0.0    # NPU transfer time

    # Transfer times (in/out breakdown)
    cpu_transfer_in: float = 0.0     # CPU data transfer in time
    cpu_transfer_out: float = 0.0    # CPU data transfer out time
    gpu_transfer_in: float = 0.0     # GPU data transfer in time
    gpu_transfer_out: float = 0.0    # GPU data transfer out time
    npu_transfer_in: float = 0.0     # NPU data transfer in time
    npu_transfer_out: float = 0.0    # NPU data transfer out time

    # Transfer sizes (in/out breakdown, in bytes)
    cpu_transfer_in_size: float = 0.0     # CPU data transfer in size
    cpu_transfer_out_size: float = 0.0    # CPU data transfer out size
    gpu_transfer_in_size: float = 0.0     # GPU data transfer in size
    gpu_transfer_out_size: float = 0.0    # GPU data transfer out size
    npu_transfer_in_size: float = 0.0     # NPU data transfer in size
    npu_transfer_out_size: float = 0.0    # NPU data transfer out size

    # Total times (compute + transfer)
    cpu_total: float = 0.0       # CPU total time
    gpu_total: float = 0.0       # GPU total time
    npu_total: float = 0.0       # NPU total time

    # DP split ratio for this cycle's Stage 1 subgraph
    dp_split_ratio: float = 0.0  # α (CPU portion)

    # Derived utilization (total_time / cycle_time)
    cpu_util: float = 0.0
    gpu_util: float = 0.0
    npu_util: float = 0.0

    # Subgraph IDs active in this cycle
    stage1_subgraph_id: int = -1  # Subgraph doing Stage 1 in this cycle (-1 if none)
    stage2_subgraph_id: int = -1  # Subgraph doing Stage 2 in this cycle (-1 if none)

    def compute_utilization(self):
        """Compute utilization for each device based on total time"""
        if self.cycle_time > 0:
            self.cpu_util = self.cpu_total / self.cycle_time
            self.gpu_util = self.gpu_total / self.cycle_time
            self.npu_util = self.npu_total / self.cycle_time

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'cycle_time': self.cycle_time,
            'stage1_subgraph_id': self.stage1_subgraph_id,
            'stage2_subgraph_id': self.stage2_subgraph_id,
            'dp_split_ratio': self.dp_split_ratio,
            'cpu': {
                'compute': self.cpu_compute,
                'transfer': self.cpu_transfer,
                'transfer_in': {
                    'time_ms': self.cpu_transfer_in,
                    'size_bytes': self.cpu_transfer_in_size,
                },
                'transfer_out': {
                    'time_ms': self.cpu_transfer_out,
                    'size_bytes': self.cpu_transfer_out_size,
                },
                'total': self.cpu_total,
                'utilization': self.cpu_util,
            },
            'gpu': {
                'compute': self.gpu_compute,
                'transfer': self.gpu_transfer,
                'transfer_in': {
                    'time_ms': self.gpu_transfer_in,
                    'size_bytes': self.gpu_transfer_in_size,
                },
                'transfer_out': {
                    'time_ms': self.gpu_transfer_out,
                    'size_bytes': self.gpu_transfer_out_size,
                },
                'total': self.gpu_total,
                'utilization': self.gpu_util,
            },
            'npu': {
                'compute': self.npu_compute,
                'transfer': self.npu_transfer,
                'transfer_in': {
                    'time_ms': self.npu_transfer_in,
                    'size_bytes': self.npu_transfer_in_size,
                },
                'transfer_out': {
                    'time_ms': self.npu_transfer_out,
                    'size_bytes': self.npu_transfer_out_size,
                },
                'total': self.npu_total,
                'utilization': self.npu_util,
            },
        }


@dataclass
class ScheduleResult:
    """Result of pipeline scheduling"""
    order: List[int]  # Subgraph IDs in optimal order
    makespan: float  # Total time in ms
    stage1_times: List[float]  # Stage 1 time for each subgraph (in order)
    stage2_times: List[float]  # Stage 2 time for each subgraph (in order)
    stage1_end_times: List[float]  # When each subgraph finishes stage 1
    stage2_end_times: List[float]  # When each subgraph finishes stage 2
    cycle_times: List[float] = None  # Time for each cycle (N+1 cycles for N subgraphs)
    cycle_device_times: List[CycleDeviceTimes] = None  # Per-cycle device breakdown

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        # Calculate average utilization
        if self.cycle_device_times and self.makespan > 0:
            total_cpu = sum(cdt.cpu_total for cdt in self.cycle_device_times)
            total_gpu = sum(cdt.gpu_total for cdt in self.cycle_device_times)
            total_npu = sum(cdt.npu_total for cdt in self.cycle_device_times)
            avg_cpu_util = total_cpu / self.makespan
            avg_gpu_util = total_gpu / self.makespan
            avg_npu_util = total_npu / self.makespan
        else:
            avg_cpu_util = avg_gpu_util = avg_npu_util = 0.0

        return {
            'order': self.order,
            'makespan': self.makespan,
            'num_cycles': len(self.cycle_times) if self.cycle_times else 0,
            'average_utilization': {
                'cpu': avg_cpu_util,
                'gpu': avg_gpu_util,
                'npu': avg_npu_util,
            },
            'per_subgraph': [
                {
                    'subgraph_id': self.order[i],
                    'stage1_time': self.stage1_times[i],
                    'stage2_time': self.stage2_times[i],
                    'stage1_end_time': self.stage1_end_times[i],
                    'stage2_end_time': self.stage2_end_times[i],
                }
                for i in range(len(self.order))
            ],
            'cycles': [
                cdt.to_dict() for cdt in self.cycle_device_times
            ] if self.cycle_device_times else [],
        }

    def save_to_json(self, filepath: str, subgraphs: List['SubgraphInfo'] = None):
        """
        Save result to JSON file.

        Args:
            filepath: Output JSON file path
            subgraphs: Optional list of SubgraphInfo to include detailed info
        """
        data = self.to_dict()

        # Add subgraph details if provided
        if subgraphs:
            sg_dict = {sg.id: sg for sg in subgraphs}
            data['subgraphs'] = [
                sg_dict[sg_id].to_dict() for sg_id in self.order
            ]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filepath}")


class PipelineScheduler:
    """
    Johnson's Rule based 2-stage pipeline scheduler.

    Johnson's Rule provides optimal scheduling for 2-machine flow shop problems,
    minimizing the total makespan (completion time of the last job).
    """

    def __init__(
        self,
        profiling_dir: str,
        stage1_stages: List[int] = [1, 2, 3, 4],
        stage2_stages: List[int] = [5, 6, 7],
        device: str = 'CPU',
        # Memory transfer parameters
        stage1_bw_gbps: float = 10.0,  # Bandwidth to stage1 PU in GB/s
        stage2_bw_gbps: float = 10.0,  # Bandwidth to stage2 PU in GB/s
        feature_dim: int = 128,        # Node feature dimension
        bytes_per_element: int = 4,    # sizeof(float) = 4 bytes
    ):
        """
        Initialize the pipeline scheduler.

        Args:
            profiling_dir: Path to profiling results directory containing lookup_table.json
            stage1_stages: List of primitive stage numbers for pipeline stage 1 (default: [1,2,3,4])
            stage2_stages: List of primitive stage numbers for pipeline stage 2 (default: [5,6,7])
            device: Device to use for time lookup (CPU/GPU/NPU)
            stage1_bw_gbps: Memory bandwidth from CPU to stage1 PU in GB/s
            stage2_bw_gbps: Memory bandwidth from CPU to stage2 PU in GB/s
            feature_dim: Node feature dimension for data size estimation
            bytes_per_element: Bytes per data element (4 for float32)
        """
        self.stage1_stages = stage1_stages
        self.stage2_stages = stage2_stages
        self.device = device

        # Memory transfer parameters
        self.stage1_bw_gbps = stage1_bw_gbps
        self.stage2_bw_gbps = stage2_bw_gbps
        self.feature_dim = feature_dim
        self.bytes_per_element = bytes_per_element

        # Load profiling data
        lookup_table_path = Path(profiling_dir) / "lookup_table.json"
        with open(lookup_table_path, 'r') as f:
            self.lookup_table = json.load(f)

        # Build interpolators for each stage
        self.interpolators: Dict[int, Interpolator2D] = {}
        self._build_interpolators()

    @classmethod
    def from_markdown_files(
        cls,
        cpu_profiling_file: str,
        gpu_profiling_file: str,
        npu_profiling_file: str,
        cpu_bw_gbps: float = 50.0,
        gpu_bw_gbps: float = 16.0,
        npu_bw_gbps: float = 10.0,
        feature_dim: int = 500,  # Default from profiling model
        bytes_per_element: int = 4,
        gpu_edge_limit: int = 2_500_000,
        dp_max_iterations: int = 9,
    ) -> 'PipelineScheduler':
        """
        Create a PipelineScheduler from markdown profiling files.

        Args:
            cpu_profiling_file: Path to CPU profiling data (Stage 1)
            gpu_profiling_file: Path to GPU profiling data (Stage 1 DP)
            npu_profiling_file: Path to NPU profiling data (Stage 2)
            cpu_bw_gbps: CPU memory bandwidth in GB/s (default: 50 for DDR4)
            gpu_bw_gbps: GPU PCIe bandwidth in GB/s (default: 16 for PCIe 3.0 x16)
            npu_bw_gbps: NPU bandwidth in GB/s (default: 10)
            feature_dim: Node feature dimension
            bytes_per_element: Bytes per data element
            gpu_edge_limit: Maximum edges GPU can handle
            dp_max_iterations: Max iterations for binary search

        Returns:
            PipelineScheduler instance
        """
        # Create instance without calling normal __init__
        instance = cls.__new__(cls)

        # Set attributes
        instance.stage1_stages = ['fused_1234']  # Placeholder
        instance.stage2_stages = ['fused_567']   # Placeholder
        instance.device = 'mixed'

        instance.cpu_bw_gbps = cpu_bw_gbps
        instance.gpu_bw_gbps = gpu_bw_gbps
        instance.npu_bw_gbps = npu_bw_gbps
        # Backward compatibility
        instance.stage1_bw_gbps = cpu_bw_gbps
        instance.stage2_bw_gbps = npu_bw_gbps
        instance.feature_dim = feature_dim
        instance.bytes_per_element = bytes_per_element
        instance.gpu_edge_limit = gpu_edge_limit
        instance.dp_max_iterations = dp_max_iterations

        instance.lookup_table = None

        # Parse profiling data from markdown files
        cpu_data = parse_profiling_markdown(cpu_profiling_file)
        gpu_data = parse_profiling_markdown(gpu_profiling_file)
        npu_data = parse_profiling_markdown(npu_profiling_file)

        print(f"Loaded {len(cpu_data)} data points for CPU (Stage 1)")
        print(f"Loaded {len(gpu_data)} data points for GPU (Stage 1 DP)")
        print(f"Loaded {len(npu_data)} data points for NPU (Stage 2)")

        # Build interpolators directly
        instance.interpolators = {}
        instance.cpu_interpolator = Interpolator2D(cpu_data) if cpu_data else None
        instance.gpu_interpolator = Interpolator2D(gpu_data) if gpu_data else None
        instance.npu_interpolator = Interpolator2D(npu_data) if npu_data else None

        # For backward compatibility
        instance.stage1_interpolator = instance.cpu_interpolator
        instance.stage2_interpolator = instance.npu_interpolator

        return instance

    def _build_interpolators(self):
        """Build Interpolator2D instances for each primitive stage."""
        all_stages = set(self.stage1_stages) | set(self.stage2_stages)

        for stage in all_stages:
            # Extract data points for this stage and device
            data_points: Dict[Tuple[int, int], float] = {}

            for key, value in self.lookup_table.items():
                parts = key.split(',')
                if len(parts) != 4:
                    continue

                n, m, dev, s = int(parts[0]), int(parts[1]), parts[2], int(parts[3])

                if dev == self.device and s == stage:
                    data_points[(n, m)] = value['total_time_ms']

            if data_points:
                self.interpolators[stage] = Interpolator2D(data_points)
            else:
                print(f"Warning: No data points found for stage {stage}, device {self.device}")

    def get_stage_time(self, n: int, m: int, stages: List[int]) -> float:
        """
        Get total compute time for a set of primitive stages using interpolation.

        Args:
            n: Number of nodes in the subgraph
            m: Number of edges in the subgraph
            stages: List of primitive stage numbers to sum

        Returns:
            Total compute time in ms for all specified stages
        """
        total_time = 0.0

        for stage in stages:
            if stage in self.interpolators:
                total_time += self.interpolators[stage].query(n, m)
            else:
                print(f"Warning: No interpolator for stage {stage}, using 0")

        return total_time

    def estimate_data_size_bytes(self, n: int, m: int) -> float:
        """
        Estimate data size in bytes for a subgraph (legacy, for compatibility).

        Args:
            n: Number of nodes
            m: Number of edges

        Returns:
            Estimated data size in bytes
        """
        node_data = n * self.feature_dim * self.bytes_per_element
        edge_indices = m * 2 * 8  # COO format: (src, dst) as int64
        return node_data + edge_indices

    def get_stage1_input_size(self, n: int, m: int) -> float:
        """
        Stage 1-4 input size: x [n, feat_dim] + edge_index [2, m]

        Args:
            n: Number of nodes
            m: Number of edges

        Returns:
            Input size in bytes
        """
        x_size = n * self.feature_dim * self.bytes_per_element  # [n, feat_dim] float32
        edge_index_size = m * 2 * 8  # [2, m] int64
        return x_size + edge_index_size

    def get_stage1_output_size(self, n: int) -> float:
        """
        Stage 1-4 output size: sum_agg [n, feat_dim] + count [n]

        Args:
            n: Number of nodes

        Returns:
            Output size in bytes
        """
        sum_agg_size = n * self.feature_dim * self.bytes_per_element  # [n, feat_dim] float32
        count_size = n * self.bytes_per_element  # [n] float32
        return sum_agg_size + count_size

    def get_stage2_input_size(self, n: int) -> float:
        """
        Stage 5-7 input size: sum_agg [n, feat_dim] + count [n] + x [n, feat_dim]

        Args:
            n: Number of nodes

        Returns:
            Input size in bytes
        """
        sum_agg_size = n * self.feature_dim * self.bytes_per_element
        count_size = n * self.bytes_per_element
        x_size = n * self.feature_dim * self.bytes_per_element
        return sum_agg_size + count_size + x_size

    def get_stage2_output_size(self, n: int) -> float:
        """
        Stage 5-7 output size: activated [n, feat_dim]

        Args:
            n: Number of nodes

        Returns:
            Output size in bytes
        """
        return n * self.feature_dim * self.bytes_per_element

    def get_transfer_time_ms(self, n: int, m: int, bw_gbps: float) -> float:
        """
        Calculate round-trip memory transfer time in milliseconds (legacy).

        Args:
            n: Number of nodes
            m: Number of edges
            bw_gbps: Bandwidth in GB/s

        Returns:
            Round-trip transfer time in ms
        """
        data_size_bytes = self.estimate_data_size_bytes(n, m)
        transfer_time_ms = 2 * data_size_bytes / (bw_gbps * 1e9) * 1000
        return transfer_time_ms

    def get_stage1_transfer_time(self, n: int, m: int, bw_gbps: float) -> float:
        """
        Stage 1-4 transfer time: input + output.

        Args:
            n: Number of nodes
            m: Number of edges
            bw_gbps: Bandwidth in GB/s

        Returns:
            Transfer time in ms
        """
        input_size = self.get_stage1_input_size(n, m)
        output_size = self.get_stage1_output_size(n)
        total_bytes = input_size + output_size
        return total_bytes / (bw_gbps * 1e9) * 1000

    def get_stage2_transfer_time(self, n: int, bw_gbps: float) -> float:
        """
        Stage 5-7 transfer time: input + output (edge-independent).

        Args:
            n: Number of nodes
            bw_gbps: Bandwidth in GB/s

        Returns:
            Transfer time in ms
        """
        input_size = self.get_stage2_input_size(n)
        output_size = self.get_stage2_output_size(n)
        total_bytes = input_size + output_size
        return total_bytes / (bw_gbps * 1e9) * 1000

    def get_stage1_transfer_time_detailed(
        self, n: int, m: int, bw_gbps: float
    ) -> Tuple[float, float, float, float]:
        """
        Stage 1-4 transfer time with detailed breakdown.

        Args:
            n: Number of nodes
            m: Number of edges
            bw_gbps: Bandwidth in GB/s

        Returns:
            Tuple of (transfer_in_time_ms, transfer_out_time_ms, transfer_in_size_bytes, transfer_out_size_bytes)
        """
        input_size = self.get_stage1_input_size(n, m)
        output_size = self.get_stage1_output_size(n)
        transfer_in_time = input_size / (bw_gbps * 1e9) * 1000
        transfer_out_time = output_size / (bw_gbps * 1e9) * 1000
        return transfer_in_time, transfer_out_time, input_size, output_size

    def get_stage2_transfer_time_detailed(
        self, n: int, bw_gbps: float
    ) -> Tuple[float, float, float, float]:
        """
        Stage 5-7 transfer time with detailed breakdown (edge-independent).

        Args:
            n: Number of nodes
            bw_gbps: Bandwidth in GB/s

        Returns:
            Tuple of (transfer_in_time_ms, transfer_out_time_ms, transfer_in_size_bytes, transfer_out_size_bytes)
        """
        input_size = self.get_stage2_input_size(n)
        output_size = self.get_stage2_output_size(n)
        transfer_in_time = input_size / (bw_gbps * 1e9) * 1000
        transfer_out_time = output_size / (bw_gbps * 1e9) * 1000
        return transfer_in_time, transfer_out_time, input_size, output_size

    def find_optimal_dp_split(
        self,
        n: int,
        m: int,
        max_iterations: int = None,
        gpu_edge_limit: int = None
    ) -> dict:
        """
        Use binary search to find optimal CPU/GPU split ratio for data parallelism.

        Optimizes: min(max(cpu_compute + cpu_transfer, gpu_compute + gpu_transfer))

        Args:
            n: Number of nodes
            m: Number of edges
            max_iterations: Maximum binary search iterations (default: self.dp_max_iterations or 9)
            gpu_edge_limit: Maximum edges GPU can handle (default: self.gpu_edge_limit or 2.5M)

        Returns:
            Dictionary with detailed transfer info:
            {
                'alpha': float,
                'cpu': {compute, transfer, transfer_in, transfer_out, transfer_in_size, transfer_out_size, total},
                'gpu': {compute, transfer, transfer_in, transfer_out, transfer_in_size, transfer_out_size, total},
            }
        """
        if max_iterations is None:
            max_iterations = getattr(self, 'dp_max_iterations', 9)
        if gpu_edge_limit is None:
            gpu_edge_limit = getattr(self, 'gpu_edge_limit', 2_500_000)

        # Get bandwidth values
        cpu_bw = getattr(self, 'cpu_bw_gbps', getattr(self, 'stage1_bw_gbps', 50.0))
        gpu_bw = getattr(self, 'gpu_bw_gbps', 16.0)

        # Check if GPU interpolator is available
        if not hasattr(self, 'gpu_interpolator') or self.gpu_interpolator is None:
            # No GPU, use CPU only
            cpu_compute = self.cpu_interpolator.query(n, m) if hasattr(self, 'cpu_interpolator') else 0
            cpu_in_time, cpu_out_time, cpu_in_size, cpu_out_size = self.get_stage1_transfer_time_detailed(n, m, cpu_bw)
            cpu_transfer = cpu_in_time + cpu_out_time
            cpu_total = cpu_compute + cpu_transfer
            return {
                'alpha': 1.0,
                'cpu': {
                    'compute': cpu_compute, 'transfer': cpu_transfer,
                    'transfer_in': cpu_in_time, 'transfer_out': cpu_out_time,
                    'transfer_in_size': cpu_in_size, 'transfer_out_size': cpu_out_size,
                    'total': cpu_total
                },
                'gpu': {
                    'compute': 0.0, 'transfer': 0.0,
                    'transfer_in': 0.0, 'transfer_out': 0.0,
                    'transfer_in_size': 0.0, 'transfer_out_size': 0.0,
                    'total': 0.0
                },
            }

        # Calculate alpha bounds
        # alpha_min: minimum CPU ratio (GPU gets at most gpu_edge_limit edges)
        if m > gpu_edge_limit:
            alpha_min = max(0.01, 1 - gpu_edge_limit / m)
        else:
            alpha_min = 0.01
        alpha_max = 0.99

        # Binary search for optimal split (considering compute + transfer)
        for _ in range(max_iterations):
            alpha = (alpha_min + alpha_max) / 2

            cpu_n, cpu_m = alpha * n, alpha * m
            gpu_n, gpu_m = (1 - alpha) * n, (1 - alpha) * m

            cpu_compute = self.cpu_interpolator.query(cpu_n, cpu_m)
            gpu_compute = self.gpu_interpolator.query(gpu_n, gpu_m)

            # Use stage1 transfer time (input + output)
            cpu_transfer = self.get_stage1_transfer_time(cpu_n, cpu_m, cpu_bw)
            gpu_transfer = self.get_stage1_transfer_time(gpu_n, gpu_m, gpu_bw)

            cpu_total = cpu_compute + cpu_transfer
            gpu_total = gpu_compute + gpu_transfer

            if cpu_total > gpu_total:
                # CPU is slower, reduce CPU load
                alpha_max = alpha
            else:
                # GPU is slower, increase CPU load
                alpha_min = alpha

        # Final result with all timing components (detailed)
        cpu_n, cpu_m = alpha * n, alpha * m
        gpu_n, gpu_m = (1 - alpha) * n, (1 - alpha) * m

        cpu_compute = self.cpu_interpolator.query(cpu_n, cpu_m)
        gpu_compute = self.gpu_interpolator.query(gpu_n, gpu_m)

        # Use detailed stage1 transfer time (input + output separate)
        cpu_in_time, cpu_out_time, cpu_in_size, cpu_out_size = self.get_stage1_transfer_time_detailed(cpu_n, cpu_m, cpu_bw)
        gpu_in_time, gpu_out_time, gpu_in_size, gpu_out_size = self.get_stage1_transfer_time_detailed(gpu_n, gpu_m, gpu_bw)

        cpu_transfer = cpu_in_time + cpu_out_time
        gpu_transfer = gpu_in_time + gpu_out_time

        cpu_total = cpu_compute + cpu_transfer
        gpu_total = gpu_compute + gpu_transfer

        return {
            'alpha': alpha,
            'cpu': {
                'compute': cpu_compute, 'transfer': cpu_transfer,
                'transfer_in': cpu_in_time, 'transfer_out': cpu_out_time,
                'transfer_in_size': cpu_in_size, 'transfer_out_size': cpu_out_size,
                'total': cpu_total
            },
            'gpu': {
                'compute': gpu_compute, 'transfer': gpu_transfer,
                'transfer_in': gpu_in_time, 'transfer_out': gpu_out_time,
                'transfer_in_size': gpu_in_size, 'transfer_out_size': gpu_out_size,
                'total': gpu_total
            },
        }

    def get_pipeline_stage_times(self, subgraph: SubgraphInfo) -> Tuple[float, float]:
        """
        Get pipeline stage 1 and stage 2 total times for a subgraph.

        Stage 1: CPU + GPU data parallel (if GPU available)
        Stage 2: NPU only

        Total time = compute_time + round_trip_transfer_time

        Args:
            subgraph: SubgraphInfo with total_nodes and internal_edges

        Returns:
            Tuple of (stage1_total_time, stage2_total_time) in ms
        """
        n, m = subgraph.total_nodes, subgraph.internal_edges

        # Get bandwidth values
        cpu_bw = getattr(self, 'cpu_bw_gbps', getattr(self, 'stage1_bw_gbps', 50.0))
        gpu_bw = getattr(self, 'gpu_bw_gbps', 16.0)
        npu_bw = getattr(self, 'npu_bw_gbps', getattr(self, 'stage2_bw_gbps', 10.0))

        # Stage 1: CPU + GPU Data Parallel
        if hasattr(self, 'gpu_interpolator') and self.gpu_interpolator is not None:
            # Use DP with binary search for optimal split (includes transfer times)
            dp_result = self.find_optimal_dp_split(n, m)
            cpu_info = dp_result['cpu']
            gpu_info = dp_result['gpu']

            # Store DP info
            subgraph.dp_split_ratio = dp_result['alpha']
            subgraph.cpu_compute_time = cpu_info['compute']
            subgraph.gpu_compute_time = gpu_info['compute']
            subgraph.cpu_transfer_time = cpu_info['transfer']
            subgraph.gpu_transfer_time = gpu_info['transfer']
            subgraph.cpu_total_time = cpu_info['total']
            subgraph.gpu_total_time = gpu_info['total']
            # Store detailed transfer info
            subgraph.cpu_transfer_in_time = cpu_info['transfer_in']
            subgraph.cpu_transfer_out_time = cpu_info['transfer_out']
            subgraph.cpu_transfer_in_size = cpu_info['transfer_in_size']
            subgraph.cpu_transfer_out_size = cpu_info['transfer_out_size']
            subgraph.gpu_transfer_in_time = gpu_info['transfer_in']
            subgraph.gpu_transfer_out_time = gpu_info['transfer_out']
            subgraph.gpu_transfer_in_size = gpu_info['transfer_in_size']
            subgraph.gpu_transfer_out_size = gpu_info['transfer_out_size']

            # Stage 1 time = max(cpu_total, gpu_total) - DP parallel execution
            stage1_time = max(cpu_info['total'], gpu_info['total'])
            # For compatibility, set stage1_compute as max of compute times
            stage1_compute = max(cpu_info['compute'], gpu_info['compute'])
            stage1_transfer = max(cpu_info['transfer'], gpu_info['transfer'])  # Approximate

        elif hasattr(self, 'cpu_interpolator') and self.cpu_interpolator:
            # CPU only (no GPU available)
            cpu_compute = self.cpu_interpolator.query(n, m)
            cpu_in_time, cpu_out_time, cpu_in_size, cpu_out_size = self.get_stage1_transfer_time_detailed(n, m, cpu_bw)
            cpu_transfer = cpu_in_time + cpu_out_time
            cpu_total = cpu_compute + cpu_transfer

            subgraph.dp_split_ratio = 1.0
            subgraph.cpu_compute_time = cpu_compute
            subgraph.gpu_compute_time = 0.0
            subgraph.cpu_transfer_time = cpu_transfer
            subgraph.gpu_transfer_time = 0.0
            subgraph.cpu_total_time = cpu_total
            subgraph.gpu_total_time = 0.0
            # Store detailed transfer info
            subgraph.cpu_transfer_in_time = cpu_in_time
            subgraph.cpu_transfer_out_time = cpu_out_time
            subgraph.cpu_transfer_in_size = cpu_in_size
            subgraph.cpu_transfer_out_size = cpu_out_size
            subgraph.gpu_transfer_in_time = 0.0
            subgraph.gpu_transfer_out_time = 0.0
            subgraph.gpu_transfer_in_size = 0.0
            subgraph.gpu_transfer_out_size = 0.0

            stage1_time = cpu_total
            stage1_compute = cpu_compute
            stage1_transfer = cpu_transfer

        elif hasattr(self, 'stage1_interpolator') and self.stage1_interpolator:
            # Backward compatibility
            stage1_compute = self.stage1_interpolator.query(n, m)
            cpu_in_time, cpu_out_time, cpu_in_size, cpu_out_size = self.get_stage1_transfer_time_detailed(n, m, cpu_bw)
            stage1_transfer = cpu_in_time + cpu_out_time
            stage1_time = stage1_compute + stage1_transfer

            subgraph.dp_split_ratio = 1.0
            subgraph.cpu_compute_time = stage1_compute
            subgraph.gpu_compute_time = 0.0
            subgraph.cpu_transfer_time = stage1_transfer
            subgraph.gpu_transfer_time = 0.0
            subgraph.cpu_total_time = stage1_time
            subgraph.gpu_total_time = 0.0
            # Store detailed transfer info
            subgraph.cpu_transfer_in_time = cpu_in_time
            subgraph.cpu_transfer_out_time = cpu_out_time
            subgraph.cpu_transfer_in_size = cpu_in_size
            subgraph.cpu_transfer_out_size = cpu_out_size
            subgraph.gpu_transfer_in_time = 0.0
            subgraph.gpu_transfer_out_time = 0.0
            subgraph.gpu_transfer_in_size = 0.0
            subgraph.gpu_transfer_out_size = 0.0

        else:
            stage1_compute = self.get_stage_time(n, m, self.stage1_stages)
            cpu_in_time, cpu_out_time, cpu_in_size, cpu_out_size = self.get_stage1_transfer_time_detailed(n, m, cpu_bw)
            stage1_transfer = cpu_in_time + cpu_out_time
            stage1_time = stage1_compute + stage1_transfer

            subgraph.dp_split_ratio = 1.0
            subgraph.cpu_compute_time = stage1_compute
            subgraph.gpu_compute_time = 0.0
            subgraph.cpu_transfer_time = stage1_transfer
            subgraph.gpu_transfer_time = 0.0
            subgraph.cpu_total_time = stage1_time
            subgraph.gpu_total_time = 0.0
            # Store detailed transfer info
            subgraph.cpu_transfer_in_time = cpu_in_time
            subgraph.cpu_transfer_out_time = cpu_out_time
            subgraph.cpu_transfer_in_size = cpu_in_size
            subgraph.cpu_transfer_out_size = cpu_out_size
            subgraph.gpu_transfer_in_time = 0.0
            subgraph.gpu_transfer_out_time = 0.0
            subgraph.gpu_transfer_in_size = 0.0
            subgraph.gpu_transfer_out_size = 0.0

        # Stage 2: NPU only (edge-independent)
        if hasattr(self, 'npu_interpolator') and self.npu_interpolator:
            stage2_compute = self.npu_interpolator.query(n, m)
        elif hasattr(self, 'stage2_interpolator') and self.stage2_interpolator:
            stage2_compute = self.stage2_interpolator.query(n, m)
        else:
            stage2_compute = self.get_stage_time(n, m, self.stage2_stages)

        # NPU transfer is edge-independent: input (sum_agg + count + x) + output (activated)
        npu_in_time, npu_out_time, npu_in_size, npu_out_size = self.get_stage2_transfer_time_detailed(n, npu_bw)
        stage2_transfer = npu_in_time + npu_out_time
        stage2_time = stage2_compute + stage2_transfer

        # Store breakdown in subgraph
        subgraph.stage1_compute = stage1_compute
        subgraph.stage1_transfer = stage1_transfer
        subgraph.stage2_compute = stage2_compute
        subgraph.stage2_transfer = stage2_transfer
        # Store detailed NPU transfer info
        subgraph.npu_transfer_in_time = npu_in_time
        subgraph.npu_transfer_out_time = npu_out_time
        subgraph.npu_transfer_in_size = npu_in_size
        subgraph.npu_transfer_out_size = npu_out_size

        return stage1_time, stage2_time

    def johnsons_rule(self, subgraphs: List[SubgraphInfo]) -> List[int]:
        """
        Apply Johnson's Rule to find optimal ordering of subgraphs.

        Johnson's Rule for 2-machine scheduling:
        1. Split jobs into two groups:
           - Group A: Jobs where Stage1 time < Stage2 time
           - Group B: Jobs where Stage1 time >= Stage2 time
        2. Sort:
           - Group A: Sort by Stage1 time (ascending)
           - Group B: Sort by Stage2 time (descending)
        3. Concatenate: Final order = sorted Group A + sorted Group B

        Args:
            subgraphs: List of SubgraphInfo objects

        Returns:
            List of subgraph IDs in optimal order
        """
        # Calculate pipeline stage times for each subgraph
        for sg in subgraphs:
            sg.stage1_time, sg.stage2_time = self.get_pipeline_stage_times(sg)

        # Split into two groups
        group_a = []  # Stage1 time < Stage2 time
        group_b = []  # Stage1 time >= Stage2 time

        for sg in subgraphs:
            if sg.stage1_time < sg.stage2_time:
                group_a.append(sg)
            else:
                group_b.append(sg)

        # Sort Group A by Stage1 time (ascending) - short first stage first
        group_a.sort(key=lambda x: x.stage1_time)

        # Sort Group B by Stage2 time (descending) - long second stage last
        group_b.sort(key=lambda x: x.stage2_time, reverse=True)

        # Concatenate and return IDs
        optimal_order = [sg.id for sg in group_a] + [sg.id for sg in group_b]

        return optimal_order

    def calculate_makespan(
        self,
        order: List[int],
        subgraphs: List[SubgraphInfo]
    ) -> ScheduleResult:
        """
        Calculate makespan for a given order with synchronous pipeline.

        Synchronous pipeline model with N+1 cycles for N subgraphs:
        - Cycle 0: S[0].stage1 only (pipeline warmup)
        - Cycle i (1 <= i <= N-1): max(S[i].stage1, S[i-1].stage2) (sync point)
        - Cycle N: S[N-1].stage2 only (pipeline drain)

        Args:
            order: List of subgraph IDs in scheduling order
            subgraphs: List of SubgraphInfo objects

        Returns:
            ScheduleResult with makespan and detailed timing info
        """
        # Create lookup dict for subgraphs
        sg_dict = {sg.id: sg for sg in subgraphs}

        # Ensure stage times are calculated
        for sg in subgraphs:
            if sg.stage1_time == 0 and sg.stage2_time == 0:
                sg.stage1_time, sg.stage2_time = self.get_pipeline_stage_times(sg)

        n = len(order)
        if n == 0:
            return ScheduleResult(
                order=order,
                makespan=0.0,
                stage1_times=[],
                stage2_times=[],
                stage1_end_times=[],
                stage2_end_times=[],
                cycle_times=[],
                cycle_device_times=[]
            )

        # Collect all timing info per subgraph in order
        stage1_times = []
        stage2_times = []

        # Per-subgraph detailed timing
        sg_info = []  # List of dicts with all timing info
        for sg_id in order:
            sg = sg_dict[sg_id]
            stage1_times.append(sg.stage1_time)
            stage2_times.append(sg.stage2_time)
            sg_info.append({
                'id': sg_id,
                'dp_split_ratio': sg.dp_split_ratio,
                'cpu_compute': sg.cpu_compute_time,
                'gpu_compute': sg.gpu_compute_time,
                'cpu_transfer': sg.cpu_transfer_time,
                'gpu_transfer': sg.gpu_transfer_time,
                'cpu_transfer_in': sg.cpu_transfer_in_time,
                'cpu_transfer_out': sg.cpu_transfer_out_time,
                'cpu_transfer_in_size': sg.cpu_transfer_in_size,
                'cpu_transfer_out_size': sg.cpu_transfer_out_size,
                'gpu_transfer_in': sg.gpu_transfer_in_time,
                'gpu_transfer_out': sg.gpu_transfer_out_time,
                'gpu_transfer_in_size': sg.gpu_transfer_in_size,
                'gpu_transfer_out_size': sg.gpu_transfer_out_size,
                'cpu_total': sg.cpu_total_time,
                'gpu_total': sg.gpu_total_time,
                'npu_compute': sg.stage2_compute,
                'npu_transfer': sg.stage2_transfer,
                'npu_transfer_in': sg.npu_transfer_in_time,
                'npu_transfer_out': sg.npu_transfer_out_time,
                'npu_transfer_in_size': sg.npu_transfer_in_size,
                'npu_transfer_out_size': sg.npu_transfer_out_size,
                'npu_total': sg.stage2_time,
            })

        # Calculate cycle times and device times (N+1 cycles for N subgraphs)
        cycle_times = []
        cycle_device_times = []

        # Cycle 0: only S[0].stage1 (warmup)
        # CPU+GPU work on S[0], NPU idle
        s0 = sg_info[0]
        cycle_time_0 = stage1_times[0]
        cycle_times.append(cycle_time_0)
        cdt_0 = CycleDeviceTimes(
            cycle_time=cycle_time_0,
            cpu_compute=s0['cpu_compute'],
            gpu_compute=s0['gpu_compute'],
            npu_compute=0.0,
            cpu_transfer=s0['cpu_transfer'],
            gpu_transfer=s0['gpu_transfer'],
            npu_transfer=0.0,
            cpu_transfer_in=s0['cpu_transfer_in'],
            cpu_transfer_out=s0['cpu_transfer_out'],
            gpu_transfer_in=s0['gpu_transfer_in'],
            gpu_transfer_out=s0['gpu_transfer_out'],
            npu_transfer_in=0.0,
            npu_transfer_out=0.0,
            cpu_transfer_in_size=s0['cpu_transfer_in_size'],
            cpu_transfer_out_size=s0['cpu_transfer_out_size'],
            gpu_transfer_in_size=s0['gpu_transfer_in_size'],
            gpu_transfer_out_size=s0['gpu_transfer_out_size'],
            npu_transfer_in_size=0.0,
            npu_transfer_out_size=0.0,
            cpu_total=s0['cpu_total'],
            gpu_total=s0['gpu_total'],
            npu_total=0.0,
            dp_split_ratio=s0['dp_split_ratio'],
            stage1_subgraph_id=order[0],
            stage2_subgraph_id=-1,
        )
        cdt_0.compute_utilization()
        cycle_device_times.append(cdt_0)

        # Cycle 1 to N-1: max(S[i].stage1, S[i-1].stage2)
        # CPU+GPU work on S[i], NPU works on S[i-1]
        for i in range(1, n):
            si = sg_info[i]      # Stage 1 subgraph
            si_prev = sg_info[i-1]  # Stage 2 subgraph

            cycle_time_i = max(stage1_times[i], stage2_times[i-1])
            cycle_times.append(cycle_time_i)

            cdt_i = CycleDeviceTimes(
                cycle_time=cycle_time_i,
                cpu_compute=si['cpu_compute'],
                gpu_compute=si['gpu_compute'],
                npu_compute=si_prev['npu_compute'],
                cpu_transfer=si['cpu_transfer'],
                gpu_transfer=si['gpu_transfer'],
                npu_transfer=si_prev['npu_transfer'],
                cpu_transfer_in=si['cpu_transfer_in'],
                cpu_transfer_out=si['cpu_transfer_out'],
                gpu_transfer_in=si['gpu_transfer_in'],
                gpu_transfer_out=si['gpu_transfer_out'],
                npu_transfer_in=si_prev['npu_transfer_in'],
                npu_transfer_out=si_prev['npu_transfer_out'],
                cpu_transfer_in_size=si['cpu_transfer_in_size'],
                cpu_transfer_out_size=si['cpu_transfer_out_size'],
                gpu_transfer_in_size=si['gpu_transfer_in_size'],
                gpu_transfer_out_size=si['gpu_transfer_out_size'],
                npu_transfer_in_size=si_prev['npu_transfer_in_size'],
                npu_transfer_out_size=si_prev['npu_transfer_out_size'],
                cpu_total=si['cpu_total'],
                gpu_total=si['gpu_total'],
                npu_total=si_prev['npu_total'],
                dp_split_ratio=si['dp_split_ratio'],
                stage1_subgraph_id=order[i],
                stage2_subgraph_id=order[i-1],
            )
            cdt_i.compute_utilization()
            cycle_device_times.append(cdt_i)

        # Cycle N: only S[N-1].stage2 (drain)
        # CPU+GPU idle, NPU works on S[N-1]
        sn = sg_info[n-1]
        cycle_time_n = stage2_times[n-1]
        cycle_times.append(cycle_time_n)
        cdt_n = CycleDeviceTimes(
            cycle_time=cycle_time_n,
            cpu_compute=0.0,
            gpu_compute=0.0,
            npu_compute=sn['npu_compute'],
            cpu_transfer=0.0,
            gpu_transfer=0.0,
            npu_transfer=sn['npu_transfer'],
            cpu_transfer_in=0.0,
            cpu_transfer_out=0.0,
            gpu_transfer_in=0.0,
            gpu_transfer_out=0.0,
            npu_transfer_in=sn['npu_transfer_in'],
            npu_transfer_out=sn['npu_transfer_out'],
            cpu_transfer_in_size=0.0,
            cpu_transfer_out_size=0.0,
            gpu_transfer_in_size=0.0,
            gpu_transfer_out_size=0.0,
            npu_transfer_in_size=sn['npu_transfer_in_size'],
            npu_transfer_out_size=sn['npu_transfer_out_size'],
            cpu_total=0.0,
            gpu_total=0.0,
            npu_total=sn['npu_total'],
            dp_split_ratio=0.0,
            stage1_subgraph_id=-1,
            stage2_subgraph_id=order[n-1],
        )
        cdt_n.compute_utilization()
        cycle_device_times.append(cdt_n)

        # Calculate cumulative end times
        stage1_end = []
        stage2_end = []

        for i in range(n):
            # Stage 1 of subgraph i ends after cycle i completes
            cumulative_time_after_cycle_i = sum(cycle_times[:i+1])
            stage1_end.append(cumulative_time_after_cycle_i)

            # Stage 2 of subgraph i ends after cycle i+1 completes
            cumulative_time_after_cycle_i_plus_1 = sum(cycle_times[:i+2])
            stage2_end.append(cumulative_time_after_cycle_i_plus_1)

        makespan = sum(cycle_times)

        return ScheduleResult(
            order=order,
            makespan=makespan,
            stage1_times=stage1_times,
            stage2_times=stage2_times,
            stage1_end_times=stage1_end,
            stage2_end_times=stage2_end,
            cycle_times=cycle_times,
            cycle_device_times=cycle_device_times
        )

    def optimize(self, subgraphs: List[SubgraphInfo]) -> ScheduleResult:
        """
        Find optimal scheduling order and calculate makespan.

        Args:
            subgraphs: List of SubgraphInfo objects with total_nodes and internal_edges

        Returns:
            ScheduleResult with optimal order, makespan, and timing breakdown
        """
        # Apply Johnson's Rule to get optimal order
        optimal_order = self.johnsons_rule(subgraphs)

        # Calculate makespan for optimal order
        result = self.calculate_makespan(optimal_order, subgraphs)

        return result

    def optimize_from_tuples(
        self,
        subgraph_data: List[Tuple[int, int]]
    ) -> ScheduleResult:
        """
        Convenience method to optimize from (total_nodes, internal_edges) tuples.

        Args:
            subgraph_data: List of (total_nodes, internal_edges) tuples

        Returns:
            ScheduleResult with optimal order, makespan, and timing breakdown
        """
        subgraphs = [
            SubgraphInfo(id=i, total_nodes=n, internal_edges=m)
            for i, (n, m) in enumerate(subgraph_data)
        ]

        return self.optimize(subgraphs)

    def print_schedule(self, result: ScheduleResult, subgraphs: List[SubgraphInfo] = None):
        """Print a formatted schedule summary."""
        print("\n" + "="*80)
        print("Pipeline Schedule (Johnson's Rule) - Synchronous Pipeline")
        print("="*80)
        print(f"Stage 1 primitives: {self.stage1_stages}")
        print(f"Stage 2 primitives: {self.stage2_stages}")
        print(f"Device: {self.device}")
        print(f"Stage 1 Bandwidth: {self.stage1_bw_gbps:.1f} GB/s")
        print(f"Stage 2 Bandwidth: {self.stage2_bw_gbps:.1f} GB/s")
        print(f"Feature dimension: {self.feature_dim}")
        print("-"*80)
        print(f"Optimal Order: {result.order}")
        print(f"Number of Cycles: {len(result.cycle_times)} (N+1 for {len(result.order)} subgraphs)")
        print(f"Total Makespan: {result.makespan:.2f} ms")
        print("-"*80)

        # Print cycle breakdown with device utilization
        if result.cycle_times and result.cycle_device_times:
            print("\nCycle Breakdown with Device Utilization (Total = Compute + Transfer):")
            print(f"{'Cyc':<4} {'Time':<8} {'S1 SG':<6} {'α%':<6} │ {'CPU Tot':<9} {'GPU Tot':<9} {'NPU Tot':<9} │ {'CPU%':<6} {'GPU%':<6} {'NPU%':<6}")
            print("-"*105)
            n = len(result.order)
            for i, (ct, cdt) in enumerate(zip(result.cycle_times, result.cycle_device_times)):
                s1_sg = str(cdt.stage1_subgraph_id) if cdt.stage1_subgraph_id >= 0 else "-"
                alpha = f"{cdt.dp_split_ratio*100:.1f}" if cdt.dp_split_ratio > 0 else "-"
                print(f"{i:<4} {ct:<8.2f} {s1_sg:<6} {alpha:<6} │ "
                      f"{cdt.cpu_total:<9.2f} {cdt.gpu_total:<9.2f} {cdt.npu_total:<9.2f} │ "
                      f"{cdt.cpu_util*100:<6.1f} {cdt.gpu_util*100:<6.1f} {cdt.npu_util*100:<6.1f}")
            print("-"*105)

            # Calculate average utilization
            total_cpu = sum(cdt.cpu_total for cdt in result.cycle_device_times)
            total_gpu = sum(cdt.gpu_total for cdt in result.cycle_device_times)
            total_npu = sum(cdt.npu_total for cdt in result.cycle_device_times)
            avg_cpu_util = total_cpu / result.makespan if result.makespan > 0 else 0
            avg_gpu_util = total_gpu / result.makespan if result.makespan > 0 else 0
            avg_npu_util = total_npu / result.makespan if result.makespan > 0 else 0
            print(f"{'Avg':<4} {result.makespan:<8.2f} {'-':<6} {'-':<6} │ "
                  f"{total_cpu:<9.2f} {total_gpu:<9.2f} {total_npu:<9.2f} │ "
                  f"{avg_cpu_util*100:<6.1f} {avg_gpu_util*100:<6.1f} {avg_npu_util*100:<6.1f}")
            print("-"*105)

        print("\nDetailed Timeline:")
        print(f"{'Subgraph':<10} {'S1 Time':<12} {'S2 Time':<12} {'S1 End':<12} {'S2 End':<12}")
        print("-"*80)

        for i, sg_id in enumerate(result.order):
            print(f"{sg_id:<10} {result.stage1_times[i]:<12.2f} {result.stage2_times[i]:<12.2f} "
                  f"{result.stage1_end_times[i]:<12.2f} {result.stage2_end_times[i]:<12.2f}")

        # Print breakdown if subgraphs provided
        if subgraphs:
            sg_dict = {sg.id: sg for sg in subgraphs}

            # Check if DP info is available
            has_dp = any(sg.gpu_compute_time > 0 for sg in subgraphs)

            if has_dp:
                print("\n" + "-"*120)
                print("Stage 1 DP Breakdown (CPU + GPU Data Parallel, each with compute + transfer):")
                print(f"{'SG':<4} {'α%':<6} {'CPU Comp':<10} {'CPU Xfer':<10} {'CPU Tot':<10} │ "
                      f"{'GPU Comp':<10} {'GPU Xfer':<10} {'GPU Tot':<10} │ {'S1 Time':<10}")
                print("-"*120)
                for sg_id in result.order:
                    sg = sg_dict[sg_id]
                    print(f"{sg_id:<4} {sg.dp_split_ratio*100:<6.1f} "
                          f"{sg.cpu_compute_time:<10.2f} {sg.cpu_transfer_time:<10.2f} {sg.cpu_total_time:<10.2f} │ "
                          f"{sg.gpu_compute_time:<10.2f} {sg.gpu_transfer_time:<10.2f} {sg.gpu_total_time:<10.2f} │ "
                          f"{sg.stage1_time:<10.2f}")
            else:
                print("\n" + "-"*80)
                print("Time Breakdown (Compute + Transfer):")
                print(f"{'Subgraph':<10} {'S1 Comp':<10} {'S1 Xfer':<10} {'S2 Comp':<10} {'S2 Xfer':<10} {'Data MB':<10}")
                print("-"*80)
                for sg_id in result.order:
                    sg = sg_dict[sg_id]
                    data_mb = self.estimate_data_size_bytes(sg.total_nodes, sg.internal_edges) / 1e6
                    print(f"{sg_id:<10} {sg.stage1_compute:<10.2f} {sg.stage1_transfer:<10.2f} "
                          f"{sg.stage2_compute:<10.2f} {sg.stage2_transfer:<10.2f} {data_mb:<10.2f}")

        print("="*80)


def brute_force_optimal(scheduler: PipelineScheduler, subgraphs: List[SubgraphInfo]) -> ScheduleResult:
    """
    Find optimal order by brute force (for verification with small N).

    Args:
        scheduler: PipelineScheduler instance
        subgraphs: List of SubgraphInfo objects

    Returns:
        ScheduleResult with truly optimal order
    """
    from itertools import permutations

    # Ensure stage times are calculated
    for sg in subgraphs:
        sg.stage1_time, sg.stage2_time = scheduler.get_pipeline_stage_times(sg)

    best_result = None
    best_makespan = float('inf')

    ids = [sg.id for sg in subgraphs]

    for perm in permutations(ids):
        result = scheduler.calculate_makespan(list(perm), subgraphs)
        if result.makespan < best_makespan:
            best_makespan = result.makespan
            best_result = result

    return best_result


if __name__ == "__main__":
    # ========================================================================
    # Test with REAL data: Reddit2 K=10 Partition + CPU/GPU DP + NPU Pipeline
    # ========================================================================
    print("="*100)
    print("Pipeline Scheduler Test: Reddit2 K=10 with CPU+GPU Data Parallel + NPU")
    print("="*100)

    # Paths to profiling data
    cpu_file = "/home/haoyang/private/GNX_final/profiling_fused/results/186H/CPU_1234.md"
    gpu_file = "/home/haoyang/private/GNX_final/profiling_fused/results/186H/GPUI_1234.md"
    npu_file = "/home/haoyang/private/GNX_final/profiling_fused/results/186H/NPU_567.md"
    partition_file = "/home/haoyang/private/GNX_final/experiments/pipeline_time/data..md"

    # Create scheduler from markdown files with DP support
    try:
        scheduler_real = PipelineScheduler.from_markdown_files(
            cpu_profiling_file=cpu_file,
            gpu_profiling_file=gpu_file,
            npu_profiling_file=npu_file,
            cpu_bw_gbps=50.0,      # CPU memory bandwidth (DDR4)
            gpu_bw_gbps=16.0,      # GPU PCIe bandwidth (PCIe 3.0 x16)
            npu_bw_gbps=10.0,      # NPU bandwidth
            feature_dim=128,
            bytes_per_element=4,
            gpu_edge_limit=2_500_000,  # GPU memory limit
            dp_max_iterations=9,       # Binary search iterations (0.2% precision)
        )

        # Parse partition data
        partition_data = parse_partition_markdown(partition_file)
        print(f"\nLoaded {len(partition_data)} partitions from K=10 data:")

        subgraphs_real = []
        for id_, total_nodes, internal_edges in partition_data:
            sg = SubgraphInfo(id=id_, total_nodes=total_nodes, internal_edges=internal_edges)
            subgraphs_real.append(sg)
            data_mb = scheduler_real.estimate_data_size_bytes(total_nodes, internal_edges) / 1e6
            print(f"  Partition {id_}: {total_nodes:,} nodes, {internal_edges:,} edges, ~{data_mb:.1f} MB")

        # Optimize with Johnson's Rule
        result_real = scheduler_real.optimize(subgraphs_real)
        scheduler_real.print_schedule(result_real, subgraphs_real)

        # Compare with sequential (original order)
        print("\n" + "-"*80)
        print("Comparison: Johnson's Rule vs Sequential Order")
        print("-"*80)

        # Reset stage times
        for sg in subgraphs_real:
            sg.stage1_time = 0
            sg.stage2_time = 0

        sequential_order = [sg.id for sg in subgraphs_real]
        result_sequential = scheduler_real.calculate_makespan(sequential_order, subgraphs_real)

        print(f"Sequential Order: {sequential_order}")
        print(f"Sequential Makespan: {result_sequential.makespan:.2f} ms")
        print(f"Johnson's Rule Order: {result_real.order}")
        print(f"Johnson's Rule Makespan: {result_real.makespan:.2f} ms")
        improvement = (result_sequential.makespan - result_real.makespan) / result_sequential.makespan * 100
        print(f"Improvement: {result_sequential.makespan - result_real.makespan:.2f} ms ({improvement:.1f}%)")

        # Verify with brute force (only if small enough)
        if len(subgraphs_real) <= 6:
            print("\n" + "-"*80)
            print("Verification with Brute Force (may take time for N>6)")
            print("-"*80)

            # Reset stage times
            for sg in subgraphs_real:
                sg.stage1_time = 0
                sg.stage2_time = 0

            bf_result_real = brute_force_optimal(scheduler_real, subgraphs_real)
            print(f"Johnson's Rule: {result_real.makespan:.2f} ms")
            print(f"Brute Force:    {bf_result_real.makespan:.2f} ms")

            if abs(result_real.makespan - bf_result_real.makespan) < 0.01:
                print("Verification PASSED!")
            else:
                print(f"Difference: {abs(result_real.makespan - bf_result_real.makespan):.2f} ms")

    except FileNotFoundError as e:
        print(f"Could not load real data files: {e}")
    except Exception as e:
        print(f"Error during real data test: {e}")
        import traceback
        traceback.print_exc()
