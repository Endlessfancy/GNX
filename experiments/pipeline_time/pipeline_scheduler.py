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


@dataclass
class ScheduleResult:
    """Result of pipeline scheduling"""
    order: List[int]  # Subgraph IDs in optimal order
    makespan: float  # Total time in ms
    stage1_times: List[float]  # Stage 1 time for each subgraph (in order)
    stage2_times: List[float]  # Stage 2 time for each subgraph (in order)
    stage1_end_times: List[float]  # When each subgraph finishes stage 1
    stage2_end_times: List[float]  # When each subgraph finishes stage 2


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
        stage1_profiling_file: str,
        stage2_profiling_file: str,
        stage1_bw_gbps: float = 10.0,
        stage2_bw_gbps: float = 10.0,
        feature_dim: int = 128,
        bytes_per_element: int = 4,
    ) -> 'PipelineScheduler':
        """
        Create a PipelineScheduler from markdown profiling files.

        Args:
            stage1_profiling_file: Path to markdown file with Stage 1 profiling data
            stage2_profiling_file: Path to markdown file with Stage 2 profiling data
            stage1_bw_gbps: Memory bandwidth for stage1 PU in GB/s
            stage2_bw_gbps: Memory bandwidth for stage2 PU in GB/s
            feature_dim: Node feature dimension
            bytes_per_element: Bytes per data element

        Returns:
            PipelineScheduler instance
        """
        # Create instance without calling normal __init__
        instance = cls.__new__(cls)

        # Set attributes
        instance.stage1_stages = ['fused_1234']  # Placeholder
        instance.stage2_stages = ['fused_567']   # Placeholder
        instance.device = 'mixed'

        instance.stage1_bw_gbps = stage1_bw_gbps
        instance.stage2_bw_gbps = stage2_bw_gbps
        instance.feature_dim = feature_dim
        instance.bytes_per_element = bytes_per_element

        instance.lookup_table = None

        # Parse profiling data from markdown files
        stage1_data = parse_profiling_markdown(stage1_profiling_file)
        stage2_data = parse_profiling_markdown(stage2_profiling_file)

        print(f"Loaded {len(stage1_data)} data points for Stage 1 (CPU)")
        print(f"Loaded {len(stage2_data)} data points for Stage 2 (NPU)")

        # Build interpolators directly
        instance.interpolators = {}
        instance.stage1_interpolator = Interpolator2D(stage1_data) if stage1_data else None
        instance.stage2_interpolator = Interpolator2D(stage2_data) if stage2_data else None

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
        Estimate data size in bytes for a subgraph.

        Data includes:
        - Node features: n * feature_dim * bytes_per_element
        - Edge indices (COO): m * 2 * 4 (int32 for src, dst)
        - Edge data/intermediate: m * bytes_per_element

        Args:
            n: Number of nodes
            m: Number of edges

        Returns:
            Estimated data size in bytes
        """
        node_data = n * self.feature_dim * self.bytes_per_element
        edge_indices = m * 2 * 4  # COO format: (src, dst) as int32
        edge_data = m * self.bytes_per_element  # Edge features or intermediate data

        return node_data + edge_indices + edge_data

    def get_transfer_time_ms(self, n: int, m: int, bw_gbps: float) -> float:
        """
        Calculate round-trip memory transfer time in milliseconds.

        Transfer time = 2 * data_size / bandwidth (round trip: to PU and back)

        Args:
            n: Number of nodes
            m: Number of edges
            bw_gbps: Bandwidth in GB/s

        Returns:
            Round-trip transfer time in ms
        """
        data_size_bytes = self.estimate_data_size_bytes(n, m)
        # Convert: bytes / (GB/s * 1e9 bytes/GB) * 1000 ms/s = ms
        # Round trip: multiply by 2
        transfer_time_ms = 2 * data_size_bytes / (bw_gbps * 1e9) * 1000
        return transfer_time_ms

    def get_pipeline_stage_times(self, subgraph: SubgraphInfo) -> Tuple[float, float]:
        """
        Get pipeline stage 1 and stage 2 total times for a subgraph.

        Total time = compute_time + round_trip_transfer_time

        Args:
            subgraph: SubgraphInfo with total_nodes and internal_edges

        Returns:
            Tuple of (stage1_total_time, stage2_total_time) in ms
        """
        n, m = subgraph.total_nodes, subgraph.internal_edges

        # Compute times - use direct interpolators if available (from markdown)
        if hasattr(self, 'stage1_interpolator') and self.stage1_interpolator:
            stage1_compute = self.stage1_interpolator.query(n, m)
        else:
            stage1_compute = self.get_stage_time(n, m, self.stage1_stages)

        if hasattr(self, 'stage2_interpolator') and self.stage2_interpolator:
            stage2_compute = self.stage2_interpolator.query(n, m)
        else:
            stage2_compute = self.get_stage_time(n, m, self.stage2_stages)

        # Transfer times (round trip)
        stage1_transfer = self.get_transfer_time_ms(n, m, self.stage1_bw_gbps)
        stage2_transfer = self.get_transfer_time_ms(n, m, self.stage2_bw_gbps)

        # Store breakdown in subgraph
        subgraph.stage1_compute = stage1_compute
        subgraph.stage1_transfer = stage1_transfer
        subgraph.stage2_compute = stage2_compute
        subgraph.stage2_transfer = stage2_transfer

        # Total time = compute + transfer
        stage1_time = stage1_compute + stage1_transfer
        stage2_time = stage2_compute + stage2_transfer

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
        Calculate makespan for a given order with pipeline parallelism.

        The makespan calculation accounts for pipeline parallelism:
        - Stage 2 of job i can start only after:
          1. Stage 1 of job i completes
          2. Stage 2 of job i-1 completes

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
        stage1_times = []
        stage2_times = []
        stage1_end = []
        stage2_end = []

        for i, sg_id in enumerate(order):
            sg = sg_dict[sg_id]
            t_s1 = sg.stage1_time
            t_s2 = sg.stage2_time

            stage1_times.append(t_s1)
            stage2_times.append(t_s2)

            if i == 0:
                # First subgraph
                s1_end = t_s1
                s2_end = s1_end + t_s2
            else:
                # Stage 1 finishes after previous stage 1 finishes
                s1_end = stage1_end[i-1] + t_s1
                # Stage 2 can only start after both:
                # - Current stage 1 finishes
                # - Previous stage 2 finishes
                s2_start = max(s1_end, stage2_end[i-1])
                s2_end = s2_start + t_s2

            stage1_end.append(s1_end)
            stage2_end.append(s2_end)

        makespan = stage2_end[-1] if stage2_end else 0.0

        return ScheduleResult(
            order=order,
            makespan=makespan,
            stage1_times=stage1_times,
            stage2_times=stage2_times,
            stage1_end_times=stage1_end,
            stage2_end_times=stage2_end
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
        print("Pipeline Schedule (Johnson's Rule)")
        print("="*80)
        print(f"Stage 1 primitives: {self.stage1_stages}")
        print(f"Stage 2 primitives: {self.stage2_stages}")
        print(f"Device: {self.device}")
        print(f"Stage 1 Bandwidth: {self.stage1_bw_gbps:.1f} GB/s")
        print(f"Stage 2 Bandwidth: {self.stage2_bw_gbps:.1f} GB/s")
        print(f"Feature dimension: {self.feature_dim}")
        print("-"*80)
        print(f"Optimal Order: {result.order}")
        print(f"Total Makespan: {result.makespan:.2f} ms")
        print("-"*80)
        print("\nDetailed Timeline:")
        print(f"{'Subgraph':<10} {'S1 Time':<12} {'S2 Time':<12} {'S1 End':<12} {'S2 End':<12}")
        print("-"*80)

        for i, sg_id in enumerate(result.order):
            print(f"{sg_id:<10} {result.stage1_times[i]:<12.2f} {result.stage2_times[i]:<12.2f} "
                  f"{result.stage1_end_times[i]:<12.2f} {result.stage2_end_times[i]:<12.2f}")

        # Print breakdown if subgraphs provided
        if subgraphs:
            sg_dict = {sg.id: sg for sg in subgraphs}
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
    # Example usage and verification
    profiling_dir = "/home/haoyang/private/GNX_final/profiling/results"

    # Create scheduler with memory transfer parameters
    # Example: Stage 1 on CPU (high bandwidth), Stage 2 on NPU (lower bandwidth)
    scheduler = PipelineScheduler(
        profiling_dir=profiling_dir,
        stage1_stages=[1, 2, 3, 4],
        stage2_stages=[5, 6, 7],
        device='CPU',
        stage1_bw_gbps=50.0,   # CPU memory bandwidth ~50 GB/s
        stage2_bw_gbps=10.0,   # PCIe to NPU ~10 GB/s
        feature_dim=128,       # 128-dim node features
        bytes_per_element=4,   # float32
    )

    # Test with example subgraphs (total_nodes, internal_edges)
    test_subgraph_data = [
        (5000, 15000),   # Subgraph 0
        (10000, 30000),  # Subgraph 1
        (3000, 8000),    # Subgraph 2
        (8000, 25000),   # Subgraph 3
    ]

    print("Test Subgraphs (nodes, edges):")
    for i, (n, m) in enumerate(test_subgraph_data):
        data_mb = scheduler.estimate_data_size_bytes(n, m) / 1e6
        print(f"  Subgraph {i}: {n} nodes, {m} edges, ~{data_mb:.2f} MB data")

    # Create SubgraphInfo objects
    subgraphs = [
        SubgraphInfo(id=i, total_nodes=n, internal_edges=m)
        for i, (n, m) in enumerate(test_subgraph_data)
    ]

    # Optimize using Johnson's Rule
    result = scheduler.optimize(subgraphs)
    scheduler.print_schedule(result, subgraphs)

    # Verify with brute force for small examples
    print("\n" + "="*80)
    print("Verification: Brute Force vs Johnson's Rule")
    print("="*80)

    # Reset stage times for fresh calculation
    for sg in subgraphs:
        sg.stage1_time = 0
        sg.stage2_time = 0

    bf_result = brute_force_optimal(scheduler, subgraphs)

    print(f"Johnson's Rule Order: {result.order}, Makespan: {result.makespan:.2f} ms")
    print(f"Brute Force Order:    {bf_result.order}, Makespan: {bf_result.makespan:.2f} ms")

    if abs(result.makespan - bf_result.makespan) < 0.01:
        print("\nVerification PASSED: Johnson's Rule achieves optimal makespan!")
    else:
        print(f"\nVerification FAILED: Difference = {abs(result.makespan - bf_result.makespan):.2f} ms")
        print("Note: Johnson's Rule is optimal for 2-machine problems, so this should not happen.")

    # Compare with compute-only (no transfer time)
    print("\n" + "="*80)
    print("Comparison: With vs Without Memory Transfer Time")
    print("="*80)

    scheduler_no_transfer = PipelineScheduler(
        profiling_dir=profiling_dir,
        stage1_stages=[1, 2, 3, 4],
        stage2_stages=[5, 6, 7],
        device='CPU',
        stage1_bw_gbps=float('inf'),  # Infinite bandwidth = no transfer time
        stage2_bw_gbps=float('inf'),
        feature_dim=128,
        bytes_per_element=4,
    )

    subgraphs_no_xfer = [
        SubgraphInfo(id=i, total_nodes=n, internal_edges=m)
        for i, (n, m) in enumerate(test_subgraph_data)
    ]
    result_no_xfer = scheduler_no_transfer.optimize(subgraphs_no_xfer)

    print(f"With transfer:    Makespan = {result.makespan:.2f} ms, Order = {result.order}")
    print(f"Without transfer: Makespan = {result_no_xfer.makespan:.2f} ms, Order = {result_no_xfer.order}")
    print(f"Transfer overhead: {result.makespan - result_no_xfer.makespan:.2f} ms "
          f"({(result.makespan - result_no_xfer.makespan) / result_no_xfer.makespan * 100:.1f}%)")

    # ========================================================================
    # Test with REAL data from profiling_fused and K=10 partition
    # ========================================================================
    print("\n\n" + "="*80)
    print("REAL DATA TEST: K=10 Partition with Fused Profiling Data")
    print("="*80)

    # Paths to profiling data
    stage1_file = "/home/haoyang/private/GNX_final/profiling_fused/results/CPU_1234.md"
    stage2_file = "/home/haoyang/private/GNX_final/profiling_fused/results/NPU_567.md"
    partition_file = "/home/haoyang/private/GNX_final/experiments/pipeline_time/data..md"

    # Create scheduler from markdown files
    try:
        scheduler_real = PipelineScheduler.from_markdown_files(
            stage1_profiling_file=stage1_file,
            stage2_profiling_file=stage2_file,
            stage1_bw_gbps=50.0,   # CPU memory bandwidth
            stage2_bw_gbps=10.0,   # PCIe to NPU
            feature_dim=128,
            bytes_per_element=4,
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
