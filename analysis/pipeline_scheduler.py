"""
Pipeline Scheduler using Johnson's Rule for 2-stage pipeline optimization.

This version uses compute-only profiling data (transfer times excluded from profiling).
Transfer times are calculated separately using measured bandwidth.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp2d


class Interpolator2D:
    """2D interpolator for profiling lookup table queries."""

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
                    self.z_grid[i, j] = self._nearest_neighbor(n, m)

        # Create interpolation function
        try:
            self.interp_func = interp2d(
                self.n_values, self.m_values, self.z_grid.T,
                kind='linear', bounds_error=False, fill_value=None
            )
        except Exception:
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
        """Query the interpolated value for (n, m)."""
        # Clamp to valid range
        n = max(self.n_values[0], min(self.n_values[-1], n))
        m = max(self.m_values[0], min(self.m_values[-1], m))

        if (n, m) in self.data_points:
            return self.data_points[(n, m)]

        if self.interp_func is not None:
            try:
                result = self.interp_func(n, m)[0]
                return max(0.0, float(result))
            except Exception:
                pass

        return self._nearest_neighbor(n, m)


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
    dp_split_ratio: float = 0.0
    cpu_compute_time: float = 0.0
    gpu_compute_time: float = 0.0
    cpu_transfer_time: float = 0.0
    gpu_transfer_time: float = 0.0
    cpu_total_time: float = 0.0
    gpu_total_time: float = 0.0
    # Detailed transfer breakdown
    cpu_transfer_in_time: float = 0.0
    cpu_transfer_out_time: float = 0.0
    gpu_transfer_in_time: float = 0.0
    gpu_transfer_out_time: float = 0.0
    npu_transfer_in_time: float = 0.0
    npu_transfer_out_time: float = 0.0
    cpu_transfer_in_size: float = 0.0
    cpu_transfer_out_size: float = 0.0
    gpu_transfer_in_size: float = 0.0
    gpu_transfer_out_size: float = 0.0
    npu_transfer_in_size: float = 0.0
    npu_transfer_out_size: float = 0.0

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
                    'transfer_in': {'time_ms': self.cpu_transfer_in_time, 'size_bytes': self.cpu_transfer_in_size},
                    'transfer_out': {'time_ms': self.cpu_transfer_out_time, 'size_bytes': self.cpu_transfer_out_size},
                    'total': self.cpu_total_time,
                },
                'gpu': {
                    'compute': self.gpu_compute_time,
                    'transfer': self.gpu_transfer_time,
                    'transfer_in': {'time_ms': self.gpu_transfer_in_time, 'size_bytes': self.gpu_transfer_in_size},
                    'transfer_out': {'time_ms': self.gpu_transfer_out_time, 'size_bytes': self.gpu_transfer_out_size},
                    'total': self.gpu_total_time,
                },
                'npu': {
                    'transfer_in': {'time_ms': self.npu_transfer_in_time, 'size_bytes': self.npu_transfer_in_size},
                    'transfer_out': {'time_ms': self.npu_transfer_out_time, 'size_bytes': self.npu_transfer_out_size},
                },
            },
        }


@dataclass
class CycleDeviceTimes:
    """Device times for a single cycle"""
    cycle_time: float = 0.0
    # Compute times
    cpu_compute: float = 0.0
    gpu_compute: float = 0.0
    npu_compute: float = 0.0
    # Transfer times
    cpu_transfer: float = 0.0
    gpu_transfer: float = 0.0
    npu_transfer: float = 0.0
    # Detailed transfer
    cpu_transfer_in: float = 0.0
    cpu_transfer_out: float = 0.0
    gpu_transfer_in: float = 0.0
    gpu_transfer_out: float = 0.0
    npu_transfer_in: float = 0.0
    npu_transfer_out: float = 0.0
    cpu_transfer_in_size: float = 0.0
    cpu_transfer_out_size: float = 0.0
    gpu_transfer_in_size: float = 0.0
    gpu_transfer_out_size: float = 0.0
    npu_transfer_in_size: float = 0.0
    npu_transfer_out_size: float = 0.0
    # Total times
    cpu_total: float = 0.0
    gpu_total: float = 0.0
    npu_total: float = 0.0
    # Utilization
    cpu_util: float = 0.0
    gpu_util: float = 0.0
    npu_util: float = 0.0
    # Subgraph IDs
    dp_split_ratio: float = 0.0
    stage1_subgraph_id: int = -1
    stage2_subgraph_id: int = -1

    def compute_utilization(self):
        """Calculate utilization percentages"""
        if self.cycle_time > 0:
            self.cpu_util = self.cpu_total / self.cycle_time
            self.gpu_util = self.gpu_total / self.cycle_time
            self.npu_util = self.npu_total / self.cycle_time

    def to_dict(self) -> dict:
        return {
            'cycle_time': self.cycle_time,
            'cpu': {
                'compute': self.cpu_compute,
                'transfer': self.cpu_transfer,
                'transfer_in': {'time_ms': self.cpu_transfer_in, 'size_bytes': self.cpu_transfer_in_size},
                'transfer_out': {'time_ms': self.cpu_transfer_out, 'size_bytes': self.cpu_transfer_out_size},
                'total': self.cpu_total,
                'utilization': self.cpu_util,
            },
            'gpu': {
                'compute': self.gpu_compute,
                'transfer': self.gpu_transfer,
                'transfer_in': {'time_ms': self.gpu_transfer_in, 'size_bytes': self.gpu_transfer_in_size},
                'transfer_out': {'time_ms': self.gpu_transfer_out, 'size_bytes': self.gpu_transfer_out_size},
                'total': self.gpu_total,
                'utilization': self.gpu_util,
            },
            'npu': {
                'compute': self.npu_compute,
                'transfer': self.npu_transfer,
                'transfer_in': {'time_ms': self.npu_transfer_in, 'size_bytes': self.npu_transfer_in_size},
                'transfer_out': {'time_ms': self.npu_transfer_out, 'size_bytes': self.npu_transfer_out_size},
                'total': self.npu_total,
                'utilization': self.npu_util,
            },
            'dp_split_ratio': self.dp_split_ratio,
            'stage1_subgraph_id': self.stage1_subgraph_id,
            'stage2_subgraph_id': self.stage2_subgraph_id,
        }


@dataclass
class ScheduleResult:
    """Result of pipeline scheduling optimization"""
    order: List[int]
    makespan: float
    stage1_times: List[float]
    stage2_times: List[float]
    stage1_end_times: List[float]
    stage2_end_times: List[float]
    cycle_times: List[float] = None
    cycle_device_times: List[CycleDeviceTimes] = None

    def to_dict(self) -> dict:
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
            'cycles': [cdt.to_dict() for cdt in self.cycle_device_times] if self.cycle_device_times else [],
        }

    def save_to_json(self, filepath: str, subgraphs: List['SubgraphInfo'] = None):
        data = self.to_dict()
        if subgraphs:
            sg_dict = {sg.id: sg for sg in subgraphs}
            data['subgraphs'] = [sg_dict[sg_id].to_dict() for sg_id in self.order]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filepath}")


class PipelineScheduler:
    """Johnson's Rule based 2-stage pipeline scheduler with compute-only profiling."""

    def __init__(
        self,
        cpu_data: Dict[Tuple[int, int], float],
        gpu_data: Dict[Tuple[int, int], float],
        npu_data: Dict[Tuple[int, int], float],
        bandwidth_config: dict,
        feature_dim: int = 500,
        bytes_per_element: int = 4,
        gpu_edge_limit: int = 2_500_000,
        dp_max_iterations: int = 9,
    ):
        """
        Initialize scheduler with compute-only profiling data.

        Args:
            cpu_data: {(n, m): compute_time_ms} for CPU Stage 1-4
            gpu_data: {(n, m): compute_time_ms} for GPU Stage 1-4
            npu_data: {(n, m): compute_time_ms} for NPU Stage 5-7
            bandwidth_config: Bandwidth configuration with input/output separation
            feature_dim: Node feature dimension
            bytes_per_element: Bytes per element (4 for float32)
            gpu_edge_limit: Maximum edges GPU can handle
            dp_max_iterations: Max iterations for binary search
        """
        self.feature_dim = feature_dim
        self.bytes_per_element = bytes_per_element
        self.gpu_edge_limit = gpu_edge_limit
        self.dp_max_iterations = dp_max_iterations

        # Build interpolators
        self.cpu_interpolator = Interpolator2D(cpu_data) if cpu_data else None
        self.gpu_interpolator = Interpolator2D(gpu_data) if gpu_data else None
        self.npu_interpolator = Interpolator2D(npu_data) if npu_data else None

        # Parse bandwidth config (GB/s)
        bw = bandwidth_config.get('bandwidth_corrected', bandwidth_config.get('bandwidth_original', {}))
        self.cpu_bw_in = bw.get('CPU', {}).get('input', 8.0)
        self.cpu_bw_out = bw.get('CPU', {}).get('output', 3.4)
        self.gpu_bw_in = bw.get('GPU', {}).get('input', 5.2)
        self.gpu_bw_out = bw.get('GPU', {}).get('output', 3.3)
        self.npu_bw_in = bw.get('NPU', {}).get('input', 0.7)
        self.npu_bw_out = bw.get('NPU', {}).get('output', 1.2)

    def get_stage1_input_size(self, n: int, m: int) -> float:
        """Stage 1-4 input: x [n, feat_dim] + edge_index [2, m]"""
        x_size = n * self.feature_dim * self.bytes_per_element
        edge_size = m * 2 * 8  # int64
        return x_size + edge_size

    def get_stage1_output_size(self, n: int) -> float:
        """Stage 1-4 output: sum_agg [n, feat_dim] + count [n]"""
        sum_agg_size = n * self.feature_dim * self.bytes_per_element
        count_size = n * self.bytes_per_element
        return sum_agg_size + count_size

    def get_stage2_input_size(self, n: int) -> float:
        """Stage 5-7 input: sum_agg [n, feat_dim] + count [n] + x [n, feat_dim]"""
        return n * self.feature_dim * self.bytes_per_element * 2 + n * self.bytes_per_element

    def get_stage2_output_size(self, n: int) -> float:
        """Stage 5-7 output: activated [n, feat_dim]"""
        return n * self.feature_dim * self.bytes_per_element

    def get_stage1_transfer_time_detailed(
        self, n: int, m: int, bw_in: float, bw_out: float
    ) -> Tuple[float, float, float, float]:
        """Calculate Stage 1 transfer times with separate in/out bandwidth."""
        in_size = self.get_stage1_input_size(n, m)
        out_size = self.get_stage1_output_size(n)
        in_time = in_size / (bw_in * 1e9) * 1000  # ms
        out_time = out_size / (bw_out * 1e9) * 1000
        return in_time, out_time, in_size, out_size

    def get_stage2_transfer_time_detailed(
        self, n: int, bw_in: float, bw_out: float
    ) -> Tuple[float, float, float, float]:
        """Calculate Stage 2 transfer times (edge-independent)."""
        in_size = self.get_stage2_input_size(n)
        out_size = self.get_stage2_output_size(n)
        in_time = in_size / (bw_in * 1e9) * 1000
        out_time = out_size / (bw_out * 1e9) * 1000
        return in_time, out_time, in_size, out_size

    def find_optimal_dp_split(self, n: int, m: int) -> dict:
        """
        Binary search for optimal CPU/GPU split ratio.

        Optimizes: min(max(cpu_compute + cpu_transfer, gpu_compute + gpu_transfer))
        """
        if self.gpu_interpolator is None:
            # No GPU, use CPU only
            cpu_compute = self.cpu_interpolator.query(n, m) if self.cpu_interpolator else 0
            cpu_in, cpu_out, cpu_in_size, cpu_out_size = self.get_stage1_transfer_time_detailed(
                n, m, self.cpu_bw_in, self.cpu_bw_out)
            cpu_transfer = cpu_in + cpu_out
            return {
                'alpha': 1.0,
                'cpu': {
                    'compute': cpu_compute, 'transfer': cpu_transfer,
                    'transfer_in': cpu_in, 'transfer_out': cpu_out,
                    'transfer_in_size': cpu_in_size, 'transfer_out_size': cpu_out_size,
                    'total': cpu_compute + cpu_transfer
                },
                'gpu': {
                    'compute': 0.0, 'transfer': 0.0,
                    'transfer_in': 0.0, 'transfer_out': 0.0,
                    'transfer_in_size': 0.0, 'transfer_out_size': 0.0,
                    'total': 0.0
                },
            }

        # Calculate alpha bounds
        if m > self.gpu_edge_limit:
            alpha_min = max(0.01, 1 - self.gpu_edge_limit / m)
        else:
            alpha_min = 0.01
        alpha_max = 0.99

        # Binary search
        for _ in range(self.dp_max_iterations):
            alpha = (alpha_min + alpha_max) / 2

            cpu_n, cpu_m = alpha * n, alpha * m
            gpu_n, gpu_m = (1 - alpha) * n, (1 - alpha) * m

            cpu_compute = self.cpu_interpolator.query(cpu_n, cpu_m)
            gpu_compute = self.gpu_interpolator.query(gpu_n, gpu_m)

            cpu_in, cpu_out, _, _ = self.get_stage1_transfer_time_detailed(
                cpu_n, cpu_m, self.cpu_bw_in, self.cpu_bw_out)
            gpu_in, gpu_out, _, _ = self.get_stage1_transfer_time_detailed(
                gpu_n, gpu_m, self.gpu_bw_in, self.gpu_bw_out)

            cpu_total = cpu_compute + cpu_in + cpu_out
            gpu_total = gpu_compute + gpu_in + gpu_out

            if cpu_total > gpu_total:
                alpha_max = alpha
            else:
                alpha_min = alpha

        # Final result
        alpha = (alpha_min + alpha_max) / 2
        cpu_n, cpu_m = alpha * n, alpha * m
        gpu_n, gpu_m = (1 - alpha) * n, (1 - alpha) * m

        cpu_compute = self.cpu_interpolator.query(cpu_n, cpu_m)
        gpu_compute = self.gpu_interpolator.query(gpu_n, gpu_m)

        cpu_in, cpu_out, cpu_in_size, cpu_out_size = self.get_stage1_transfer_time_detailed(
            cpu_n, cpu_m, self.cpu_bw_in, self.cpu_bw_out)
        gpu_in, gpu_out, gpu_in_size, gpu_out_size = self.get_stage1_transfer_time_detailed(
            gpu_n, gpu_m, self.gpu_bw_in, self.gpu_bw_out)

        return {
            'alpha': alpha,
            'cpu': {
                'compute': cpu_compute, 'transfer': cpu_in + cpu_out,
                'transfer_in': cpu_in, 'transfer_out': cpu_out,
                'transfer_in_size': cpu_in_size, 'transfer_out_size': cpu_out_size,
                'total': cpu_compute + cpu_in + cpu_out
            },
            'gpu': {
                'compute': gpu_compute, 'transfer': gpu_in + gpu_out,
                'transfer_in': gpu_in, 'transfer_out': gpu_out,
                'transfer_in_size': gpu_in_size, 'transfer_out_size': gpu_out_size,
                'total': gpu_compute + gpu_in + gpu_out
            },
        }

    def get_pipeline_stage_times(self, subgraph: SubgraphInfo) -> Tuple[float, float]:
        """Calculate stage 1 and stage 2 total times."""
        n, m = subgraph.total_nodes, subgraph.internal_edges

        # Stage 1: CPU + GPU DP
        dp_result = self.find_optimal_dp_split(n, m)
        cpu_info = dp_result['cpu']
        gpu_info = dp_result['gpu']

        subgraph.dp_split_ratio = dp_result['alpha']
        subgraph.cpu_compute_time = cpu_info['compute']
        subgraph.gpu_compute_time = gpu_info['compute']
        subgraph.cpu_transfer_time = cpu_info['transfer']
        subgraph.gpu_transfer_time = gpu_info['transfer']
        subgraph.cpu_total_time = cpu_info['total']
        subgraph.gpu_total_time = gpu_info['total']
        subgraph.cpu_transfer_in_time = cpu_info['transfer_in']
        subgraph.cpu_transfer_out_time = cpu_info['transfer_out']
        subgraph.cpu_transfer_in_size = cpu_info['transfer_in_size']
        subgraph.cpu_transfer_out_size = cpu_info['transfer_out_size']
        subgraph.gpu_transfer_in_time = gpu_info['transfer_in']
        subgraph.gpu_transfer_out_time = gpu_info['transfer_out']
        subgraph.gpu_transfer_in_size = gpu_info['transfer_in_size']
        subgraph.gpu_transfer_out_size = gpu_info['transfer_out_size']

        stage1_time = max(cpu_info['total'], gpu_info['total'])
        stage1_compute = max(cpu_info['compute'], gpu_info['compute'])
        stage1_transfer = max(cpu_info['transfer'], gpu_info['transfer'])

        # Stage 2: NPU only
        stage2_compute = self.npu_interpolator.query(n, m) if self.npu_interpolator else 0
        npu_in, npu_out, npu_in_size, npu_out_size = self.get_stage2_transfer_time_detailed(
            n, self.npu_bw_in, self.npu_bw_out)
        stage2_transfer = npu_in + npu_out
        stage2_time = stage2_compute + stage2_transfer

        subgraph.stage1_compute = stage1_compute
        subgraph.stage1_transfer = stage1_transfer
        subgraph.stage2_compute = stage2_compute
        subgraph.stage2_transfer = stage2_transfer
        subgraph.npu_transfer_in_time = npu_in
        subgraph.npu_transfer_out_time = npu_out
        subgraph.npu_transfer_in_size = npu_in_size
        subgraph.npu_transfer_out_size = npu_out_size

        return stage1_time, stage2_time

    def johnsons_rule(self, subgraphs: List[SubgraphInfo]) -> List[int]:
        """Apply Johnson's Rule to find optimal ordering."""
        for sg in subgraphs:
            sg.stage1_time, sg.stage2_time = self.get_pipeline_stage_times(sg)

        group_a = [sg for sg in subgraphs if sg.stage1_time < sg.stage2_time]
        group_b = [sg for sg in subgraphs if sg.stage1_time >= sg.stage2_time]

        group_a.sort(key=lambda x: x.stage1_time)
        group_b.sort(key=lambda x: x.stage2_time, reverse=True)

        return [sg.id for sg in group_a] + [sg.id for sg in group_b]

    def calculate_makespan(self, order: List[int], subgraphs: List[SubgraphInfo]) -> ScheduleResult:
        """Calculate makespan with synchronous pipeline (N+1 cycles for N subgraphs)."""
        sg_dict = {sg.id: sg for sg in subgraphs}

        for sg in subgraphs:
            if sg.stage1_time == 0 and sg.stage2_time == 0:
                sg.stage1_time, sg.stage2_time = self.get_pipeline_stage_times(sg)

        n = len(order)
        if n == 0:
            return ScheduleResult([], 0.0, [], [], [], [], [], [])

        stage1_times = [sg_dict[sg_id].stage1_time for sg_id in order]
        stage2_times = [sg_dict[sg_id].stage2_time for sg_id in order]

        sg_info = []
        for sg_id in order:
            sg = sg_dict[sg_id]
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

        cycle_times = []
        cycle_device_times = []

        # Cycle 0: warmup
        s0 = sg_info[0]
        cycle_time_0 = stage1_times[0]
        cycle_times.append(cycle_time_0)
        cdt_0 = CycleDeviceTimes(
            cycle_time=cycle_time_0,
            cpu_compute=s0['cpu_compute'], gpu_compute=s0['gpu_compute'], npu_compute=0.0,
            cpu_transfer=s0['cpu_transfer'], gpu_transfer=s0['gpu_transfer'], npu_transfer=0.0,
            cpu_transfer_in=s0['cpu_transfer_in'], cpu_transfer_out=s0['cpu_transfer_out'],
            gpu_transfer_in=s0['gpu_transfer_in'], gpu_transfer_out=s0['gpu_transfer_out'],
            npu_transfer_in=0.0, npu_transfer_out=0.0,
            cpu_transfer_in_size=s0['cpu_transfer_in_size'], cpu_transfer_out_size=s0['cpu_transfer_out_size'],
            gpu_transfer_in_size=s0['gpu_transfer_in_size'], gpu_transfer_out_size=s0['gpu_transfer_out_size'],
            npu_transfer_in_size=0.0, npu_transfer_out_size=0.0,
            cpu_total=s0['cpu_total'], gpu_total=s0['gpu_total'], npu_total=0.0,
            dp_split_ratio=s0['dp_split_ratio'],
            stage1_subgraph_id=order[0], stage2_subgraph_id=-1,
        )
        cdt_0.compute_utilization()
        cycle_device_times.append(cdt_0)

        # Cycle 1 to N-1: parallel execution
        for i in range(1, n):
            si = sg_info[i]
            si_prev = sg_info[i-1]
            cycle_time_i = max(stage1_times[i], stage2_times[i-1])
            cycle_times.append(cycle_time_i)

            cdt_i = CycleDeviceTimes(
                cycle_time=cycle_time_i,
                cpu_compute=si['cpu_compute'], gpu_compute=si['gpu_compute'], npu_compute=si_prev['npu_compute'],
                cpu_transfer=si['cpu_transfer'], gpu_transfer=si['gpu_transfer'], npu_transfer=si_prev['npu_transfer'],
                cpu_transfer_in=si['cpu_transfer_in'], cpu_transfer_out=si['cpu_transfer_out'],
                gpu_transfer_in=si['gpu_transfer_in'], gpu_transfer_out=si['gpu_transfer_out'],
                npu_transfer_in=si_prev['npu_transfer_in'], npu_transfer_out=si_prev['npu_transfer_out'],
                cpu_transfer_in_size=si['cpu_transfer_in_size'], cpu_transfer_out_size=si['cpu_transfer_out_size'],
                gpu_transfer_in_size=si['gpu_transfer_in_size'], gpu_transfer_out_size=si['gpu_transfer_out_size'],
                npu_transfer_in_size=si_prev['npu_transfer_in_size'], npu_transfer_out_size=si_prev['npu_transfer_out_size'],
                cpu_total=si['cpu_total'], gpu_total=si['gpu_total'], npu_total=si_prev['npu_total'],
                dp_split_ratio=si['dp_split_ratio'],
                stage1_subgraph_id=order[i], stage2_subgraph_id=order[i-1],
            )
            cdt_i.compute_utilization()
            cycle_device_times.append(cdt_i)

        # Cycle N: drain
        sn = sg_info[n-1]
        cycle_time_n = stage2_times[n-1]
        cycle_times.append(cycle_time_n)
        cdt_n = CycleDeviceTimes(
            cycle_time=cycle_time_n,
            cpu_compute=0.0, gpu_compute=0.0, npu_compute=sn['npu_compute'],
            cpu_transfer=0.0, gpu_transfer=0.0, npu_transfer=sn['npu_transfer'],
            cpu_transfer_in=0.0, cpu_transfer_out=0.0,
            gpu_transfer_in=0.0, gpu_transfer_out=0.0,
            npu_transfer_in=sn['npu_transfer_in'], npu_transfer_out=sn['npu_transfer_out'],
            cpu_transfer_in_size=0.0, cpu_transfer_out_size=0.0,
            gpu_transfer_in_size=0.0, gpu_transfer_out_size=0.0,
            npu_transfer_in_size=sn['npu_transfer_in_size'], npu_transfer_out_size=sn['npu_transfer_out_size'],
            cpu_total=0.0, gpu_total=0.0, npu_total=sn['npu_total'],
            dp_split_ratio=0.0,
            stage1_subgraph_id=-1, stage2_subgraph_id=order[n-1],
        )
        cdt_n.compute_utilization()
        cycle_device_times.append(cdt_n)

        stage1_end = [sum(cycle_times[:i+1]) for i in range(n)]
        stage2_end = [sum(cycle_times[:i+2]) for i in range(n)]
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
        """Find optimal scheduling order and calculate makespan."""
        optimal_order = self.johnsons_rule(subgraphs)
        return self.calculate_makespan(optimal_order, subgraphs)
