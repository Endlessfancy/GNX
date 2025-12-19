"""
Cost Estimator - Phase 2
估算PEP的执行代价
"""

from typing import List, Dict
from .pep_generator import PEP, PEPBlock


class CostEstimator:
    """
    PEP代价估算器

    使用profiling lookup table估算PEP的执行时间
    """

    def __init__(self, profiling_loader, config):
        """
        Args:
            profiling_loader: ProfilingLoader对象
            config: CompilerConfig对象
        """
        self.profiling = profiling_loader
        self.config = config
        self.feature_dim = config.feature_dim

    def estimate_pep_cost(self, pep: PEP, subgraph) -> Dict:
        """
        估算一个PEP在给定subgraph上的执行代价

        Args:
            pep: PEP对象
            subgraph: Subgraph对象

        Returns:
            代价字典，包含total_time, block_times, transfer_times等
        """
        block_times = []
        transfer_times = []
        total_time = 0.0

        for block_idx, block in enumerate(pep.blocks):
            # 估算block的计算时间
            compute_time = self._estimate_block_time(block, subgraph)
            block_times.append(compute_time)

            # 估算数据传输时间（如果不是最后一个block）
            if block_idx < len(pep.blocks) - 1:
                transfer_time = self._estimate_transfer_time(
                    block,
                    pep.blocks[block_idx + 1],
                    subgraph
                )
                transfer_times.append(transfer_time)
            else:
                transfer_times.append(0.0)

            # 累加总时间（串行执行）
            total_time += compute_time + transfer_times[block_idx]

        return {
            'total_time': total_time,
            'block_times': block_times,
            'transfer_times': transfer_times,
            'breakdown': {
                'compute': sum(block_times),
                'transfer': sum(transfer_times)
            }
        }

    def _estimate_block_time(self, block: PEPBlock, subgraph) -> float:
        """
        估算一个block的执行时间（考虑DP并行）

        Args:
            block: PEPBlock对象
            subgraph: Subgraph对象

        Returns:
            执行时间（ms）
        """
        devices = block.devices
        ratios = block.ratios
        stages = block.stages

        device_times = []

        for dev_idx, device in enumerate(devices):
            ratio = ratios[dev_idx]

            # 计算该设备处理的数据量
            if device == 'NPU':
                # NPU使用padding后的大小
                n = int(subgraph.n_pad * ratio)
                m = int(subgraph.m_pad * ratio)
                # 加上padding overhead
                padding_overhead = self._estimate_padding_overhead(subgraph.n, subgraph.n_pad)
            else:
                n = int(subgraph.n * ratio)
                m = int(subgraph.m * ratio)
                padding_overhead = 0.0

            # 查询profiling数据
            compute_time = self.profiling.get_block_time(device, stages, n, m)

            # 加上padding overhead
            total_time = compute_time + padding_overhead

            device_times.append(total_time)

        # Block时间 = max(并行设备的时间)
        return max(device_times)

    def _estimate_transfer_time(self, src_block: PEPBlock, dst_block: PEPBlock, subgraph) -> float:
        """
        估算两个block之间的数据传输时间

        Args:
            src_block: 源block
            dst_block: 目标block
            subgraph: Subgraph对象

        Returns:
            传输时间（ms）
        """
        # 检查设备是否有重叠
        src_devices = set(src_block.devices)
        dst_devices = set(dst_block.devices)

        if src_devices & dst_devices:
            # 有重叠设备，无需传输
            return 0.0

        # 数据大小 = 节点数 * 特征维度 * 4字节（float32）
        data_size = subgraph.n * self.feature_dim * 4  # bytes

        # 查询带宽
        src_dev = src_block.devices[0]
        dst_dev = dst_block.devices[0]
        bandwidth = self.profiling.get_transfer_bandwidth(src_dev, dst_dev)

        # 传输时间（秒） = 数据大小 / 带宽
        transfer_time_sec = data_size / bandwidth

        # 转换为ms
        return transfer_time_sec * 1000

    def _estimate_padding_overhead(self, n_actual: int, n_pad: int) -> float:
        """
        估算NPU padding的overhead

        Args:
            n_actual: 实际节点数
            n_pad: padding后节点数

        Returns:
            Overhead时间（ms）
        """
        if n_pad <= n_actual:
            return 0.0

        # 简化：假设padding overhead = 0.5ms * (padding比例)
        padding_ratio = (n_pad - n_actual) / n_actual
        base_overhead = 0.5  # ms
        return base_overhead * padding_ratio
