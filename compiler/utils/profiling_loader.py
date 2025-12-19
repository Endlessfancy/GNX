"""
Profiling Data Loader
加载并管理profiling lookup tables
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from utils.interpolator import Interpolator2D


class ProfilingLoader:
    """
    加载profiling结果并提供查询接口
    """

    def __init__(self, profiling_dir: Path):
        """
        初始化ProfilingLoader

        Args:
            profiling_dir: profiling结果目录
        """
        self.profiling_dir = Path(profiling_dir)

        # 加载lookup table
        lookup_table_path = self.profiling_dir / 'lookup_table.json'
        with open(lookup_table_path, 'r') as f:
            self.lookup_table = json.load(f)

        # 加载bandwidth table
        bandwidth_table_path = self.profiling_dir / 'bandwidth_table.json'
        with open(bandwidth_table_path, 'r') as f:
            self.bandwidth_table = json.load(f)

        # 为每个(device, stage)组合创建插值器
        self.interpolators = self._build_interpolators()

    def _build_interpolators(self) -> Dict[Tuple[str, int], Interpolator2D]:
        """为每个(device, stage)组合构建插值器"""
        interpolators = {}

        # 从lookup_table中提取数据
        for key, value in self.lookup_table.items():
            # key格式: "n,m,device,stage"
            parts = key.split(',')
            if len(parts) != 4:
                continue

            n, m, device, stage = int(parts[0]), int(parts[1]), parts[2], int(parts[3])

            interp_key = (device, stage)
            if interp_key not in interpolators:
                interpolators[interp_key] = {}

            # 使用total_time_ms
            interpolators[interp_key][(n, m)] = value['total_time_ms']

        # 创建Interpolator2D对象
        return {
            key: Interpolator2D(data_points)
            for key, data_points in interpolators.items()
        }

    def get_stage_time(self, device: str, stage: int, n: int, m: int) -> float:
        """
        查询单个stage的执行时间（带插值）

        Args:
            device: 设备名 ('CPU', 'GPU', 'NPU')
            stage: stage编号 (1-7)
            n: 节点数
            m: 边数

        Returns:
            执行时间（ms）
        """
        key = (device, stage)
        if key not in self.interpolators:
            # 如果没有该组合（例如NPU的stage 3/4），返回无穷大
            return float('inf')

        return self.interpolators[key].query(n, m)

    def get_block_time(self, device: str, stages: List[int], n: int, m: int) -> float:
        """
        查询一个block（多个stage）的执行时间

        Args:
            device: 设备名
            stages: stage列表，例如[1, 2, 3]
            n: 节点数
            m: 边数

        Returns:
            Block执行时间（ms）= sum(各stage时间)
        """
        total_time = 0.0
        for stage in stages:
            total_time += self.get_stage_time(device, stage, n, m)

        return total_time

    def get_bandwidth(self, device: str, stage: Optional[int] = None) -> float:
        """
        查询设备的带宽（bytes/sec）

        Args:
            device: 设备名
            stage: stage编号（可选，如果提供则查询该stage的带宽）

        Returns:
            带宽（bytes/sec）
        """
        if stage is not None:
            key = f"{device}_stage{stage}"
        else:
            # 返回该设备所有stage的平均带宽
            device_keys = [k for k in self.bandwidth_table.keys() if k.startswith(device)]
            if not device_keys:
                return 1e9  # 默认1GB/s

            bandwidths = [self.bandwidth_table[k] for k in device_keys]
            return sum(bandwidths) / len(bandwidths)

        return self.bandwidth_table.get(key, 1e9)

    def get_transfer_bandwidth(self, src_device: str, dst_device: str) -> float:
        """
        查询设备间传输带宽

        Args:
            src_device: 源设备
            dst_device: 目标设备

        Returns:
            带宽（bytes/sec）
        """
        # 简化：使用两个设备中较慢的带宽
        src_bw = self.get_bandwidth(src_device)
        dst_bw = self.get_bandwidth(dst_device)

        return min(src_bw, dst_bw)

    def check_memory_feasible(self, device: str, stages: List[int], n: int, m: int) -> bool:
        """
        检查内存是否可行（简化版，实际需要memory profiling数据）

        Args:
            device: 设备名
            stages: stage列表
            n: 节点数
            m: 边数

        Returns:
            是否可行
        """
        # 简化：假设内存需求 = 节点数 * 特征维度 * 4字节 * 2（输入+输出）
        feature_dim = 500
        estimated_memory = n * feature_dim * 4 * 2

        device_memory = {
            'CPU': 32 * 1024 * 1024 * 1024,  # 32GB
            'GPU': 16 * 1024 * 1024 * 1024,  # 16GB
            'NPU': 8 * 1024 * 1024 * 1024,   # 8GB
        }

        return estimated_memory < device_memory.get(device, float('inf'))
