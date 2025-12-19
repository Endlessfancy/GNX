"""
2D Interpolation for Profiling Lookup Tables
"""

import numpy as np
from scipy.interpolate import interp2d
from typing import Dict, Tuple, List


class Interpolator2D:
    """
    2D插值器，用于查询profiling lookup table中没有的(n, m)组合
    """

    def __init__(self, data_points: Dict[Tuple[int, int], float]):
        """
        初始化插值器

        Args:
            data_points: {(n, m): time_ms} 字典
        """
        self.data_points = data_points

        # 提取所有的n和m值
        points = list(data_points.keys())
        self.n_values = sorted(set(p[0] for p in points))
        self.m_values = sorted(set(p[1] for p in points))

        # 构建2D网格
        self.n_grid, self.m_grid = np.meshgrid(self.n_values, self.m_values, indexing='ij')
        self.z_grid = np.zeros_like(self.n_grid, dtype=float)

        # 填充已知数据点
        for i, n in enumerate(self.n_values):
            for j, m in enumerate(self.m_values):
                if (n, m) in data_points:
                    self.z_grid[i, j] = data_points[(n, m)]
                else:
                    # 对于缺失的点，使用最近邻
                    self.z_grid[i, j] = self._nearest_neighbor(n, m)

        # 创建插值函数
        try:
            self.interp_func = interp2d(
                self.n_values, self.m_values, self.z_grid.T,
                kind='linear', bounds_error=False, fill_value=None
            )
        except Exception:
            # 如果数据点不够，fallback到nearest neighbor
            self.interp_func = None

    def _nearest_neighbor(self, n: int, m: int) -> float:
        """最近邻查找"""
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
        查询(n, m)对应的值（带插值）

        Args:
            n: 节点数
            m: 边数

        Returns:
            插值后的时间（ms）
        """
        # 检查是否在已知点中
        if (n, m) in self.data_points:
            return self.data_points[(n, m)]

        # 使用插值
        if self.interp_func is not None:
            try:
                result = self.interp_func(n, m)[0]
                # 确保非负
                return max(0.0, float(result))
            except Exception:
                pass

        # Fallback to nearest neighbor
        return self._nearest_neighbor(n, m)

    def query_batch(self, queries: List[Tuple[int, int]]) -> List[float]:
        """批量查询"""
        return [self.query(n, m) for n, m in queries]
