"""
Graph Partitioner - Phase 1
使用METIS进行图划分
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass

try:
    import pymetis
    METIS_AVAILABLE = True
except ImportError:
    METIS_AVAILABLE = False
    print("Warning: pymetis not installed. Falling back to random partitioning.")


@dataclass
class Subgraph:
    """Subgraph数据结构"""
    id: int
    n: int  # 节点数
    m: int  # 边数
    n_pad: int  # NPU padding后的节点数
    m_pad: int  # NPU padding后的边数
    cut_edges: int = 0  # 跨subgraph的边数

    def __repr__(self):
        return f"Subgraph(id={self.id}, n={self.n}, m={self.m}, cut={self.cut_edges})"


class GraphPartitioner:
    """
    图划分器

    支持两种模式：
    1. METIS模式：使用真实图拓扑，最小化cut edges
    2. Random模式：简单随机划分（fallback）
    """

    def __init__(self, padding_multiple: int = 1000, use_metis: bool = True):
        """
        Args:
            padding_multiple: NPU padding的倍数
            use_metis: 是否使用METIS（如果可用）
        """
        self.padding_multiple = padding_multiple
        self.use_metis = use_metis and METIS_AVAILABLE

        if use_metis and not METIS_AVAILABLE:
            print("  Warning: METIS requested but not available, using random partitioning")

    def partition_multiple_k(self,
                            data_or_stats: Union[torch.Tensor, Tuple[int, int]],
                            k_set: List[int]) -> Dict[int, List[Subgraph]]:
        """
        对多个k值进行划分

        Args:
            data_or_stats:
                - 如果是Tensor: edge_index [2, num_edges]（METIS模式）
                - 如果是Tuple: (total_nodes, total_edges)（Random模式）
            k_set: k值列表

        Returns:
            {k: [subgraphs]} 字典
        """
        results = {}

        # 检测输入类型
        if isinstance(data_or_stats, torch.Tensor):
            # METIS模式：有真实图结构
            edge_index = data_or_stats
            for k in k_set:
                if self.use_metis:
                    results[k] = self.partition_with_metis(edge_index, k)
                else:
                    # 退化到random，但需要提取节点/边数
                    num_nodes = edge_index.max().item() + 1
                    num_edges = edge_index.shape[1]
                    results[k] = self.partition_random(num_nodes, num_edges, k)
        else:
            # Random模式：只有统计信息
            total_nodes, total_edges = data_or_stats
            for k in k_set:
                results[k] = self.partition_random(total_nodes, total_edges, k)

        return results

    def partition_with_metis(self,
                            edge_index: torch.Tensor,
                            k: int,
                            ufactor: int = 30,
                            seed: int = 42) -> List[Subgraph]:
        """
        使用METIS进行真实图划分

        Args:
            edge_index: [2, num_edges] edge index tensor
            k: 划分数量
            ufactor: 平衡约束 (1-50, 默认30 = 3%不平衡度)
            seed: 随机种子

        Returns:
            k个Subgraph对象，包含真实的cut_edges
        """
        if not METIS_AVAILABLE:
            raise RuntimeError("METIS not available. Install with: pip install pymetis")

        # 设置随机种子（METIS内部使用随机化）
        np.random.seed(seed)

        # 确保无向图（METIS要求）
        edge_index = self._to_undirected(edge_index)
        num_nodes = edge_index.max().item() + 1
        num_edges = edge_index.shape[1]

        print(f"\n  Running METIS k-way partition (k={k})...")
        print(f"    Graph: {num_nodes:,} nodes, {num_edges:,} edges")

        # 构建adjacency list（METIS输入格式）
        adjacency_list = self._build_adjacency_list(edge_index, num_nodes)

        # 调用METIS
        # 设置options：ufactor控制平衡度（默认30=3%）
        # 注意：pymetis需要Options对象
        options = pymetis.Options()
        options.ufactor = ufactor  # 不平衡度因子

        try:
            n_cuts, membership = pymetis.part_graph(
                nparts=k,
                adjacency=adjacency_list,
                options=options,
                recursive=False  # k-way划分
            )
        except Exception as e:
            print(f"  Warning: METIS failed ({e}), falling back to random")
            return self.partition_random(num_nodes, num_edges, k)

        # membership[i] = partition ID for node i (0 to k-1)
        membership = np.array(membership)

        # 计算每个partition的统计信息
        subgraphs = []
        total_ghost_nodes = 0
        total_internal_edges = 0

        for part_id in range(k):
            # 获取该partition的节点
            part_nodes = np.where(membership == part_id)[0]
            n = len(part_nodes)

            # 统计内部边（两端点都在partition内）
            edge_src = edge_index[0].numpy()
            edge_dst = edge_index[1].numpy()

            internal_mask = np.isin(edge_src, part_nodes) & np.isin(edge_dst, part_nodes)
            m = internal_mask.sum() // 2  # 无向图，除以2

            total_internal_edges += m

            # 统计cut edges（一端在内，一端在外）
            cut_mask = (np.isin(edge_src, part_nodes) & ~np.isin(edge_dst, part_nodes))
            cut_edges = cut_mask.sum()

            # 统计ghost nodes（来自其他partition的邻居节点）
            ghost_neighbors = edge_dst[cut_mask]
            ghost_count = len(np.unique(ghost_neighbors))
            total_ghost_nodes += ghost_count

            # Padding
            n_pad = self._pad_to_multiple(n)
            m_pad = self._pad_to_multiple(m)

            sg = Subgraph(
                id=int(part_id),
                n=int(n),
                m=int(m),
                n_pad=int(n_pad),
                m_pad=int(m_pad),
                cut_edges=int(cut_edges)
            )
            subgraphs.append(sg)

        # 打印统计信息
        print(f"    ✓ METIS partition complete:")
        print(f"      Total cut edges: {n_cuts:,} ({n_cuts/num_edges*100:.1f}% of edges)")
        print(f"      Total ghost nodes: {total_ghost_nodes:,} ({total_ghost_nodes/num_nodes*100:.1f}% overhead)")
        print(f"      Avg nodes/partition: {num_nodes/k:.0f}")
        print(f"      Node balance: {min(sg.n for sg in subgraphs)} - {max(sg.n for sg in subgraphs)}")

        return subgraphs

    def partition_random(self,
                        total_nodes: int,
                        total_edges: int,
                        k: int,
                        seed: int = 42) -> List[Subgraph]:
        """
        简单随机划分（fallback）

        Args:
            total_nodes: 总节点数
            total_edges: 总边数
            k: 划分数量
            seed: 随机种子

        Returns:
            k个Subgraph对象（cut_edges=0）
        """
        np.random.seed(seed)

        print(f"\n  Using random partition (k={k})...")

        # 生成节点数（尽量平衡，带少量随机性）
        avg_nodes = total_nodes // k
        node_counts = []

        for i in range(k):
            if i < k - 1:
                # 前k-1个：在平均值附近随机
                variation = int(avg_nodes * 0.2)  # 20%变化
                n = avg_nodes + np.random.randint(-variation, variation + 1)
                n = max(500, n)  # 至少500个节点
                node_counts.append(n)
            else:
                # 最后一个：补齐剩余
                n = total_nodes - sum(node_counts)
                node_counts.append(n)

        # 生成边数（与节点数大致成正比）
        edge_counts = []
        for n in node_counts:
            # 边数大约是节点数的1.5-2.5倍
            ratio = 1.5 + np.random.random()  # 1.5-2.5
            m = int(n * ratio)
            edge_counts.append(m)

        # 归一化边数，使总和匹配
        total_m = sum(edge_counts)
        edge_counts = [int(m * total_edges / total_m) for m in edge_counts]
        edge_counts[-1] += total_edges - sum(edge_counts)  # 补齐差值

        # 创建Subgraph对象
        subgraphs = []
        for i in range(k):
            n = node_counts[i]
            m = edge_counts[i]

            # 计算padding后的大小
            n_pad = self._pad_to_multiple(n)
            m_pad = self._pad_to_multiple(m)

            sg = Subgraph(
                id=i,
                n=n,
                m=m,
                n_pad=n_pad,
                m_pad=m_pad,
                cut_edges=0  # Random模式不计算cut edges
            )
            subgraphs.append(sg)

        print(f"    ✓ Random partition complete (cut_edges not computed)")

        return subgraphs

    def _to_undirected(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        确保图是无向的（METIS要求）

        Args:
            edge_index: [2, num_edges]

        Returns:
            无向图的edge_index
        """
        # 添加反向边
        edge_index_reversed = edge_index.flip(0)
        edge_index_combined = torch.cat([edge_index, edge_index_reversed], dim=1)

        # 去重
        edge_index_unique = torch.unique(edge_index_combined, dim=1)

        return edge_index_unique

    def _build_adjacency_list(self,
                              edge_index: torch.Tensor,
                              num_nodes: int) -> List[np.ndarray]:
        """
        将edge_index转换为adjacency list格式（METIS输入）

        Args:
            edge_index: [2, num_edges]
            num_nodes: 节点总数

        Returns:
            List of numpy arrays, adjacency[i] = neighbors of node i
        """
        adjacency = [[] for _ in range(num_nodes)]

        edge_src = edge_index[0].numpy()
        edge_dst = edge_index[1].numpy()

        for i in range(edge_index.shape[1]):
            src = edge_src[i]
            dst = edge_dst[i]

            # 跳过自环
            if src != dst:
                adjacency[src].append(dst)

        # 转换为numpy array（METIS要求）
        # 并且去重（以防有重复边）
        adjacency_np = [
            np.array(sorted(set(neighbors)), dtype=np.int32)
            for neighbors in adjacency
        ]

        return adjacency_np

    def _pad_to_multiple(self, value: int) -> int:
        """Padding到指定倍数"""
        return ((value + self.padding_multiple - 1) // self.padding_multiple) * self.padding_multiple
