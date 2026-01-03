"""
Graph Data Loader with Partitioning Support
支持METIS partition的图数据加载
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class GraphDataLoader:
    """
    图数据加载器

    功能:
    1. 加载完整图数据（PyG格式）
    2. 根据partition配置划分subgraph
    3. 识别ghost nodes（跨partition的邻居）
    4. 提取subgraph数据
    """

    def __init__(self, dataset_name: str, partition_config: Dict):
        """
        Args:
            dataset_name: 数据集名称 ('flickr', 'reddit', etc.)
            partition_config: Compiler输出的partition配置
        """
        self.dataset_name = dataset_name
        self.partition_config = partition_config
        self.k = partition_config['k']
        self.num_subgraphs = partition_config['num_subgraphs']

        # 加载完整图
        self.full_data = self._load_full_graph()

        # 构建partition映射（需要从METIS结果重建）
        self.partition_mapping = self._build_partition_mapping()

        # 构建ghost node映射
        self.ghost_node_mapping = self._build_ghost_node_mapping()

        print(f"  ✓ Graph loaded: {self.full_data.num_nodes:,} nodes, {self.full_data.num_edges:,} edges")
        print(f"  ✓ Partitioned into {self.k} subgraphs")

    def _load_full_graph(self):
        """Load complete graph data"""
        import sys
        # Add compiler to path using relative path
        compiler_path = str(Path(__file__).parent.parent / 'compiler')
        if compiler_path not in sys.path:
            sys.path.insert(0, compiler_path)
        from utils.graph_loader import GraphLoader

        loader = GraphLoader()
        data = loader.load_dataset(self.dataset_name)

        return data

    def _build_partition_mapping(self) -> Dict[int, torch.Tensor]:
        """
        从compiler的partition配置重建节点到subgraph的映射

        注意: Compiler只给了每个subgraph的统计信息（n, m），
        没有给具体的节点分配。我们需要重新运行METIS或者使用均匀划分。

        简化方案: 根据节点数（n）均匀划分节点ID
        """
        partition_mapping = {}
        node_ptr = 0

        for sg_config in self.partition_config['subgraphs']:
            sg_id = sg_config['id']
            n = sg_config['n']  # 该subgraph的节点数

            # 分配节点ID范围: [node_ptr, node_ptr + n)
            partition_mapping[sg_id] = torch.arange(node_ptr, node_ptr + n)
            node_ptr += n

        return partition_mapping

    def _build_ghost_node_mapping(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        识别每个subgraph需要的ghost nodes

        Ghost nodes: 不属于该subgraph，但作为该subgraph节点的邻居

        Returns:
            {sg_id: {
                'ghost_nodes': tensor of ghost node IDs,
                'source_subgraphs': which subgraphs they belong to
            }}
        """
        edge_index = self.full_data.edge_index
        ghost_mapping = {}

        for sg_id in range(self.num_subgraphs):
            owned_nodes = self.partition_mapping[sg_id]

            # 找到目标节点在owned_nodes中的所有边
            mask = torch.isin(edge_index[1], owned_nodes)
            incoming_edges = edge_index[:, mask]

            # 源节点中不在owned_nodes中的 = ghost nodes
            src_nodes = incoming_edges[0]
            ghost_mask = ~torch.isin(src_nodes, owned_nodes)
            ghost_nodes = src_nodes[ghost_mask].unique()

            # 找到这些ghost nodes属于哪个subgraph
            source_subgraphs = []
            for gn in ghost_nodes:
                for other_sg_id, other_nodes in self.partition_mapping.items():
                    if gn in other_nodes:
                        source_subgraphs.append(other_sg_id)
                        break

            ghost_mapping[sg_id] = {
                'ghost_nodes': ghost_nodes,
                'num_ghosts': len(ghost_nodes)
            }

        return ghost_mapping

    def get_subgraph_data(self, subgraph_id: int) -> Dict:
        """
        获取subgraph的数据

        Args:
            subgraph_id: Subgraph ID (0 to k-1)

        Returns:
            {
                'edge_index': [2, m] 该subgraph内的边（包括指向ghost nodes的边）
                'x': [n + num_ghosts, feat_dim] 节点特征（owned + ghost）
                'owned_nodes': [n] 拥有的节点ID
                'ghost_nodes': [num_ghosts] ghost节点ID
                'node_mapping': {original_id: local_id} 全局ID到局部ID的映射
            }
        """
        owned_nodes = self.partition_mapping[subgraph_id]
        ghost_info = self.ghost_node_mapping[subgraph_id]
        ghost_nodes = ghost_info['ghost_nodes']

        # 合并owned和ghost nodes
        all_nodes = torch.cat([owned_nodes, ghost_nodes])

        # 构建全局ID到局部ID的映射
        node_mapping = {int(nid): i for i, nid in enumerate(all_nodes)}

        # 提取该subgraph的边
        edge_index = self.full_data.edge_index
        mask = torch.isin(edge_index[1], owned_nodes)  # 目标节点在owned_nodes中
        subgraph_edges = edge_index[:, mask]

        # 将全局节点ID映射到局部ID
        local_edge_index = torch.zeros_like(subgraph_edges)
        for i in range(subgraph_edges.shape[1]):
            src_global = int(subgraph_edges[0, i])
            dst_global = int(subgraph_edges[1, i])
            local_edge_index[0, i] = node_mapping[src_global]
            local_edge_index[1, i] = node_mapping[dst_global]

        # 提取特征
        x = self.full_data.x[all_nodes]

        return {
            'edge_index': local_edge_index,
            'x': x,
            'owned_nodes': torch.arange(len(owned_nodes)),  # 局部ID: [0, n)
            'ghost_nodes': torch.arange(len(owned_nodes), len(all_nodes)),  # 局部ID: [n, n+ghosts)
            'node_mapping': node_mapping,
            'global_owned_nodes': owned_nodes,  # 全局ID
            'num_owned': len(owned_nodes),
            'num_ghosts': len(ghost_nodes)
        }

    def get_full_features(self) -> torch.Tensor:
        """返回完整图的节点特征（用于ghost node收集）"""
        return self.full_data.x

    def get_partition_mapping(self) -> Dict[int, torch.Tensor]:
        """返回partition映射"""
        return self.partition_mapping

    def get_max_subgraph_size(self) -> Tuple[int, int]:
        """
        计算所有 subgraph 中的最大节点数和边数（包含 ghost nodes）
        用于 NPU 静态 shape 模型导出

        Returns:
            (max_nodes, max_edges): 最大节点数（owned + ghost）和最大边数
        """
        max_nodes = 0
        max_edges = 0

        for sg_id in range(self.num_subgraphs):
            # 获取 owned nodes 数量
            owned_nodes = self.partition_mapping[sg_id]
            num_owned = len(owned_nodes)

            # 获取 ghost nodes 数量
            ghost_info = self.ghost_node_mapping[sg_id]
            num_ghosts = ghost_info['num_ghosts']

            # 总节点数 = owned + ghost
            total_nodes = num_owned + num_ghosts

            # 计算该 subgraph 的边数
            edge_index = self.full_data.edge_index
            mask = torch.isin(edge_index[1], owned_nodes)
            num_edges = mask.sum().item()

            max_nodes = max(max_nodes, total_nodes)
            max_edges = max(max_edges, num_edges)

        # 添加 10% 安全余量
        max_nodes = int(max_nodes * 1.1)
        max_edges = int(max_edges * 1.1)

        return max_nodes, max_edges
