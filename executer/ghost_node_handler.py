"""
Ghost Node Feature Handler
预先收集所有ghost node特征
"""

import torch
from typing import Dict


class GhostNodeHandler:
    """
    Ghost Node特征收集器

    策略: 预先收集所有subgraph需要的ghost node特征
    优点: 简单，无通信开销
    缺点: 内存占用（但Flickr数据集不大）
    """

    def __init__(self, data_loader):
        """
        Args:
            data_loader: GraphDataLoader实例
        """
        self.data_loader = data_loader
        self.full_features = data_loader.get_full_features()

        # 预先收集所有ghost features
        self.ghost_features = self._collect_all_ghost_features()

        total_ghosts = sum(len(gf) for gf in self.ghost_features.values())
        print(f"  ✓ Ghost features collected: {total_ghosts:,} total ghost nodes")

    def _collect_all_ghost_features(self) -> Dict[int, torch.Tensor]:
        """
        为每个subgraph收集ghost node特征

        Returns:
            {sg_id: ghost_features tensor}
        """
        ghost_features = {}

        for sg_id in range(self.data_loader.num_subgraphs):
            ghost_info = self.data_loader.ghost_node_mapping[sg_id]
            ghost_nodes = ghost_info['ghost_nodes']

            if len(ghost_nodes) > 0:
                # 从完整特征中提取ghost nodes的特征
                ghost_features[sg_id] = self.full_features[ghost_nodes]
            else:
                # 没有ghost nodes
                ghost_features[sg_id] = torch.zeros(0, self.full_features.shape[1])

        return ghost_features

    def get_ghost_features(self, subgraph_id: int) -> torch.Tensor:
        """
        获取subgraph的ghost node特征

        Args:
            subgraph_id: Subgraph ID

        Returns:
            ghost_features: [num_ghosts, feat_dim]
        """
        return self.ghost_features[subgraph_id]

    def get_combined_features(self, subgraph_id: int, owned_features: torch.Tensor) -> torch.Tensor:
        """
        合并owned和ghost特征

        Args:
            subgraph_id: Subgraph ID
            owned_features: [n, feat_dim] 拥有节点的特征

        Returns:
            combined: [n + num_ghosts, feat_dim] 完整特征
        """
        ghost_feats = self.get_ghost_features(subgraph_id)

        if len(ghost_feats) == 0:
            return owned_features

        return torch.cat([owned_features, ghost_feats], dim=0)
