"""
GCN Stage Models - ONNX Compatible Version

将 scatter 操作替换为 ONNX 支持的操作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear


class GCNStage1_ComputeNorm_ONNX(torch.nn.Module):
    """
    Stage 1: COMPUTE_NORM - ONNX 兼容版本

    计算边的归一化权重: norm_ij = 1 / sqrt(deg_i * deg_j)

    使用 one-hot 矩阵替代 scatter_add 来计算度数
    """
    def __init__(self, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_index: [2, num_edges]
        Returns:
            norm: [num_edges] - 边的归一化权重
        """
        row, col = edge_index[0], edge_index[1]
        num_edges = edge_index.size(1)

        # 使用 one-hot 计算度数
        # col_onehot[e, n] = 1 if edge e points to node n
        col_onehot = F.one_hot(col, num_classes=self.num_nodes).float()  # [E, N]

        # deg[n] = 入度 = 有多少条边指向节点 n
        deg = col_onehot.sum(dim=0)  # [N]

        # 计算 deg^(-0.5)，处理孤立节点
        deg_inv_sqrt = torch.pow(deg + 1e-16, -0.5)
        deg_inv_sqrt = torch.where(deg > 0, deg_inv_sqrt, torch.zeros_like(deg_inv_sqrt))

        # norm_ij = deg_i^(-0.5) * deg_j^(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [E]

        return norm


class GCNStage2_Gather(torch.nn.Module):
    """Stage 2: GATHER - 无需修改，已兼容 ONNX"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return x[edge_index[0]]


class GCNStage3_Message(torch.nn.Module):
    """Stage 3: MESSAGE - 无需修改，已兼容 ONNX"""
    def __init__(self):
        super().__init__()

    def forward(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.unsqueeze(-1) * x_j


class GCNStage4_ReduceSum_ONNX(torch.nn.Module):
    """
    Stage 4: REDUCE_SUM - ONNX 兼容版本

    使用矩阵乘法替代 scatter_add
    """
    def __init__(self, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes

    def forward(self, msg: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            msg: [num_edges, feat_dim]
            edge_index: [2, num_edges]
        Returns:
            agg: [num_nodes, feat_dim]
        """
        target_nodes = edge_index[1]

        # 创建聚合矩阵 [N, E]
        agg_matrix = F.one_hot(target_nodes, num_classes=self.num_nodes).float().T

        # 矩阵乘法实现 scatter_add
        agg = agg_matrix @ msg  # [N, E] @ [E, F] = [N, F]

        return agg


class GCNStage5_Transform(torch.nn.Module):
    """Stage 5: TRANSFORM - 无需修改，已兼容 ONNX"""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = Linear(in_channels, out_channels, bias=bias)

    def forward(self, agg: torch.Tensor) -> torch.Tensor:
        return self.lin(agg)


class GCNStage6_Activate(torch.nn.Module):
    """Stage 6: ACTIVATE - 无需修改，已兼容 ONNX"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


# ============================================================================
# 完整的 ONNX 兼容 GCN Layer
# ============================================================================

class GCNLayerONNX(torch.nn.Module):
    """
    ONNX 兼容的 GCN Layer

    注意：需要在初始化时指定 num_nodes
    """
    def __init__(self, in_channels: int, out_channels: int, num_nodes: int):
        super().__init__()
        self.stage1 = GCNStage1_ComputeNorm_ONNX(num_nodes)
        self.stage2 = GCNStage2_Gather()
        self.stage3 = GCNStage3_Message()
        self.stage4 = GCNStage4_ReduceSum_ONNX(num_nodes)
        self.stage5 = GCNStage5_Transform(in_channels, out_channels)
        self.stage6 = GCNStage6_Activate()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        norm = self.stage1(edge_index)
        x_j = self.stage2(x, edge_index)
        msg = self.stage3(x_j, norm)
        agg = self.stage4(msg, edge_index)
        out = self.stage5(agg)
        h = self.stage6(out)
        return h
