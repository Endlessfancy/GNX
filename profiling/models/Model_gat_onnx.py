"""
GAT Stage Models - ONNX Compatible Version

将 scatter 操作替换为 ONNX 支持的操作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear


class GATStage1_Linear(torch.nn.Module):
    """Stage 1: LINEAR - 无需修改，已兼容 ONNX"""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.lin = Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class GATStage2_GatherBoth(torch.nn.Module):
    """Stage 2: GATHER_BOTH - 无需修改，已兼容 ONNX"""
    def __init__(self):
        super().__init__()

    def forward(self, Wx: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        Wx_j = Wx[edge_index[0]]  # Source features
        Wx_i = Wx[edge_index[1]]  # Target features
        return Wx_i, Wx_j


class GATStage3_AttentionScore(torch.nn.Module):
    """Stage 3: ATTENTION_SCORE - 无需修改，已兼容 ONNX"""
    def __init__(self, out_channels: int, negative_slope: float = 0.2):
        super().__init__()
        self.att_src = nn.Parameter(torch.Tensor(1, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, out_channels))
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, Wx_i: torch.Tensor, Wx_j: torch.Tensor) -> torch.Tensor:
        e = (Wx_i * self.att_dst).sum(dim=-1) + (Wx_j * self.att_src).sum(dim=-1)
        return F.leaky_relu(e, self.negative_slope)


class GATStage4_AttentionSoftmax_ONNX(torch.nn.Module):
    """
    Stage 4: ATTENTION_SOFTMAX - ONNX 兼容版本

    使用邻接矩阵实现 softmax per target node

    原理：
    - 构建稀疏邻接矩阵 A[dst, edge_idx] = 1
    - 使用矩阵运算实现 scatter_max 和 scatter_add
    """
    def __init__(self, num_nodes: int, num_edges: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def forward(self, e: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e: [num_edges] - 未归一化的注意力分数
            edge_index: [2, num_edges]
        Returns:
            alpha: [num_edges] - 归一化的注意力权重
        """
        target_nodes = edge_index[1]  # [E]

        # 方法: 使用 one-hot + 矩阵乘法
        # 创建 one-hot 矩阵 [E, N]，表示每条边属于哪个目标节点
        edge_to_node = F.one_hot(target_nodes, num_classes=self.num_nodes).float()  # [E, N]

        # 1. 计算每个目标节点的 max(e) 用于数值稳定性
        # e_expanded: [E, 1]
        e_expanded = e.unsqueeze(-1)  # [E, 1]

        # 对于每个节点，找到指向它的边的最大 e 值
        # 使用一个大负数掩码非相关边
        mask = edge_to_node.T  # [N, E]
        e_masked = e.unsqueeze(0) * mask + (1 - mask) * (-1e9)  # [N, E]
        e_max_per_node = e_masked.max(dim=1, keepdim=True)[0]  # [N, 1]

        # 获取每条边对应的 e_max
        e_max = (edge_to_node @ e_max_per_node).squeeze(-1)  # [E]

        # 2. 数值稳定的 exp
        e_stable = e - e_max
        exp_e = torch.exp(e_stable)  # [E]

        # 3. 计算每个目标节点的 sum(exp(e))
        # sum_exp[n] = sum of exp_e for all edges pointing to node n
        sum_exp_per_node = edge_to_node.T @ exp_e.unsqueeze(-1)  # [N, 1]

        # 获取每条边对应的 sum_exp
        sum_exp = (edge_to_node @ sum_exp_per_node).squeeze(-1)  # [E]

        # 4. 归一化
        alpha = exp_e / (sum_exp + 1e-16)

        return alpha


class GATStage5_MessageWeighted(torch.nn.Module):
    """Stage 5: MESSAGE_WEIGHTED - 无需修改，已兼容 ONNX"""
    def __init__(self):
        super().__init__()

    def forward(self, Wx_j: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return alpha.unsqueeze(-1) * Wx_j


class GATStage6_ReduceSum_ONNX(torch.nn.Module):
    """
    Stage 6: REDUCE_SUM - ONNX 兼容版本

    使用矩阵乘法替代 scatter_add:
        out = A.T @ msg
    其中 A[edge_idx, dst] = 1
    """
    def __init__(self, num_nodes: int, num_edges: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def forward(self, msg: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            msg: [num_edges, feat_dim]
            edge_index: [2, num_edges]
        Returns:
            h: [num_nodes, feat_dim]
        """
        target_nodes = edge_index[1]  # [E]

        # 创建聚合矩阵 [N, E]
        # agg_matrix[n, e] = 1 if edge e points to node n
        agg_matrix = F.one_hot(target_nodes, num_classes=self.num_nodes).float().T  # [N, E]

        # 矩阵乘法实现 scatter_add
        # h[n] = sum of msg[e] for all edges e pointing to node n
        h = agg_matrix @ msg  # [N, E] @ [E, F] = [N, F]

        return h


class GATStage7_Activate(torch.nn.Module):
    """Stage 7: ACTIVATE - 无需修改，已兼容 ONNX"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x)


# ============================================================================
# 完整的 ONNX 兼容 GAT Layer
# ============================================================================

class GATLayerONNX(torch.nn.Module):
    """
    ONNX 兼容的 GAT Layer

    注意：需要在初始化时指定 num_nodes 和 num_edges
    这意味着这是一个静态图的实现
    """
    def __init__(self, in_channels: int, out_channels: int,
                 num_nodes: int, num_edges: int,
                 negative_slope: float = 0.2):
        super().__init__()
        self.stage1 = GATStage1_Linear(in_channels, out_channels)
        self.stage2 = GATStage2_GatherBoth()
        self.stage3 = GATStage3_AttentionScore(out_channels, negative_slope)
        self.stage4 = GATStage4_AttentionSoftmax_ONNX(num_nodes, num_edges)
        self.stage5 = GATStage5_MessageWeighted()
        self.stage6 = GATStage6_ReduceSum_ONNX(num_nodes, num_edges)
        self.stage7 = GATStage7_Activate()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        Wx = self.stage1(x)
        Wx_i, Wx_j = self.stage2(Wx, edge_index)
        e = self.stage3(Wx_i, Wx_j)
        alpha = self.stage4(e, edge_index)
        msg = self.stage5(Wx_j, alpha)
        h = self.stage6(msg, edge_index)
        out = self.stage7(h)
        return out
