"""
GAT Stage Models - Version 2 (Standard PyTorch Operations)

使用标准 PyTorch 操作替代 torch_scatter，使其能够导出 ONNX
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear


class GATStage1_Linear(torch.nn.Module):
    """Stage 1: LINEAR - Node-level linear transformation"""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.lin = Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class GATStage2_GatherBoth(torch.nn.Module):
    """Stage 2: GATHER_BOTH - Collect features for both ends of each edge"""
    def __init__(self):
        super().__init__()

    def forward(self, Wx: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        Wx_j = Wx[edge_index[0]]  # Source features
        Wx_i = Wx[edge_index[1]]  # Target features
        return Wx_i, Wx_j


class GATStage3_AttentionScore(torch.nn.Module):
    """Stage 3: ATTENTION_SCORE - Compute attention coefficients"""
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


class GATStage4_AttentionSoftmax(torch.nn.Module):
    """
    Stage 4: ATTENTION_SOFTMAX - 使用标准 PyTorch 操作

    用 index_add_ 和 index_select 替代 scatter_max 和 scatter_add
    这样在 Windows MIX 环境下可以导出 ONNX（虽然有警告）
    """
    def __init__(self, num_nodes_static: int = None):
        super().__init__()
        self.num_nodes_static = num_nodes_static

    def forward(self, e: torch.Tensor, edge_index: torch.Tensor, num_nodes: int = None) -> torch.Tensor:
        actual_num_nodes = self.num_nodes_static if self.num_nodes_static is not None else num_nodes
        target_nodes = edge_index[1]  # [E]
        num_edges = e.size(0)

        # Step 1: Compute max per target node using index_reduce_
        # 初始化为很小的值
        e_max = torch.full((actual_num_nodes,), float('-inf'), dtype=e.dtype, device=e.device)
        # 使用 scatter_reduce 的 amax 模式（PyTorch 1.12+）
        # 如果不支持，可以用循环或其他方式
        e_max = e_max.scatter_reduce(0, target_nodes, e, reduce='amax', include_self=False)

        # 处理没有入边的节点
        e_max = torch.where(e_max == float('-inf'), torch.zeros_like(e_max), e_max)

        # Step 2: Subtract max for numerical stability
        e_stable = e - e_max[target_nodes]

        # Step 3: Compute exp
        exp_e = torch.exp(e_stable)

        # Step 4: Sum exp per target node using index_add_
        sum_exp = torch.zeros(actual_num_nodes, dtype=exp_e.dtype, device=exp_e.device)
        sum_exp.index_add_(0, target_nodes, exp_e)

        # Step 5: Normalize
        alpha = exp_e / (sum_exp[target_nodes] + 1e-16)

        return alpha


class GATStage4_AttentionSoftmax_Simple(torch.nn.Module):
    """
    Stage 4: ATTENTION_SOFTMAX - 简化版本（不做数值稳定化）

    如果 scatter_reduce 不支持，可以用这个简化版本
    注意：对于很大的 attention score 可能有数值问题
    """
    def __init__(self, num_nodes_static: int = None):
        super().__init__()
        self.num_nodes_static = num_nodes_static

    def forward(self, e: torch.Tensor, edge_index: torch.Tensor, num_nodes: int = None) -> torch.Tensor:
        actual_num_nodes = self.num_nodes_static if self.num_nodes_static is not None else num_nodes
        target_nodes = edge_index[1]

        # 直接计算 exp（不做 max 稳定化）
        exp_e = torch.exp(e)

        # Sum exp per target node
        sum_exp = torch.zeros(actual_num_nodes, dtype=exp_e.dtype, device=exp_e.device)
        sum_exp.index_add_(0, target_nodes, exp_e)

        # Normalize
        alpha = exp_e / (sum_exp[target_nodes] + 1e-16)

        return alpha


class GATStage5_MessageWeighted(torch.nn.Module):
    """Stage 5: MESSAGE_WEIGHTED - Apply attention weights to messages"""
    def __init__(self):
        super().__init__()

    def forward(self, Wx_j: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return alpha.unsqueeze(-1) * Wx_j


class GATStage6_ReduceSum(torch.nn.Module):
    """
    Stage 6: REDUCE_SUM - 使用 index_add_ (与 SAGE Stage 3 相同)

    这个操作在 Windows MIX 环境下可以导出 ONNX（虽然有警告）
    """
    def __init__(self, num_nodes_static: int = None):
        super().__init__()
        self.num_nodes_static = num_nodes_static

    def forward(self, msg: torch.Tensor, edge_index: torch.Tensor, num_nodes: int = None) -> torch.Tensor:
        actual_num_nodes = self.num_nodes_static if self.num_nodes_static is not None else num_nodes

        # 使用 index_add_，与 SAGE Stage 3 完全相同
        out = torch.zeros(actual_num_nodes, msg.size(1), dtype=msg.dtype, device=msg.device)
        target_nodes = edge_index[1]
        out.index_add_(0, target_nodes, msg)

        return out


class GATStage7_Activate(torch.nn.Module):
    """Stage 7: ACTIVATE - ELU activation"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x)


# ============================================================================
# 验证用的完整 Layer
# ============================================================================

class GATLayerV2(torch.nn.Module):
    """使用标准 PyTorch 操作的 GAT Layer"""
    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 0.2,
                 num_nodes_static: int = None, use_simple_softmax: bool = False):
        super().__init__()
        self.stage1 = GATStage1_Linear(in_channels, out_channels)
        self.stage2 = GATStage2_GatherBoth()
        self.stage3 = GATStage3_AttentionScore(out_channels, negative_slope)
        if use_simple_softmax:
            self.stage4 = GATStage4_AttentionSoftmax_Simple(num_nodes_static)
        else:
            self.stage4 = GATStage4_AttentionSoftmax(num_nodes_static)
        self.stage5 = GATStage5_MessageWeighted()
        self.stage6 = GATStage6_ReduceSum(num_nodes_static)
        self.stage7 = GATStage7_Activate()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int = None) -> torch.Tensor:
        if num_nodes is None:
            num_nodes = x.size(0)

        Wx = self.stage1(x)
        Wx_i, Wx_j = self.stage2(Wx, edge_index)
        e = self.stage3(Wx_i, Wx_j)
        alpha = self.stage4(e, edge_index, num_nodes)
        msg = self.stage5(Wx_j, alpha)
        h = self.stage6(msg, edge_index, num_nodes)
        out = self.stage7(h)
        return out
