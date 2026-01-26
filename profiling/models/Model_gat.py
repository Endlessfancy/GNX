import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, Size
# torch_scatter no longer needed - using native PyTorch scatter operations

# Optional imports for ONNX export and OpenVINO
try:
    import torch.onnx
    import pandas as pd
    import subprocess
    from openvino.runtime import Core
    from openvino.runtime import Tensor
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False


# ====================================================================
# Full GAT Model (Reference Implementation)
# ====================================================================

class gat_full_10(torch.nn.Module):
    """Full GAT model using PyG's GATConv for reference/validation."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=1):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        feature_sizes = [in_channels] + [hidden_channels] * num_layers
        self.feature_sizes = feature_sizes
        for idx in range(num_layers):
            self.convs.append(GATConv(feature_sizes[idx], feature_sizes[idx + 1], heads=heads, concat=False))

    def forward(self, x, edge_index):
        for i in range(10):
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.elu(x)
        return x


# ====================================================================
# 7-Stage Decomposition of GAT (Single-Head Attention)
# ====================================================================
#
# GAT Computation Flow:
#   h_i' = σ( Σ_j α_ij · W·h_j )
#
# Execution Order:
#   1. LINEAR: W @ x (node-level linear transform BEFORE gather)
#   2. GATHER_BOTH: Collect Wx_i, Wx_j for each edge
#   3. ATTENTION_SCORE: e_ij = LeakyReLU(a · [Wx_i || Wx_j])
#   4. ATTENTION_SOFTMAX: α_ij = softmax_j(e_ij) per target node
#   5. MESSAGE_WEIGHTED: msg = α_ij * Wx_j
#   6. REDUCE_SUM: h = Σ_j msg_ij (scatter_add)
#   7. ACTIVATE: output = ELU(h)
#
# Key Difference from SAGE:
#   - GAT: LINEAR → GATHER (transform before gather)
#   - SAGE: GATHER → ... → TRANSFORM (transform after aggregate)
#
# Reusable Stages (same as SAGE):
#   - Stage 6 REDUCE_SUM: identical to SAGE Stage 3
#   - Stage 7 ACTIVATE: similar to SAGE Stage 7 (different activation)
# ====================================================================


class GATStage1_Linear(torch.nn.Module):
    """
    Stage 1: LINEAR - Node-level linear transformation (BEFORE gather)

    Input: x [num_nodes, in_channels]
    Output: Wx [num_nodes, out_channels]

    Note: This is the same type of operation as SAGE Stage 6 TRANSFORM,
    but executed BEFORE gather in GAT (different execution order).
    Can reuse profiling results for LINEAR operations.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.lin = Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)  # [N, F] -> [N, F']


class GATStage2_GatherBoth(torch.nn.Module):
    """
    Stage 2: GATHER_BOTH - Collect features for both ends of each edge

    Input: Wx [num_nodes, out_channels], edge_index [2, num_edges]
    Output: Wx_i [num_edges, out_channels], Wx_j [num_edges, out_channels]

    GAT-specific: Unlike SAGE's GATHER (single output), GAT needs both
    source (Wx_j) and target (Wx_i) features for attention computation.

    Note: Returns two tensors - target (i) and source (j) features.
    """
    def __init__(self):
        super().__init__()

    def forward(self, Wx: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        # edge_index[0] = source nodes, edge_index[1] = target nodes
        Wx_j = Wx[edge_index[0]]  # Source features: [E, F']
        Wx_i = Wx[edge_index[1]]  # Target features: [E, F']
        return Wx_i, Wx_j


class GATStage3_AttentionScore(torch.nn.Module):
    """
    Stage 3: ATTENTION_SCORE - Compute attention coefficients

    Input: Wx_i [num_edges, out_channels], Wx_j [num_edges, out_channels]
    Output: e [num_edges] - unnormalized attention scores

    Computes: e_ij = LeakyReLU(a · [Wx_i || Wx_j])
    where a is a learnable attention vector of size [2 * out_channels]

    GAT-specific: This stage is unique to GAT.
    """
    def __init__(self, out_channels: int, negative_slope: float = 0.2):
        super().__init__()
        # Attention parameters: split into att_src and att_dst for numerical stability
        self.att_src = nn.Parameter(torch.Tensor(1, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, out_channels))
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, Wx_i: torch.Tensor, Wx_j: torch.Tensor) -> torch.Tensor:
        # Compute attention scores using decomposed attention vector
        # e_ij = LeakyReLU((Wx_i · att_dst) + (Wx_j · att_src))
        e = (Wx_i * self.att_dst).sum(dim=-1) + (Wx_j * self.att_src).sum(dim=-1)
        return F.leaky_relu(e, self.negative_slope)  # [E]


class GATStage4_AttentionSoftmax(torch.nn.Module):
    """
    Stage 4: ATTENTION_SOFTMAX - Normalize attention scores per target node

    Input: e [num_edges], edge_index [2, num_edges], num_nodes
    Output: alpha [num_edges] - normalized attention weights

    Computes: α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k exp(e_ik)
    where the softmax is computed over all incoming edges to each node.

    GAT-specific: Contains scatter operations (NPU incompatible).
    """
    def __init__(self, num_nodes_static: int = None):
        super().__init__()
        self.num_nodes_static = num_nodes_static  # For NPU static compilation

    def forward(self, e: torch.Tensor, edge_index: torch.Tensor, num_nodes: int = None) -> torch.Tensor:
        actual_num_nodes = self.num_nodes_static if self.num_nodes_static is not None else num_nodes
        target_nodes = edge_index[1]

        # Numerically stable softmax per target node
        # Step 1: Subtract max for numerical stability (using scatter_reduce instead of scatter_max)
        e_max = torch.full((actual_num_nodes,), -1e9, dtype=e.dtype, device=e.device)
        e_max = e_max.scatter_reduce(0, target_nodes, e, reduce='amax', include_self=True)
        e_stable = e - e_max[target_nodes]

        # Step 2: Compute exp
        exp_e = torch.exp(e_stable)

        # Step 3: Sum exp per target node (using scatter_add)
        sum_exp = torch.zeros(actual_num_nodes, dtype=exp_e.dtype, device=exp_e.device)
        sum_exp = sum_exp.scatter_add(0, target_nodes, exp_e)

        # Step 4: Normalize
        alpha = exp_e / (sum_exp[target_nodes] + 1e-16)

        return alpha  # [E]


class GATStage5_MessageWeighted(torch.nn.Module):
    """
    Stage 5: MESSAGE_WEIGHTED - Apply attention weights to messages

    Input: Wx_j [num_edges, out_channels], alpha [num_edges]
    Output: msg [num_edges, out_channels]

    Computes: msg_ij = α_ij * Wx_j

    Note: Different from SAGE's MESSAGE (identity function).
    Similar operation to GCN's MESSAGE (weighted multiplication).
    """
    def __init__(self):
        super().__init__()

    def forward(self, Wx_j: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return alpha.unsqueeze(-1) * Wx_j  # [E, F']


class GATStage6_ReduceSum(torch.nn.Module):
    """
    Stage 6: REDUCE_SUM - Aggregate weighted messages via scatter_add

    Input: msg [num_edges, out_channels], edge_index [2, num_edges], num_nodes
    Output: h [num_nodes, out_channels]

    ✅ REUSABLE: Identical to SAGE Stage 3 (SAGEStage3_ReduceSum)
    Can share profiling results with SAGE.

    Contains scatter operation (NPU incompatible).
    """
    def __init__(self, num_nodes_static: int = None):
        super().__init__()
        self.num_nodes_static = num_nodes_static  # For NPU static compilation

    def forward(self, msg: torch.Tensor, edge_index: torch.Tensor, num_nodes: int = None) -> torch.Tensor:
        actual_num_nodes = self.num_nodes_static if self.num_nodes_static is not None else num_nodes

        # Initialize output tensor
        out = torch.zeros(actual_num_nodes, msg.size(1), dtype=msg.dtype, device=msg.device)

        # Aggregate messages to target nodes using scatter_add
        target_nodes = edge_index[1]
        # Expand index to match message dimensions for scatter_add
        index_expanded = target_nodes.unsqueeze(1).expand(-1, msg.size(1))
        out = out.scatter_add(0, index_expanded, msg)

        return out  # [N, F']


class GATStage7_Activate(torch.nn.Module):
    """
    Stage 7: ACTIVATE - ELU activation

    Input: h [num_nodes, out_channels]
    Output: out [num_nodes, out_channels]

    ✅ REUSABLE: Similar to SAGE Stage 7 (SAGEStage7_Activate)
    Same operation type, different activation function (ELU vs ReLU).
    Can share profiling results for element-wise activation operations.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x)


# ====================================================================
# Combined GAT Layer (for validation)
# ====================================================================

class GATLayerDecomposed(torch.nn.Module):
    """
    Complete GAT layer using the 7-stage decomposition.
    Used for numerical validation against PyG's GATConv.
    """
    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 0.2):
        super().__init__()
        self.stage1_linear = GATStage1_Linear(in_channels, out_channels)
        self.stage2_gather = GATStage2_GatherBoth()
        self.stage3_attention = GATStage3_AttentionScore(out_channels, negative_slope)
        self.stage4_softmax = GATStage4_AttentionSoftmax()
        self.stage5_message = GATStage5_MessageWeighted()
        self.stage6_reduce = GATStage6_ReduceSum()
        self.stage7_activate = GATStage7_Activate()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)

        # Stage 1: Linear transform
        Wx = self.stage1_linear(x)

        # Stage 2: Gather both ends
        Wx_i, Wx_j = self.stage2_gather(Wx, edge_index)

        # Stage 3: Attention scores
        e = self.stage3_attention(Wx_i, Wx_j)

        # Stage 4: Softmax normalization
        alpha = self.stage4_softmax(e, edge_index, num_nodes)

        # Stage 5: Weighted messages
        msg = self.stage5_message(Wx_j, alpha)

        # Stage 6: Aggregate
        h = self.stage6_reduce(msg, edge_index, num_nodes)

        # Stage 7: Activation
        out = self.stage7_activate(h)

        return out


# ====================================================================
# Stage Summary for Profiling
# ====================================================================
#
# | Stage | Name              | NPU Compatible | Reusable From     |
# |-------|-------------------|----------------|-------------------|
# | 1     | LINEAR            | ✅ Yes         | SAGE Stage 6      |
# | 2     | GATHER_BOTH       | ✅ Yes         | GAT specific      |
# | 3     | ATTENTION_SCORE   | ✅ Yes         | GAT specific      |
# | 4     | ATTENTION_SOFTMAX | ❌ No (scatter)| GAT specific      |
# | 5     | MESSAGE_WEIGHTED  | ✅ Yes         | GCN Stage 3       |
# | 6     | REDUCE_SUM        | ❌ No (scatter)| SAGE Stage 3 ✅   |
# | 7     | ACTIVATE          | ✅ Yes         | SAGE Stage 7 ✅   |
#
# ====================================================================
