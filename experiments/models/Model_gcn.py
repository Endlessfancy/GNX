import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, Size
from torch_scatter import scatter_add

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
# Full GCN Model (Reference Implementation)
# ====================================================================

class gcn_full_10(torch.nn.Module):
    """Full GCN model using PyG's GCNConv for reference/validation."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        feature_sizes = [in_channels] + [hidden_channels] * num_layers
        self.feature_sizes = feature_sizes
        for idx in range(num_layers):
            self.convs.append(GCNConv(feature_sizes[idx], feature_sizes[idx + 1]))

    def forward(self, x, edge_index):
        for i in range(10):
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
        return x


# ====================================================================
# 6-Stage Decomposition of GCN (Graph Convolutional Network)
# ====================================================================
#
# GCN Computation Flow (with symmetric normalization):
#   H' = σ( D̃^(-1/2) Ã D̃^(-1/2) H W )
#
# Edge-level normalization implementation:
#   norm_ij = 1 / sqrt(deg_i * deg_j)
#   h_i' = σ( W · Σ_j norm_ij · h_j )
#
# Execution Order:
#   1. COMPUTE_NORM: Calculate edge normalization weights
#   2. GATHER: Collect source node features x_j
#   3. MESSAGE: msg = norm_ij * x_j (weighted message)
#   4. REDUCE_SUM: agg = Σ_j msg_ij (scatter_add)
#   5. TRANSFORM: out = W @ agg + b (linear transform)
#   6. ACTIVATE: output = ReLU(out)
#
# Key Difference from SAGE:
#   - SAGE: GATHER → AGGREGATE → NORMALIZE (node-level mean)
#   - GCN: COMPUTE_NORM → GATHER → WEIGHTED_MSG → AGGREGATE (edge-level norm)
#
# Reusable Stages (same as SAGE):
#   - Stage 2 GATHER: identical to SAGE Stage 1
#   - Stage 4 REDUCE_SUM: identical to SAGE Stage 3
#   - Stage 5 TRANSFORM: identical to SAGE Stage 6
#   - Stage 6 ACTIVATE: identical to SAGE Stage 7
# ====================================================================


class GCNStage1_ComputeNorm(torch.nn.Module):
    """
    Stage 1: COMPUTE_NORM - Calculate symmetric normalization weights

    Input: edge_index [2, num_edges], num_nodes
    Output: norm [num_edges] - edge normalization weights

    Computes: norm_ij = 1 / sqrt(deg_i * deg_j)
    where deg_i and deg_j include self-loops.

    GCN-specific: This pre-computation is unique to GCN.
    Contains scatter operation (NPU incompatible).

    Note: In practice, norm can be pre-computed once per graph and cached.
    """
    def __init__(self, num_nodes_static: int = None):
        super().__init__()
        self.num_nodes_static = num_nodes_static

    def forward(self, edge_index: torch.Tensor, num_nodes: int = None) -> torch.Tensor:
        actual_num_nodes = self.num_nodes_static if self.num_nodes_static is not None else num_nodes
        row, col = edge_index[0], edge_index[1]

        # Compute node degrees (count incoming edges)
        deg = torch.zeros(actual_num_nodes, dtype=torch.float32, device=edge_index.device)
        ones = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
        deg.scatter_add_(0, col, ones)

        # Symmetric normalization: 1 / sqrt(deg_src * deg_dst)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle isolated nodes

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [E]
        return norm


class GCNStage2_Gather(torch.nn.Module):
    """
    Stage 2: GATHER - Collect source node features

    Input: x [num_nodes, feat_dim], edge_index [2, num_edges]
    Output: x_j [num_edges, feat_dim] - features of source nodes

    ✅ REUSABLE: Identical to SAGE Stage 1 (SAGEStage1_Gather)
    Can share profiling results with SAGE.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Extract source node features for each edge
        return x[edge_index[0]]  # [E, F]


class GCNStage3_Message(torch.nn.Module):
    """
    Stage 3: MESSAGE - Apply normalization weights to messages

    Input: x_j [num_edges, feat_dim], norm [num_edges]
    Output: msg [num_edges, feat_dim]

    Computes: msg_ij = norm_ij * x_j

    Note: Different from SAGE's MESSAGE (identity function).
    Similar to GAT's MESSAGE_WEIGHTED (weighted multiplication).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.unsqueeze(-1) * x_j  # [E, F]


class GCNStage4_ReduceSum(torch.nn.Module):
    """
    Stage 4: REDUCE_SUM - Aggregate messages via scatter_add

    Input: msg [num_edges, feat_dim], edge_index [2, num_edges], num_nodes
    Output: agg [num_nodes, feat_dim]

    ✅ REUSABLE: Identical to SAGE Stage 3 (SAGEStage3_ReduceSum)
    Can share profiling results with SAGE.

    Contains scatter operation (NPU incompatible).
    """
    def __init__(self, num_nodes_static: int = None):
        super().__init__()
        self.num_nodes_static = num_nodes_static

    def forward(self, msg: torch.Tensor, edge_index: torch.Tensor, num_nodes: int = None) -> torch.Tensor:
        actual_num_nodes = self.num_nodes_static if self.num_nodes_static is not None else num_nodes

        # Initialize output tensor
        out = torch.zeros(actual_num_nodes, msg.size(1), dtype=msg.dtype, device=msg.device)

        # Aggregate messages to target nodes
        target_nodes = edge_index[1]
        out.index_add_(0, target_nodes, msg)

        return out  # [N, F]


class GCNStage5_Transform(torch.nn.Module):
    """
    Stage 5: TRANSFORM - Linear transformation

    Input: agg [num_nodes, in_channels]
    Output: out [num_nodes, out_channels]

    Computes: out = W @ agg + b

    ✅ REUSABLE: Same operation type as SAGE Stage 6 (SAGEStage6_Transform)
    Note: GCN only transforms aggregated features (no separate self-loop term)
    Can share profiling results for LINEAR operations.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, agg: torch.Tensor) -> torch.Tensor:
        return self.lin(agg)  # [N, F']


class GCNStage6_Activate(torch.nn.Module):
    """
    Stage 6: ACTIVATE - ReLU activation

    Input: out [num_nodes, out_channels]
    Output: h [num_nodes, out_channels]

    ✅ REUSABLE: Identical to SAGE Stage 7 (SAGEStage7_Activate)
    Can share profiling results with SAGE.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


# ====================================================================
# Combined GCN Layer (for validation)
# ====================================================================

class GCNLayerDecomposed(torch.nn.Module):
    """
    Complete GCN layer using the 6-stage decomposition.
    Used for numerical validation against PyG's GCNConv.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.stage1_norm = GCNStage1_ComputeNorm()
        self.stage2_gather = GCNStage2_Gather()
        self.stage3_message = GCNStage3_Message()
        self.stage4_reduce = GCNStage4_ReduceSum()
        self.stage5_transform = GCNStage5_Transform(in_channels, out_channels, bias)
        self.stage6_activate = GCNStage6_Activate()

        # Cache for norm (can be pre-computed once per graph)
        self._cached_norm = None
        self._cached_edge_index = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        num_nodes = x.size(0)

        # Stage 1: Compute normalization (with optional caching)
        if use_cache and self._cached_edge_index is not None and \
           torch.equal(edge_index, self._cached_edge_index):
            norm = self._cached_norm
        else:
            norm = self.stage1_norm(edge_index, num_nodes)
            if use_cache:
                self._cached_norm = norm
                self._cached_edge_index = edge_index.clone()

        # Stage 2: Gather source features
        x_j = self.stage2_gather(x, edge_index)

        # Stage 3: Weighted messages
        msg = self.stage3_message(x_j, norm)

        # Stage 4: Aggregate
        agg = self.stage4_reduce(msg, edge_index, num_nodes)

        # Stage 5: Transform
        out = self.stage5_transform(agg)

        # Stage 6: Activation
        h = self.stage6_activate(out)

        return h


# ====================================================================
# Stage Summary for Profiling
# ====================================================================
#
# | Stage | Name         | NPU Compatible | Reusable From     |
# |-------|--------------|----------------|-------------------|
# | 1     | COMPUTE_NORM | ❌ No (scatter)| GCN specific      |
# | 2     | GATHER       | ✅ Yes         | SAGE Stage 1 ✅   |
# | 3     | MESSAGE      | ✅ Yes         | GAT Stage 5       |
# | 4     | REDUCE_SUM   | ❌ No (scatter)| SAGE Stage 3 ✅   |
# | 5     | TRANSFORM    | ✅ Yes         | SAGE Stage 6 ✅   |
# | 6     | ACTIVATE     | ✅ Yes         | SAGE Stage 7 ✅   |
#
# GCN has the highest reusability: 4 out of 6 stages can reuse SAGE profiling!
# ====================================================================
