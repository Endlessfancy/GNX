"""
GCN Fused Models for Baseline Profiling

This module contains:
- FusedGCN: All 6 stages fused together (single-device baseline)

GCN 6-Stage Decomposition:
  Stage 1: COMPUTE_NORM - Compute normalization coefficients (deg^-0.5)
  Stage 2: GATHER       - Collect neighbor features
  Stage 3: MESSAGE      - Apply edge-wise normalization
  Stage 4: REDUCE_SUM   - Sum aggregation
  Stage 5: TRANSFORM    - Linear transformation
  Stage 6: ACTIVATE     - ReLU activation

Note: This implementation does NOT use PyG, uses manual operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class FusedGCN(nn.Module):
    """
    Fused GCN: All 6 stages combined (Single-device baseline)

    GCN formula: X' = D^(-0.5) * A * D^(-0.5) * X * W

    Stages:
    1. COMPUTE_NORM: Compute degree normalization (D^-0.5 for source and target)
    2. GATHER: Collect neighbor features
    3. MESSAGE: Apply edge-wise normalization (norm_src * norm_tgt * x_j)
    4. REDUCE_SUM: Sum aggregation
    5. TRANSFORM: Linear transformation
    6. ACTIVATE: ReLU activation

    Input: x [num_nodes, feat_dim], edge_index [2, num_edges]
    Output: activated [num_nodes, out_dim]
    """
    def __init__(self, in_channels: int, out_channels: int, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        # ===== Stage 1: COMPUTE_NORM =====
        # Compute node degrees
        ones = torch.ones(num_edges, dtype=torch.float32, device=edge_index.device)
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
        deg = scatter_add(ones, target_nodes, dim=0, out=deg)

        # Compute D^(-0.5)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1), -0.5)

        # Get normalization for each edge: norm = deg_inv_sqrt[src] * deg_inv_sqrt[tgt]
        norm = deg_inv_sqrt[source_nodes] * deg_inv_sqrt[target_nodes]

        # ===== Stage 2: GATHER =====
        # Get source node features for each edge
        x_j = x[source_nodes]

        # ===== Stage 3: MESSAGE =====
        # Apply edge-wise normalization
        messages = x_j * norm.unsqueeze(-1)

        # ===== Stage 4: REDUCE_SUM =====
        # Sum aggregation to target nodes
        agg = torch.zeros(num_nodes, x.size(1), dtype=x.dtype, device=x.device)
        agg.index_add_(0, target_nodes, messages)

        # ===== Stage 5: TRANSFORM =====
        # Linear transformation
        out = self.lin(agg)

        # ===== Stage 6: ACTIVATE =====
        # ReLU activation
        activated = F.relu(out)

        return activated
