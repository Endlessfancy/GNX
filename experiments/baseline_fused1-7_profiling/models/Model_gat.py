"""
GAT Fused Models for Baseline Profiling

This module contains:
- FusedGAT: All 7 stages fused together (single-device baseline)

GAT 7-Stage Decomposition:
  Stage 1: GATHER        - Collect neighbor features
  Stage 2: ATTN_COMPUTE  - Compute attention scores (Wh_i || Wh_j) * a
  Stage 3: ATTN_SOFTMAX  - Apply LeakyReLU + edge-wise softmax
  Stage 4: ATTN_REDUCE   - Weighted sum aggregation
  Stage 5: NORMALIZE     - Optional: concat/average multi-head
  Stage 6: TRANSFORM     - Residual connection (if needed)
  Stage 7: ACTIVATE      - ELU activation

Note: This implementation does NOT use PyG, uses manual operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max


class FusedGAT(nn.Module):
    """
    Fused GAT: All 7 stages combined (Single-device baseline)

    Single-head GAT for simplicity (can extend to multi-head).

    Stages:
    1. GATHER: Collect neighbor features and transform
    2. ATTN_COMPUTE: Compute attention scores
    3. ATTN_SOFTMAX: Softmax normalization per target node
    4. ATTN_REDUCE: Weighted sum aggregation
    5. NORMALIZE: (Identity for single head)
    6. TRANSFORM: (Identity - no residual for basic GAT)
    7. ACTIVATE: ELU activation

    Input: x [num_nodes, feat_dim], edge_index [2, num_edges]
    Output: activated [num_nodes, out_dim]
    """
    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 0.2):
        super().__init__()
        self.out_channels = out_channels
        self.negative_slope = negative_slope

        # Linear transformation for node features
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, out_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        # ===== Stage 1: GATHER + Transform =====
        # Transform all node features first
        h = self.lin(x)  # [num_nodes, out_channels]

        # Get transformed features for source and target nodes of each edge
        h_src = h[source_nodes]  # [num_edges, out_channels]
        h_dst = h[target_nodes]  # [num_edges, out_channels]

        # ===== Stage 2: ATTN_COMPUTE =====
        # Compute attention scores: a^T * [Wh_i || Wh_j]
        # Using additive attention: att_src * h_src + att_dst * h_dst
        alpha = (h_src * self.att_src).sum(dim=-1) + (h_dst * self.att_dst).sum(dim=-1)

        # Apply LeakyReLU
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)

        # ===== Stage 3: ATTN_SOFTMAX =====
        # Softmax per target node (edge-wise softmax)
        # First, compute max per target for numerical stability
        alpha_max = torch.zeros(num_nodes, dtype=alpha.dtype, device=alpha.device)
        alpha_max = scatter_max(alpha, target_nodes, dim=0, out=alpha_max)[0]
        alpha = alpha - alpha_max[target_nodes]

        # Compute exp
        alpha_exp = torch.exp(alpha)

        # Sum per target node
        alpha_sum = torch.zeros(num_nodes, dtype=alpha.dtype, device=alpha.device)
        alpha_sum = scatter_add(alpha_exp, target_nodes, dim=0, out=alpha_sum)

        # Normalize
        alpha_softmax = alpha_exp / (alpha_sum[target_nodes] + 1e-16)

        # ===== Stage 4: ATTN_REDUCE =====
        # Weighted sum aggregation
        messages = h_src * alpha_softmax.unsqueeze(-1)
        agg = torch.zeros(num_nodes, self.out_channels, dtype=h.dtype, device=h.device)
        agg.index_add_(0, target_nodes, messages)

        # ===== Stage 5: NORMALIZE =====
        # (Identity for single head - no multi-head concat/average)
        out = agg

        # ===== Stage 6: TRANSFORM =====
        # (Identity - no residual connection in basic GAT)

        # ===== Stage 7: ACTIVATE =====
        # ELU activation
        activated = F.elu(out)

        return activated
