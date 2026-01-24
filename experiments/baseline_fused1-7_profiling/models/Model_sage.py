"""
GraphSAGE Fused Models for Baseline Profiling

This module contains:
- FusedBlock0: Stages 1-4 (GATHER + MESSAGE + REDUCE_SUM + REDUCE_COUNT)
- FusedBlock1: Stages 5-7 (NORMALIZE + TRANSFORM + ACTIVATE)
- FusedBlock0_7: All 7 stages fused together (single-device baseline)

FusedBlock0_7 is used as the baseline for fair comparison against
the multi-device pipeline (FusedBlock0 on CPU/GPU + FusedBlock1 on NPU).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class FusedBlock0(nn.Module):
    """
    Fused Block 0: Stages 1-4 (Aggregation phase)
    - GATHER: Collect neighbor features via edge_index
    - MESSAGE: Identity for basic GraphSAGE
    - REDUCE_SUM: Sum neighbor features to target nodes
    - REDUCE_COUNT: Count neighbors for mean calculation

    Input: x [num_nodes, feat_dim], edge_index [2, num_edges]
    Output: sum_agg [num_nodes, feat_dim], count [num_nodes], x [num_nodes, feat_dim]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # GATHER: Get source (neighbor) node features for each edge
        x_j = x[edge_index[0]]

        # (MESSAGE is identity for basic GraphSAGE - omitted)

        # REDUCE_SUM: Aggregate neighbor features to target nodes
        target_nodes = edge_index[1]
        sum_agg = torch.zeros(num_nodes, x_j.size(1), dtype=x_j.dtype, device=x_j.device)
        sum_agg.index_add_(0, target_nodes, x_j)

        # REDUCE_COUNT: Count number of neighbors for each node
        ones = torch.ones(num_edges, dtype=torch.float32, device=edge_index.device)
        count = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
        count = scatter_add(ones, target_nodes, dim=0, out=count)

        return sum_agg, count, x


class FusedBlock1(nn.Module):
    """
    Fused Block 1: Stages 5-7 (Update phase)
    - NORMALIZE: Compute mean aggregation (sum_agg / count)
    - TRANSFORM: Linear combination (lin_l(mean_agg) + lin_r(x))
    - ACTIVATE: ReLU activation

    Input: sum_agg [num_nodes, feat_dim], count [num_nodes], x [num_nodes, feat_dim]
    Output: activated [num_nodes, out_dim]
    """
    def __init__(self, in_channels: int, out_channels: int, bias_l=True, bias_r=False):
        super().__init__()
        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias_l)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=bias_r)

    def forward(self, sum_agg: torch.Tensor, count: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # NORMALIZE: Compute mean aggregation
        count_clamped = torch.clamp(count, min=1.0)
        mean_agg = sum_agg / count_clamped.unsqueeze(-1)

        # TRANSFORM: Linear combination
        out = self.lin_l(mean_agg) + self.lin_r(x)

        # ACTIVATE: ReLU
        activated = F.relu(out)

        return activated


class FusedBlock0_7(nn.Module):
    """
    Fused Block 0-7: All 7 stages combined (Single-device baseline)

    This is a fair baseline for comparing against the multi-device pipeline.
    It uses the same stage implementations as FusedBlock0 + FusedBlock1,
    but runs entirely on a single device (CPU or GPU).

    Stages:
    1. GATHER: Collect neighbor features
    2. MESSAGE: Identity transform
    3. REDUCE_SUM: Sum aggregation
    4. REDUCE_COUNT: Count neighbors
    5. NORMALIZE: Mean aggregation
    6. TRANSFORM: Linear layers
    7. ACTIVATE: ReLU

    Input: x [num_nodes, feat_dim], edge_index [2, num_edges]
    Output: activated [num_nodes, out_dim]
    """
    def __init__(self, in_channels: int, out_channels: int, bias_l=True, bias_r=False):
        super().__init__()
        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias_l)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=bias_r)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # ===== Block 0: Stages 1-4 (Aggregation) =====

        # Stage 1 - GATHER: Get source node features for each edge
        x_j = x[edge_index[0]]

        # Stage 2 - MESSAGE: Identity (omitted)

        # Stage 3 - REDUCE_SUM: Aggregate to target nodes
        target_nodes = edge_index[1]
        sum_agg = torch.zeros(num_nodes, x_j.size(1), dtype=x_j.dtype, device=x_j.device)
        sum_agg.index_add_(0, target_nodes, x_j)

        # Stage 4 - REDUCE_COUNT: Count neighbors
        ones = torch.ones(num_edges, dtype=torch.float32, device=edge_index.device)
        count = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
        count = scatter_add(ones, target_nodes, dim=0, out=count)

        # ===== Block 1: Stages 5-7 (Update) =====

        # Stage 5 - NORMALIZE: Compute mean
        count_clamped = torch.clamp(count, min=1.0)
        mean_agg = sum_agg / count_clamped.unsqueeze(-1)

        # Stage 6 - TRANSFORM: Linear layers
        out = self.lin_l(mean_agg) + self.lin_r(x)

        # Stage 7 - ACTIVATE: ReLU
        activated = F.relu(out)

        return activated
