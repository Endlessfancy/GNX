"""
GraphSAGE Model using Common Primitive Library

This module implements GraphSAGE using the common primitive library,
enabling maximum profiling reuse across GNN models.

Primitive Mapping (7 stages → 6 primitives):
    Stage 1: GATHER        → P2: GATHER
    Stage 2: MESSAGE       → (Skip, identity - no cost)
    Stage 3: REDUCE_SUM    → P3: SCATTER_ADD
    Stage 4: REDUCE_COUNT  → P3: SCATTER_ADD (count variant)
    Stage 5: NORMALIZE     → P4: ELEWISE_DIV (or MUL with 1/count)
    Stage 6: TRANSFORM     → P1: MATMUL_DUAL
    Stage 7: ACTIVATE      → P5: ELEWISE_ACT (ReLU)

Execution Flow:
    x → P2_Gather → (identity) → P3_ScatterAdd → P3_ScatterAdd_Count
      → P4_ElewiseDiv → P1_Matmul_Dual → P5_ElewiseAct → output

NPU Compatibility:
    - P2, P4, P5: ✅ NPU compatible
    - P3 (both variants): ❌ Contains scatter operations
    - P1: ✅ NPU compatible

Author: GNX Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import (
    P1_Matmul, P1_Matmul_Dual,
    P2_Gather,
    P3_ScatterAdd, P3_ScatterAdd_Count,
    P4_ElewiseMul, P4_ElewiseDiv,
    P5_ElewiseAct,
)


# ============================================================================
# GraphSAGE Layer using Primitives
# ============================================================================

class GraphSAGELayerPrimitive(nn.Module):
    """
    GraphSAGE layer implemented using common primitives.

    Computation Flow:
        h_i' = σ( W_l · mean({h_j : j ∈ N(i)}) + W_r · h_i )

    Primitive decomposition:
        1. P2_Gather: x_j = x[edge_index[0]]
        2. (Skip): message = x_j (identity, zero cost)
        3. P3_ScatterAdd: sum_agg = scatter_add(x_j, edge_index[1])
        4. P3_ScatterAdd_Count: count = count_neighbors(edge_index[1])
        5. P4_ElewiseDiv: mean_agg = sum_agg / count
        6. P1_Matmul_Dual: out = W_l(mean_agg) + W_r(x)
        7. P5_ElewiseAct: output = ReLU(out)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Initialize primitives
        self.gather = P2_Gather()
        self.scatter_sum = P3_ScatterAdd()
        self.scatter_count = P3_ScatterAdd_Count()
        self.normalize = P4_ElewiseDiv()
        self.transform = P1_Matmul_Dual(in_channels, out_channels)
        self.activate = P5_ElewiseAct(activation='relu')

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
        Returns:
            Updated node features [N, F']
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # Stage 1: GATHER - collect source node features
        x_j = self.gather(x, edge_index[0])  # [E, F]

        # Stage 2: MESSAGE - identity (no-op, skip)
        # messages = x_j (zero cost)

        # Stage 3: REDUCE_SUM - aggregate messages
        sum_agg = self.scatter_sum(x_j, edge_index[1], num_nodes)  # [N, F]

        # Stage 4: REDUCE_COUNT - count neighbors
        count = self.scatter_count(edge_index[1], num_nodes, num_edges)  # [N]

        # Stage 5: NORMALIZE - compute mean
        mean_agg = self.normalize(sum_agg, count)  # [N, F]

        # Stage 6: TRANSFORM - linear transformation with self-loop
        out = self.transform(mean_agg, x)  # [N, F']

        # Stage 7: ACTIVATE - ReLU
        output = self.activate(out)  # [N, F']

        return output


# ============================================================================
# Individual Primitive Stages (for profiling)
# ============================================================================

class SAGEPrimitive_Gather(nn.Module):
    """
    SAGE Stage 1: GATHER using P2_Gather primitive.

    Profiling key: (P2_GATHER, num_nodes, num_edges, feat_dim)
    """

    def __init__(self):
        super().__init__()
        self.primitive = P2_Gather()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.primitive(x, edge_index[0])


class SAGEPrimitive_Message(nn.Module):
    """
    SAGE Stage 2: MESSAGE (identity function).

    This is a NO-OP stage with zero cost.
    Kept for API compatibility but should be SKIPPED in profiling.

    Profiling key: SKIP (identity operation)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j  # Identity - no computation


class SAGEPrimitive_ReduceSum(nn.Module):
    """
    SAGE Stage 3: REDUCE_SUM using P3_ScatterAdd primitive.

    Profiling key: (P3_SCATTER_ADD, num_nodes, num_edges, feat_dim)
    """

    def __init__(self, num_nodes_static: int = None):
        super().__init__()
        self.primitive = P3_ScatterAdd(num_nodes_static)

    def forward(self, messages: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int = None) -> torch.Tensor:
        return self.primitive(messages, edge_index[1], num_nodes)


class SAGEPrimitive_ReduceCount(nn.Module):
    """
    SAGE Stage 4: REDUCE_COUNT using P3_ScatterAdd_Count primitive.

    Profiling key: (P3_SCATTER_ADD_COUNT, num_nodes, num_edges)
    """

    def __init__(self):
        super().__init__()
        self.primitive = P3_ScatterAdd_Count()

    def forward(self, edge_index: torch.Tensor, num_nodes: int,
                num_edges: int) -> torch.Tensor:
        return self.primitive(edge_index[1], num_nodes, num_edges)


class SAGEPrimitive_Normalize(nn.Module):
    """
    SAGE Stage 5: NORMALIZE using P4_ElewiseDiv primitive.

    Profiling key: (P4_ELEWISE_DIV, num_nodes, feat_dim)
    """

    def __init__(self):
        super().__init__()
        self.primitive = P4_ElewiseDiv()

    def forward(self, sum_agg: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        return self.primitive(sum_agg, count)


class SAGEPrimitive_Transform(nn.Module):
    """
    SAGE Stage 6: TRANSFORM using P1_Matmul_Dual primitive.

    Profiling key: (P1_MATMUL_DUAL, num_nodes, in_dim, out_dim)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.primitive = P1_Matmul_Dual(in_channels, out_channels)

    def forward(self, mean_agg: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.primitive(mean_agg, x)


class SAGEPrimitive_Activate(nn.Module):
    """
    SAGE Stage 7: ACTIVATE using P5_ElewiseAct primitive.

    Profiling key: (P5_ELEWISE_ACT, num_nodes, feat_dim, 'relu')
    """

    def __init__(self):
        super().__init__()
        self.primitive = P5_ElewiseAct(activation='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.primitive(x)


# ============================================================================
# Full Model
# ============================================================================

class GraphSAGEPrimitive(nn.Module):
    """
    Multi-layer GraphSAGE using primitive-based layers.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int = None, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers

        if out_channels is None:
            out_channels = hidden_channels

        self.layers = nn.ModuleList()
        feature_sizes = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        for i in range(num_layers):
            self.layers.append(
                GraphSAGELayerPrimitive(feature_sizes[i], feature_sizes[i + 1])
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


# ============================================================================
# Cost Estimation Helper
# ============================================================================

def get_sage_primitive_cost_breakdown(num_nodes: int, num_edges: int,
                                       in_dim: int, out_dim: int) -> dict:
    """
    Get the primitive-based cost breakdown for SAGE.

    Returns a dict mapping primitive names to their expected costs,
    which can be looked up from the profiling table.

    Args:
        num_nodes: Number of nodes
        num_edges: Number of edges
        in_dim: Input feature dimension
        out_dim: Output feature dimension

    Returns:
        dict: Mapping of primitive key to lookup parameters
    """
    return {
        'P2_GATHER': {
            'input_shape': (num_nodes, in_dim),
            'output_shape': (num_edges, in_dim),
            'indices_size': num_edges,
        },
        'P3_SCATTER_ADD': {
            'input_shape': (num_edges, in_dim),
            'output_shape': (num_nodes, in_dim),
            'indices_size': num_edges,
        },
        'P3_SCATTER_ADD_COUNT': {
            'indices_size': num_edges,
            'output_size': num_nodes,
        },
        'P4_ELEWISE_DIV': {
            'tensor_shape': (num_nodes, in_dim),
            'divisor_size': num_nodes,
        },
        'P1_MATMUL_DUAL': {
            'input_shape': (num_nodes, in_dim),
            'output_shape': (num_nodes, out_dim),
        },
        'P5_ELEWISE_ACT': {
            'input_shape': (num_nodes, out_dim),
            'activation': 'relu',
        },
    }


# ============================================================================
# Stage Summary
# ============================================================================

SAGE_PRIMITIVE_MAPPING = """
GraphSAGE → Primitive Mapping
=============================

| Stage | Original Name  | Primitive            | NPU | Notes                    |
|-------|----------------|----------------------|-----|--------------------------|
| 1     | GATHER         | P2_GATHER            | ✅  | x[edge_index[0]]         |
| 2     | MESSAGE        | (Skip)               | -   | Identity, zero cost      |
| 3     | REDUCE_SUM     | P3_SCATTER_ADD       | ❌  | scatter_add              |
| 4     | REDUCE_COUNT   | P3_SCATTER_ADD_COUNT | ❌  | scatter_add (counting)   |
| 5     | NORMALIZE      | P4_ELEWISE_DIV       | ✅  | sum / count              |
| 6     | TRANSFORM      | P1_MATMUL_DUAL       | ✅  | W_l(agg) + W_r(x)        |
| 7     | ACTIVATE       | P5_ELEWISE_ACT       | ✅  | ReLU                     |

Profiling Reduction:
- Original: 7 stages
- With primitives: 5 unique primitives (P1, P2, P3, P4, P5)
- Reuse from common library: 100%
"""


if __name__ == '__main__':
    print(SAGE_PRIMITIVE_MAPPING)

    # Test the model
    print("\nTesting GraphSAGE Primitive Model...")
    model = GraphSAGEPrimitive(in_channels=64, hidden_channels=128, num_layers=2)

    x = torch.randn(100, 64)
    edge_index = torch.randint(0, 100, (2, 500))

    with torch.no_grad():
        out = model(x, edge_index)

    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print("✓ Test passed!")
