"""
GCN (Graph Convolutional Network) Model using Common Primitive Library

This module implements GCN using the common primitive library,
enabling maximum profiling reuse across GNN models.

Primitive Mapping (6 stages → 5 primitives):
    Stage 1: COMPUTE_NORM  → (Preprocess, NOT counted in online cost)
    Stage 2: GATHER        → P2: GATHER
    Stage 3: MESSAGE       → P4: ELEWISE_MUL (norm * x_j)
    Stage 4: REDUCE_SUM    → P3: SCATTER_ADD
    Stage 5: TRANSFORM     → P1: MATMUL
    Stage 6: ACTIVATE      → P5: ELEWISE_ACT (ReLU)

Key Insight - COMPUTE_NORM as Preprocessing:
    For static graphs, edge normalization weights need to be computed only ONCE.
    This should be done as a preprocessing step and cached.
    Therefore, COMPUTE_NORM is NOT counted in online cost estimation.

Execution Flow (Online):
    (precomputed norm) + x → P2_Gather → P4_ElewiseMul → P3_ScatterAdd
                           → P1_Matmul → P5_ElewiseAct → output

NPU Compatibility:
    - P1, P2, P4, P5: ✅ NPU compatible
    - P3: ❌ Contains scatter operations
    - Preprocessing: ❌ (but doesn't matter - one-time cost)

Author: GNX Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import (
    P1_Matmul,
    P2_Gather,
    P3_ScatterAdd,
    P4_ElewiseMul,
    P5_ElewiseAct,
    Preprocess_GCNNorm,
)


# ============================================================================
# GCN Layer using Primitives
# ============================================================================

class GCNLayerPrimitive(nn.Module):
    """
    GCN layer implemented using common primitives.

    Computation Flow:
        H' = σ( D̃^(-1/2) Ã D̃^(-1/2) H W )

    Edge-level implementation:
        h_i' = σ( W · Σ_j norm_ij · h_j )

    Primitive decomposition:
        Pre. Preprocess_GCNNorm: norm = 1/sqrt(deg_i * deg_j) [CACHED]
        1. P2_Gather: x_j = x[edge_index[0]]
        2. P4_ElewiseMul: msg = norm * x_j
        3. P3_ScatterAdd: agg = scatter_add(msg, edge_index[1])
        4. P1_Matmul: out = W @ agg + b
        5. P5_ElewiseAct: output = ReLU(out)
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()

        # Preprocessing (cached, not counted in online cost)
        self.compute_norm = Preprocess_GCNNorm()

        # Online primitives
        self.gather = P2_Gather()
        self.message = P4_ElewiseMul()
        self.aggregate = P3_ScatterAdd()
        self.transform = P1_Matmul(in_channels, out_channels, bias=bias)
        self.activate = P5_ElewiseAct(activation='relu')

        # Cache
        self._cached_norm = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                norm: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            norm: Pre-computed edge normalization weights [E] (optional)
        Returns:
            Updated node features [N, F']
        """
        num_nodes = x.size(0)

        # Preprocessing: Compute or use cached normalization
        if norm is None:
            norm = self.compute_norm(edge_index, num_nodes)

        # Stage 2: GATHER - collect source node features
        x_j = self.gather(x, edge_index[0])  # [E, F]

        # Stage 3: MESSAGE - apply normalization weights
        msg = self.message(norm, x_j)  # [E, F]

        # Stage 4: REDUCE_SUM - aggregate messages
        agg = self.aggregate(msg, edge_index[1], num_nodes)  # [N, F]

        # Stage 5: TRANSFORM - linear transformation
        out = self.transform(agg)  # [N, F']

        # Stage 6: ACTIVATE - ReLU
        output = self.activate(out)  # [N, F']

        return output


# ============================================================================
# Individual Primitive Stages (for profiling)
# ============================================================================

class GCNPrimitive_ComputeNorm(nn.Module):
    """
    GCN Stage 1: COMPUTE_NORM (Preprocessing - NOT counted in online cost)

    This is a preprocessing step that computes edge normalization weights.
    For static graphs, this should be computed ONCE and cached.

    Profiling: SKIP (preprocessing, not online cost)
    """

    def __init__(self):
        super().__init__()
        self.primitive = Preprocess_GCNNorm()

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        return self.primitive(edge_index, num_nodes)


class GCNPrimitive_Gather(nn.Module):
    """
    GCN Stage 2: GATHER using P2_Gather primitive.

    Identical to SAGE Stage 1 - full reuse!

    Profiling key: (P2_GATHER, num_nodes, num_edges, feat_dim)
    """

    def __init__(self):
        super().__init__()
        self.primitive = P2_Gather()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.primitive(x, edge_index[0])


class GCNPrimitive_Message(nn.Module):
    """
    GCN Stage 3: MESSAGE using P4_ElewiseMul primitive.

    Computes: msg = norm * x_j

    Similar to GAT Stage 5 (MESSAGE_WEIGHTED) - reusable!

    Profiling key: (P4_ELEWISE_MUL, num_edges, feat_dim)
    """

    def __init__(self):
        super().__init__()
        self.primitive = P4_ElewiseMul()

    def forward(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return self.primitive(norm, x_j)


class GCNPrimitive_ReduceSum(nn.Module):
    """
    GCN Stage 4: REDUCE_SUM using P3_ScatterAdd primitive.

    Identical to SAGE Stage 3 and GAT Stage 6 - full reuse!

    Profiling key: (P3_SCATTER_ADD, num_nodes, num_edges, feat_dim)
    """

    def __init__(self, num_nodes_static: int = None):
        super().__init__()
        self.primitive = P3_ScatterAdd(num_nodes_static)

    def forward(self, msg: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int = None) -> torch.Tensor:
        return self.primitive(msg, edge_index[1], num_nodes)


class GCNPrimitive_Transform(nn.Module):
    """
    GCN Stage 5: TRANSFORM using P1_Matmul primitive.

    Similar to SAGE Stage 6 and GAT Stage 1 - reusable!
    (Note: GCN has single linear, no self-loop like SAGE)

    Profiling key: (P1_MATMUL, num_nodes, in_dim, out_dim)
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.primitive = P1_Matmul(in_channels, out_channels, bias=bias)

    def forward(self, agg: torch.Tensor) -> torch.Tensor:
        return self.primitive(agg)


class GCNPrimitive_Activate(nn.Module):
    """
    GCN Stage 6: ACTIVATE using P5_ElewiseAct primitive.

    Identical to SAGE Stage 7 - full reuse!

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

class GCNPrimitive(nn.Module):
    """
    Multi-layer GCN using primitive-based layers.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int = None, num_layers: int = 2,
                 bias: bool = True):
        super().__init__()
        self.num_layers = num_layers

        if out_channels is None:
            out_channels = hidden_channels

        self.layers = nn.ModuleList()
        feature_sizes = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        for i in range(num_layers):
            self.layers.append(
                GCNLayerPrimitive(feature_sizes[i], feature_sizes[i + 1], bias)
            )

        # Shared norm preprocessing
        self._preprocess = Preprocess_GCNNorm()
        self._cached_norm = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Compute norm once for all layers (preprocessing)
        num_nodes = x.size(0)
        norm = self._preprocess(edge_index, num_nodes)

        # Forward through all layers
        for layer in self.layers:
            x = layer(x, edge_index, norm=norm)

        return x


# ============================================================================
# Cost Estimation Helper
# ============================================================================

def get_gcn_primitive_cost_breakdown(num_nodes: int, num_edges: int,
                                      in_dim: int, out_dim: int) -> dict:
    """
    Get the primitive-based cost breakdown for GCN.

    Note: COMPUTE_NORM is NOT included as it's preprocessing (one-time cost).

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
        # NOTE: COMPUTE_NORM is preprocessing, not counted in online cost
        'P2_GATHER': {
            'input_shape': (num_nodes, in_dim),
            'output_shape': (num_edges, in_dim),
            'indices_size': num_edges,
            'description': 'Gather source features',
        },
        'P4_ELEWISE_MUL': {
            'scalar_shape': (num_edges,),
            'tensor_shape': (num_edges, in_dim),
            'description': 'Weighted message (norm * x_j)',
        },
        'P3_SCATTER_ADD': {
            'input_shape': (num_edges, in_dim),
            'output_shape': (num_nodes, in_dim),
            'indices_size': num_edges,
            'description': 'Message aggregation',
        },
        'P1_MATMUL': {
            'input_shape': (num_nodes, in_dim),
            'output_shape': (num_nodes, out_dim),
            'description': 'Linear transform',
        },
        'P5_ELEWISE_ACT': {
            'input_shape': (num_nodes, out_dim),
            'activation': 'relu',
            'description': 'ReLU activation',
        },
    }


def estimate_gcn_cost_from_sage(sage_costs: dict) -> dict:
    """
    Estimate GCN costs by reusing SAGE profiling results.

    GCN has the highest reuse rate among the three models!

    Key reuse mappings:
        - P2_GATHER: identical to SAGE GATHER
        - P4_ELEWISE_MUL: similar to SAGE NORMALIZE (element-wise op)
        - P3_SCATTER_ADD: identical to SAGE REDUCE_SUM
        - P1_MATMUL: same operation type as SAGE TRANSFORM (single linear)
        - P5_ELEWISE_ACT: identical to SAGE ACTIVATE

    GCN-specific (NOT counted in online cost):
        - COMPUTE_NORM: preprocessing, cached

    Args:
        sage_costs: Dict of SAGE primitive costs from profiling

    Returns:
        dict: Estimated GCN costs
    """
    return {
        'P2_GATHER': sage_costs.get('P2_GATHER', 0),
        'P4_ELEWISE_MUL': sage_costs.get('P4_ELEWISE_MUL', 0),
        'P3_SCATTER_ADD': sage_costs.get('P3_SCATTER_ADD', 0),
        'P1_MATMUL': sage_costs.get('P1_MATMUL', 0),
        'P5_ELEWISE_ACT': sage_costs.get('P5_ELEWISE_ACT', 0),
        # Note: All primitives can be reused from SAGE!
    }


# ============================================================================
# Stage Summary
# ============================================================================

GCN_PRIMITIVE_MAPPING = """
GCN → Primitive Mapping
=======================

| Stage | Original Name | Primitive          | NPU | Notes                      |
|-------|---------------|--------------------|-----|----------------------------|
| Pre   | COMPUTE_NORM  | (Preprocess)       | -   | Cached, not online cost    |
| 1     | GATHER        | P2_GATHER          | ✅  | x[edge_index[0]]           |
| 2     | MESSAGE       | P4_ELEWISE_MUL     | ✅  | norm * x_j                 |
| 3     | REDUCE_SUM    | P3_SCATTER_ADD     | ❌  | scatter_add                |
| 4     | TRANSFORM     | P1_MATMUL          | ✅  | W @ agg + b                |
| 5     | ACTIVATE      | P5_ELEWISE_ACT     | ✅  | ReLU                       |

Key Insight - COMPUTE_NORM as Preprocessing:
    For static graphs, norm = 1/sqrt(deg_i * deg_j) is computed ONCE.
    This cost should NOT be included in online cost estimation.

Profiling Reduction:
- Original: 6 stages
- With primitives: 5 primitives (all common!)
- Reuse from SAGE: P1, P2, P3, P4, P5 (100%)
- GCN-specific primitives: 0 (highest reuse rate!)
"""


if __name__ == '__main__':
    print(GCN_PRIMITIVE_MAPPING)

    # Test the model
    print("\nTesting GCN Primitive Model...")
    model = GCNPrimitive(in_channels=64, hidden_channels=128, num_layers=2)

    x = torch.randn(100, 64)
    edge_index = torch.randint(0, 100, (2, 500))

    with torch.no_grad():
        out = model(x, edge_index)

    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print("✓ Test passed!")
