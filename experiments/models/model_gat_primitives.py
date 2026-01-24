"""
GAT (Graph Attention Network) Model using Common Primitive Library

This module implements GAT (single-head) using the common primitive library,
enabling maximum profiling reuse across GNN models.

Primitive Mapping (7 stages → 7 primitives):
    Stage 1: LINEAR        → P1: MATMUL
    Stage 2: GATHER_BOTH   → P2: GATHER × 2 (src + dst)
    Stage 3: ATT_SCORE     → P6: GAT_EDGE_ATT (fused)
    Stage 4: ATT_SOFTMAX   → P7: GAT_SOFTMAX (fused)
    Stage 5: MSG_WEIGHTED  → P4: ELEWISE_MUL
    Stage 6: REDUCE_SUM    → P3: SCATTER_ADD
    Stage 7: ACTIVATE      → P5: ELEWISE_ACT (ELU)

Key Insight - GATHER_BOTH decomposition:
    Original: GATHER_BOTH outputs (Wx_i, Wx_j) in one operation
    Primitive: 2 × P2_GATHER
        - Wx_j = P2_Gather(Wx, edge_index[0])  # source features
        - Wx_i = P2_Gather(Wx, edge_index[1])  # target features
    Cost: Cost(GATHER_BOTH) = 2 × Cost(P2_GATHER)

Execution Flow:
    x → P1_Matmul → P2_Gather(×2) → P6_GATEdgeAtt → P7_GATSoftmax
      → P4_ElewiseMul → P3_ScatterAdd → P5_ElewiseAct → output

NPU Compatibility:
    - P1, P2, P4, P5, P6: ✅ NPU compatible
    - P3, P7: ❌ Contains scatter operations

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
    P6_GATEdgeAtt,
    P7_GATSoftmax,
)


# ============================================================================
# GAT Layer using Primitives
# ============================================================================

class GATLayerPrimitive(nn.Module):
    """
    GAT layer (single-head) implemented using common primitives.

    Computation Flow:
        h_i' = σ( Σ_j α_ij · W·h_j )

    Primitive decomposition:
        1. P1_Matmul: Wx = W @ x (linear transform BEFORE gather)
        2. P2_Gather × 2: Wx_j = Wx[src], Wx_i = Wx[dst]
        3. P6_GATEdgeAtt: e = LeakyReLU(a · [Wx_i || Wx_j])
        4. P7_GATSoftmax: alpha = softmax(e) per target node
        5. P4_ElewiseMul: msg = alpha * Wx_j
        6. P3_ScatterAdd: h = scatter_add(msg, target)
        7. P5_ElewiseAct: output = ELU(h)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 negative_slope: float = 0.2):
        super().__init__()

        # Initialize primitives
        self.linear = P1_Matmul(in_channels, out_channels, bias=False)
        self.gather = P2_Gather()  # Will be used twice
        self.edge_att = P6_GATEdgeAtt(out_channels, negative_slope)
        self.softmax = P7_GATSoftmax()
        self.message = P4_ElewiseMul()
        self.aggregate = P3_ScatterAdd()
        self.activate = P5_ElewiseAct(activation='elu')

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
        Returns:
            Updated node features [N, F']
        """
        num_nodes = x.size(0)

        # Stage 1: LINEAR - transform all node features
        Wx = self.linear(x)  # [N, F']

        # Stage 2: GATHER_BOTH - decomposed into 2 × GATHER
        Wx_j = self.gather(Wx, edge_index[0])  # Source features [E, F']
        Wx_i = self.gather(Wx, edge_index[1])  # Target features [E, F']

        # Stage 3: ATTENTION_SCORE - compute attention coefficients
        e = self.edge_att(Wx_i, Wx_j)  # [E]

        # Stage 4: ATTENTION_SOFTMAX - normalize per target node
        alpha = self.softmax(e, edge_index, num_nodes)  # [E]

        # Stage 5: MESSAGE_WEIGHTED - weight messages by attention
        msg = self.message(alpha, Wx_j)  # [E, F']

        # Stage 6: REDUCE_SUM - aggregate weighted messages
        h = self.aggregate(msg, edge_index[1], num_nodes)  # [N, F']

        # Stage 7: ACTIVATE - ELU activation
        output = self.activate(h)  # [N, F']

        return output


# ============================================================================
# Individual Primitive Stages (for profiling)
# ============================================================================

class GATPrimitive_Linear(nn.Module):
    """
    GAT Stage 1: LINEAR using P1_Matmul primitive.

    Note: LINEAR is placed BEFORE gather in GAT (unlike SAGE).
    This is the key difference in execution order.

    Profiling key: (P1_MATMUL, num_nodes, in_dim, out_dim)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.primitive = P1_Matmul(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.primitive(x)


class GATPrimitive_GatherBoth(nn.Module):
    """
    GAT Stage 2: GATHER_BOTH using 2 × P2_Gather primitives.

    Key Insight: GATHER_BOTH = 2 × GATHER
    Cost(GATHER_BOTH) = 2 × Cost(P2_GATHER)

    Profiling key: 2 × (P2_GATHER, num_nodes, num_edges, feat_dim)
    """

    def __init__(self):
        super().__init__()
        self.gather_src = P2_Gather()
        self.gather_dst = P2_Gather()

    def forward(self, Wx: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        Wx_j = self.gather_src(Wx, edge_index[0])  # Source [E, F']
        Wx_i = self.gather_dst(Wx, edge_index[1])  # Target [E, F']
        return Wx_i, Wx_j


class GATPrimitive_AttentionScore(nn.Module):
    """
    GAT Stage 3: ATTENTION_SCORE using P6_GATEdgeAtt primitive.

    This is a GAT-specific fused primitive.

    Profiling key: (P6_GAT_EDGE_ATT, num_edges, out_dim)
    """

    def __init__(self, out_channels: int, negative_slope: float = 0.2):
        super().__init__()
        self.primitive = P6_GATEdgeAtt(out_channels, negative_slope)

    def forward(self, Wx_i: torch.Tensor, Wx_j: torch.Tensor) -> torch.Tensor:
        return self.primitive(Wx_i, Wx_j)


class GATPrimitive_AttentionSoftmax(nn.Module):
    """
    GAT Stage 4: ATTENTION_SOFTMAX using P7_GATSoftmax primitive.

    This is a GAT-specific fused primitive containing scatter operations.

    Profiling key: (P7_GAT_SOFTMAX, num_nodes, num_edges)
    """

    def __init__(self, num_nodes_static: int = None):
        super().__init__()
        self.primitive = P7_GATSoftmax(num_nodes_static)

    def forward(self, e: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int = None) -> torch.Tensor:
        return self.primitive(e, edge_index, num_nodes)


class GATPrimitive_MessageWeighted(nn.Module):
    """
    GAT Stage 5: MESSAGE_WEIGHTED using P4_ElewiseMul primitive.

    Profiling key: (P4_ELEWISE_MUL, num_edges, feat_dim)
    """

    def __init__(self):
        super().__init__()
        self.primitive = P4_ElewiseMul()

    def forward(self, Wx_j: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return self.primitive(alpha, Wx_j)


class GATPrimitive_ReduceSum(nn.Module):
    """
    GAT Stage 6: REDUCE_SUM using P3_ScatterAdd primitive.

    This is the same primitive as SAGE Stage 3 - full reuse!

    Profiling key: (P3_SCATTER_ADD, num_nodes, num_edges, feat_dim)
    """

    def __init__(self, num_nodes_static: int = None):
        super().__init__()
        self.primitive = P3_ScatterAdd(num_nodes_static)

    def forward(self, msg: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int = None) -> torch.Tensor:
        return self.primitive(msg, edge_index[1], num_nodes)


class GATPrimitive_Activate(nn.Module):
    """
    GAT Stage 7: ACTIVATE using P5_ElewiseAct primitive.

    Same primitive type as SAGE Stage 7, just different activation function.

    Profiling key: (P5_ELEWISE_ACT, num_nodes, feat_dim, 'elu')
    """

    def __init__(self):
        super().__init__()
        self.primitive = P5_ElewiseAct(activation='elu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.primitive(x)


# ============================================================================
# Full Model
# ============================================================================

class GATPrimitive(nn.Module):
    """
    Multi-layer GAT (single-head) using primitive-based layers.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int = None, num_layers: int = 2,
                 negative_slope: float = 0.2):
        super().__init__()
        self.num_layers = num_layers

        if out_channels is None:
            out_channels = hidden_channels

        self.layers = nn.ModuleList()
        feature_sizes = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        for i in range(num_layers):
            self.layers.append(
                GATLayerPrimitive(feature_sizes[i], feature_sizes[i + 1], negative_slope)
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


# ============================================================================
# Cost Estimation Helper
# ============================================================================

def get_gat_primitive_cost_breakdown(num_nodes: int, num_edges: int,
                                      in_dim: int, out_dim: int) -> dict:
    """
    Get the primitive-based cost breakdown for GAT.

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
        'P1_MATMUL': {
            'input_shape': (num_nodes, in_dim),
            'output_shape': (num_nodes, out_dim),
            'description': 'LINEAR transform',
        },
        'P2_GATHER_SRC': {
            'input_shape': (num_nodes, out_dim),
            'output_shape': (num_edges, out_dim),
            'indices_size': num_edges,
            'description': 'Gather source features',
        },
        'P2_GATHER_DST': {
            'input_shape': (num_nodes, out_dim),
            'output_shape': (num_edges, out_dim),
            'indices_size': num_edges,
            'description': 'Gather target features',
        },
        'P6_GAT_EDGE_ATT': {
            'input_shape': (num_edges, out_dim),
            'output_shape': (num_edges,),
            'description': 'Attention score computation',
        },
        'P7_GAT_SOFTMAX': {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'description': 'Edge softmax normalization',
        },
        'P4_ELEWISE_MUL': {
            'scalar_shape': (num_edges,),
            'tensor_shape': (num_edges, out_dim),
            'description': 'Weighted message',
        },
        'P3_SCATTER_ADD': {
            'input_shape': (num_edges, out_dim),
            'output_shape': (num_nodes, out_dim),
            'indices_size': num_edges,
            'description': 'Message aggregation',
        },
        'P5_ELEWISE_ACT': {
            'input_shape': (num_nodes, out_dim),
            'activation': 'elu',
            'description': 'ELU activation',
        },
    }


def estimate_gat_cost_from_sage(sage_costs: dict, num_edges: int,
                                 out_dim: int) -> dict:
    """
    Estimate GAT costs by reusing SAGE profiling results.

    Key reuse mappings:
        - P1_MATMUL: same as SAGE TRANSFORM (single linear)
        - P2_GATHER: same as SAGE GATHER, used twice
        - P3_SCATTER_ADD: identical to SAGE REDUCE_SUM
        - P4_ELEWISE_MUL: similar to SAGE NORMALIZE (mul operation)
        - P5_ELEWISE_ACT: same operation, different function (ELU vs ReLU)

    Only need to profile separately:
        - P6_GAT_EDGE_ATT: GAT-specific
        - P7_GAT_SOFTMAX: GAT-specific

    Args:
        sage_costs: Dict of SAGE primitive costs from profiling
        num_edges: Number of edges
        out_dim: Output feature dimension

    Returns:
        dict: Estimated GAT costs
    """
    return {
        'P1_MATMUL': sage_costs.get('P1_MATMUL', 0),
        'P2_GATHER': sage_costs.get('P2_GATHER', 0) * 2,  # 2× for GATHER_BOTH
        'P3_SCATTER_ADD': sage_costs.get('P3_SCATTER_ADD', 0),
        'P4_ELEWISE_MUL': sage_costs.get('P4_ELEWISE_MUL', 0),
        'P5_ELEWISE_ACT': sage_costs.get('P5_ELEWISE_ACT', 0),
        'P6_GAT_EDGE_ATT': None,  # Need separate profiling
        'P7_GAT_SOFTMAX': None,   # Need separate profiling
    }


# ============================================================================
# Stage Summary
# ============================================================================

GAT_PRIMITIVE_MAPPING = """
GAT → Primitive Mapping
=======================

| Stage | Original Name    | Primitive             | NPU | Notes                    |
|-------|------------------|-----------------------|-----|--------------------------|
| 1     | LINEAR           | P1_MATMUL             | ✅  | W @ x (before gather!)   |
| 2     | GATHER_BOTH      | P2_GATHER × 2         | ✅  | Wx[src], Wx[dst]         |
| 3     | ATTENTION_SCORE  | P6_GAT_EDGE_ATT       | ✅  | LeakyReLU(a·[xi||xj])    |
| 4     | ATTENTION_SOFTMAX| P7_GAT_SOFTMAX        | ❌  | Edge softmax (scatter)   |
| 5     | MESSAGE_WEIGHTED | P4_ELEWISE_MUL        | ✅  | alpha * Wx_j             |
| 6     | REDUCE_SUM       | P3_SCATTER_ADD        | ❌  | scatter_add              |
| 7     | ACTIVATE         | P5_ELEWISE_ACT        | ✅  | ELU                      |

Key Insight - GATHER_BOTH Decomposition:
    GATHER_BOTH = 2 × P2_GATHER
    Cost(GATHER_BOTH) = 2 × Cost(GATHER)

Profiling Reduction:
- Original: 7 stages
- With primitives: 7 primitives (5 common + 2 GAT-specific)
- Reuse from SAGE: P1, P2, P3, P4, P5 (71%)
- GAT-specific: P6, P7 (29%)
"""


if __name__ == '__main__':
    print(GAT_PRIMITIVE_MAPPING)

    # Test the model
    print("\nTesting GAT Primitive Model...")
    model = GATPrimitive(in_channels=64, hidden_channels=128, num_layers=2)

    x = torch.randn(100, 64)
    edge_index = torch.randint(0, 100, (2, 500))

    with torch.no_grad():
        out = model(x, edge_index)

    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print("✓ Test passed!")
