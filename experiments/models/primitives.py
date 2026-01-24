"""
GNN Common Primitive Library (最小公共原语集)

This module defines 7 universal primitives that can be reused across
GraphSAGE, GAT, and GCN models, reducing profiling work from ~20 stages to 7 primitives.

Primitive Set:
    - P1: MATMUL          - Matrix multiplication (F.linear / matmul)
    - P2: GATHER          - Index selection (x[idx] / index_select)
    - P3: SCATTER_ADD     - Scatter addition aggregation
    - P4: ELEWISE_MUL     - Element-wise multiplication (broadcast)
    - P5: ELEWISE_ACT     - Element-wise activation (ReLU/ELU)
    - P6: GAT_EDGE_ATT    - GAT-specific edge attention score
    - P7: GAT_SOFTMAX     - GAT-specific edge softmax

Usage:
    Each model (SAGE, GAT, GCN) can be constructed by combining these primitives,
    allowing maximum profiling reuse:

    SAGE: P2 -> (skip) -> P3 -> P3 -> P4 -> P1 -> P5
    GAT:  P1 -> P2(×2) -> P6 -> P7 -> P4 -> P3 -> P5
    GCN:  (preprocess) -> P2 -> P4 -> P3 -> P1 -> P5

NPU Compatibility:
    - P1, P2, P4, P5, P6: ✅ NPU compatible
    - P3, P7: ❌ Contains scatter operations (NPU incompatible)

Author: GNX Team
Version: 2.0 (Primitive-based refactoring)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max

# Optional imports
try:
    from torch_geometric.nn.dense.linear import Linear
except ImportError:
    Linear = nn.Linear


# ============================================================================
# P1: MATMUL - Matrix Multiplication Primitive
# ============================================================================

class P1_Matmul(nn.Module):
    """
    P1: MATMUL - Linear transformation primitive

    Operation: out = W @ x + b (or W @ x if no bias)

    Reused by:
        - SAGE Stage 6 (TRANSFORM): W_l @ mean_agg + W_r @ x
        - GAT Stage 1 (LINEAR): W @ x
        - GCN Stage 5 (TRANSFORM): W @ agg + b

    Input:
        x: [batch_size, in_features]
    Output:
        out: [batch_size, out_features]

    NPU Compatible: ✅ Yes (standard matmul operation)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.lin = Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [N, in_features] or [E, in_features]
        Returns:
            Output tensor [N, out_features] or [E, out_features]
        """
        return self.lin(x)


class P1_Matmul_Dual(nn.Module):
    """
    P1 Variant: Dual linear transformation (for SAGE self-loop)

    Operation: out = W_l @ agg + W_r @ x

    Used by:
        - SAGE Stage 6 (TRANSFORM with self-loop)

    NPU Compatible: ✅ Yes
    """

    def __init__(self, in_features: int, out_features: int, bias_l: bool = True, bias_r: bool = False):
        super().__init__()
        self.lin_l = Linear(in_features, out_features, bias=bias_l)
        self.lin_r = Linear(in_features, out_features, bias=bias_r)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, agg: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agg: Aggregated neighbor features [N, in_features]
            x: Self node features [N, in_features]
        Returns:
            Output tensor [N, out_features]
        """
        return self.lin_l(agg) + self.lin_r(x)


# ============================================================================
# P2: GATHER - Index Selection Primitive
# ============================================================================

class P2_Gather(nn.Module):
    """
    P2: GATHER - Index selection primitive

    Operation: x_j = x[indices]

    Reused by:
        - SAGE Stage 1 (GATHER): x[edge_index[0]]
        - GAT Stage 2 (GATHER_BOTH): 2× GATHER for src and dst
        - GCN Stage 2 (GATHER): x[edge_index[0]]

    Input:
        x: [num_nodes, feat_dim]
        indices: [num_indices] (1D index tensor)
    Output:
        out: [num_indices, feat_dim]

    NPU Compatible: ✅ Yes (simple indexing operation)

    Note:
        For GAT's GATHER_BOTH, call this primitive twice:
            x_src = P2_Gather(x, edge_index[0])
            x_dst = P2_Gather(x, edge_index[1])
        Cost(GATHER_BOTH) = 2 × Cost(GATHER)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [N, F]
            indices: Index tensor [E] or [num_indices]
        Returns:
            Gathered features [E, F] or [num_indices, F]
        """
        return x[indices]


# ============================================================================
# P3: SCATTER_ADD - Scatter Addition Primitive
# ============================================================================

class P3_ScatterAdd(nn.Module):
    """
    P3: SCATTER_ADD - Scatter addition aggregation primitive

    Operation: out[indices[i]] += src[i] for all i

    Reused by:
        - SAGE Stage 3 (REDUCE_SUM): aggregate messages
        - SAGE Stage 4 (REDUCE_COUNT): count neighbors (with ones)
        - GAT Stage 6 (REDUCE_SUM): aggregate weighted messages
        - GCN Stage 4 (REDUCE_SUM): aggregate weighted messages

    Input:
        src: [num_edges, feat_dim] - source values to scatter
        indices: [num_edges] - target indices
        num_nodes: int - output size (dim 0)
    Output:
        out: [num_nodes, feat_dim]

    NPU Compatible: ❌ No (dynamic scatter operation)

    Note:
        This is one of the core bottleneck operations for NPU deployment.
        Consider using CPU/GPU fallback for this primitive.
    """

    def __init__(self, num_nodes_static: int = None):
        super().__init__()
        self.num_nodes_static = num_nodes_static

    def forward(self, src: torch.Tensor, indices: torch.Tensor,
                num_nodes: int = None) -> torch.Tensor:
        """
        Args:
            src: Source tensor [E, F] to aggregate
            indices: Target indices [E]
            num_nodes: Output size (uses static if set in init)
        Returns:
            Aggregated tensor [N, F]
        """
        actual_num_nodes = self.num_nodes_static if self.num_nodes_static else num_nodes

        # Initialize output tensor
        if src.dim() == 1:
            out = torch.zeros(actual_num_nodes, dtype=src.dtype, device=src.device)
        else:
            out = torch.zeros(actual_num_nodes, src.size(-1),
                            dtype=src.dtype, device=src.device)

        # Scatter add
        out.index_add_(0, indices, src)
        return out


class P3_ScatterAdd_Count(nn.Module):
    """
    P3 Variant: Scatter addition for counting (SAGE REDUCE_COUNT)

    Operation: count[indices[i]] += 1 for all i

    Used by:
        - SAGE Stage 4 (REDUCE_COUNT): count number of neighbors

    NPU Compatible: ❌ No
    """

    def __init__(self):
        super().__init__()

    def forward(self, indices: torch.Tensor, num_nodes: int,
                num_edges: int) -> torch.Tensor:
        """
        Args:
            indices: Edge target indices [E]
            num_nodes: Number of nodes
            num_edges: Number of edges
        Returns:
            Count tensor [N]
        """
        ones = torch.ones(num_edges, dtype=torch.float32, device=indices.device)
        count = torch.zeros(num_nodes, dtype=torch.float32, device=indices.device)
        count = scatter_add(ones, indices, dim=0, out=count)
        return count


# ============================================================================
# P4: ELEWISE_MUL - Element-wise Multiplication Primitive
# ============================================================================

class P4_ElewiseMul(nn.Module):
    """
    P4: ELEWISE_MUL - Element-wise multiplication with broadcast

    Operation: out = scalar * tensor (broadcast multiplication)

    Reused by:
        - SAGE Stage 5 (NORMALIZE): sum_agg * (1/count)
        - GAT Stage 5 (MESSAGE_WEIGHTED): alpha * Wx_j
        - GCN Stage 3 (MESSAGE): norm * x_j

    Input:
        scalar: [batch_size] or [batch_size, 1] - scaling factors
        tensor: [batch_size, feat_dim] - feature tensor
    Output:
        out: [batch_size, feat_dim]

    NPU Compatible: ✅ Yes (element-wise operation)

    Note:
        For SAGE normalization: scalar = 1/count (precomputed)
        For GAT: scalar = attention weights alpha
        For GCN: scalar = edge normalization weights
    """

    def __init__(self):
        super().__init__()

    def forward(self, scalar: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scalar: Scaling factors [E] or [N]
            tensor: Feature tensor [E, F] or [N, F]
        Returns:
            Scaled tensor [E, F] or [N, F]
        """
        if scalar.dim() == 1:
            scalar = scalar.unsqueeze(-1)
        return scalar * tensor


class P4_ElewiseDiv(nn.Module):
    """
    P4 Variant: Element-wise division (for SAGE normalization)

    Operation: out = tensor / divisor

    Used by:
        - SAGE Stage 5 (NORMALIZE): sum_agg / count

    NPU Compatible: ✅ Yes
    """

    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: Feature tensor [N, F]
            divisor: Divisor [N] (will be clamped to avoid div by zero)
        Returns:
            Normalized tensor [N, F]
        """
        divisor = torch.clamp(divisor, min=1)
        return tensor / divisor.unsqueeze(-1)


# ============================================================================
# P5: ELEWISE_ACT - Element-wise Activation Primitive
# ============================================================================

class P5_ElewiseAct(nn.Module):
    """
    P5: ELEWISE_ACT - Element-wise activation primitive

    Operation: out = activation(x)

    Reused by:
        - SAGE Stage 7 (ACTIVATE): ReLU
        - GAT Stage 7 (ACTIVATE): ELU
        - GCN Stage 6 (ACTIVATE): ReLU

    Input:
        x: [batch_size, feat_dim]
    Output:
        out: [batch_size, feat_dim]

    NPU Compatible: ✅ Yes (element-wise operation)

    Supported activations: 'relu', 'elu', 'leaky_relu', 'none'
    """

    def __init__(self, activation: str = 'relu', negative_slope: float = 0.01):
        super().__init__()
        self.activation = activation.lower()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [N, F] or [E, F]
        Returns:
            Activated tensor [N, F] or [E, F]
        """
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'elu':
            return F.elu(x)
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(x, self.negative_slope)
        elif self.activation == 'none' or self.activation == 'identity':
            return x
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


# ============================================================================
# P6: GAT_EDGE_ATT - GAT-specific Edge Attention Primitive
# ============================================================================

class P6_GATEdgeAtt(nn.Module):
    """
    P6: GAT_EDGE_ATT - GAT-specific edge attention score computation

    Operation: e_ij = LeakyReLU(a_src · Wx_j + a_dst · Wx_i)

    Used by:
        - GAT Stage 3 (ATTENTION_SCORE)

    This is a fused primitive that combines:
        - Element-wise multiplication with attention vectors
        - Summation
        - LeakyReLU activation

    Input:
        Wx_i: [num_edges, out_channels] - transformed target features
        Wx_j: [num_edges, out_channels] - transformed source features
    Output:
        e: [num_edges] - unnormalized attention scores

    NPU Compatible: ✅ Yes (can be decomposed to basic ops)

    Note:
        Keeping this as a fused primitive preserves kernel fusion benefits
        and avoids over-decomposition that could hurt performance.
    """

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
        """
        Args:
            Wx_i: Transformed target features [E, F']
            Wx_j: Transformed source features [E, F']
        Returns:
            Attention scores [E]
        """
        # Decomposed attention: e = (Wx_i · att_dst) + (Wx_j · att_src)
        e = (Wx_i * self.att_dst).sum(dim=-1) + (Wx_j * self.att_src).sum(dim=-1)
        return F.leaky_relu(e, self.negative_slope)


# ============================================================================
# P7: GAT_SOFTMAX - GAT-specific Edge Softmax Primitive
# ============================================================================

class P7_GATSoftmax(nn.Module):
    """
    P7: GAT_SOFTMAX - GAT-specific edge softmax normalization

    Operation: alpha_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k exp(e_ik)

    Used by:
        - GAT Stage 4 (ATTENTION_SOFTMAX)

    This is a fused primitive that combines:
        - Scatter max (for numerical stability)
        - Exponential
        - Scatter sum
        - Division

    Input:
        e: [num_edges] - unnormalized attention scores
        edge_index: [2, num_edges] - edge connectivity
        num_nodes: int - number of nodes
    Output:
        alpha: [num_edges] - normalized attention weights

    NPU Compatible: ❌ No (contains scatter operations)

    Note:
        This primitive MUST be kept fused because:
        1. Numerical stability requires max subtraction before exp
        2. Splitting would lose correctness guarantees
        3. The scatter operations are tightly coupled
    """

    def __init__(self, num_nodes_static: int = None):
        super().__init__()
        self.num_nodes_static = num_nodes_static

    def forward(self, e: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int = None) -> torch.Tensor:
        """
        Args:
            e: Unnormalized attention scores [E]
            edge_index: Edge indices [2, E]
            num_nodes: Number of nodes
        Returns:
            Normalized attention weights [E]
        """
        actual_num_nodes = self.num_nodes_static if self.num_nodes_static else num_nodes
        target_nodes = edge_index[1]

        # Numerically stable softmax per target node
        # Step 1: Subtract max for numerical stability
        e_max = scatter_max(e, target_nodes, dim=0, dim_size=actual_num_nodes)[0]
        e_stable = e - e_max[target_nodes]

        # Step 2: Compute exp
        exp_e = torch.exp(e_stable)

        # Step 3: Sum exp per target node
        sum_exp = scatter_add(exp_e, target_nodes, dim=0, dim_size=actual_num_nodes)

        # Step 4: Normalize
        alpha = exp_e / (sum_exp[target_nodes] + 1e-16)

        return alpha


# ============================================================================
# Preprocessing Primitives (Not counted in online cost)
# ============================================================================

class Preprocess_GCNNorm(nn.Module):
    """
    GCN Normalization Pre-computation (NOT an online primitive)

    Operation: norm_ij = 1 / sqrt(deg_i * deg_j)

    Used by:
        - GCN Stage 1 (COMPUTE_NORM) - but as PREPROCESSING

    This computation should be done ONCE per static graph and cached.
    It should NOT be included in online cost estimation.

    Input:
        edge_index: [2, num_edges]
        num_nodes: int
    Output:
        norm: [num_edges] - edge normalization weights

    NPU Compatible: ❌ No (contains scatter), but doesn't matter for online cost
    """

    def __init__(self):
        super().__init__()
        self._cached_norm = None
        self._cached_edge_index_hash = None

    def forward(self, edge_index: torch.Tensor, num_nodes: int,
                use_cache: bool = True) -> torch.Tensor:
        """
        Args:
            edge_index: Edge indices [2, E]
            num_nodes: Number of nodes
            use_cache: Whether to use cached result
        Returns:
            Edge normalization weights [E]
        """
        # Simple hash for cache check
        edge_hash = hash((edge_index.data_ptr(), edge_index.shape[1]))

        if use_cache and self._cached_norm is not None and \
           self._cached_edge_index_hash == edge_hash:
            return self._cached_norm

        row, col = edge_index[0], edge_index[1]

        # Compute node degrees
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
        ones = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
        deg.scatter_add_(0, col, ones)

        # Symmetric normalization
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Cache result
        if use_cache:
            self._cached_norm = norm
            self._cached_edge_index_hash = edge_hash

        return norm


# ============================================================================
# Primitive Registry for Profiling
# ============================================================================

PRIMITIVE_REGISTRY = {
    'P1_MATMUL': {
        'class': P1_Matmul,
        'npu_compatible': True,
        'description': 'Matrix multiplication (linear transform)',
        'used_by': ['SAGE', 'GAT', 'GCN'],
        'reuse_rate': 1.0,
    },
    'P1_MATMUL_DUAL': {
        'class': P1_Matmul_Dual,
        'npu_compatible': True,
        'description': 'Dual linear transform (SAGE self-loop)',
        'used_by': ['SAGE'],
        'reuse_rate': 0.33,
    },
    'P2_GATHER': {
        'class': P2_Gather,
        'npu_compatible': True,
        'description': 'Index selection (gather features)',
        'used_by': ['SAGE', 'GAT', 'GCN'],
        'reuse_rate': 1.0,
        'note': 'GAT uses 2× for GATHER_BOTH',
    },
    'P3_SCATTER_ADD': {
        'class': P3_ScatterAdd,
        'npu_compatible': False,
        'description': 'Scatter addition (message aggregation)',
        'used_by': ['SAGE', 'GAT', 'GCN'],
        'reuse_rate': 1.0,
    },
    'P3_SCATTER_ADD_COUNT': {
        'class': P3_ScatterAdd_Count,
        'npu_compatible': False,
        'description': 'Scatter addition for counting',
        'used_by': ['SAGE'],
        'reuse_rate': 0.33,
    },
    'P4_ELEWISE_MUL': {
        'class': P4_ElewiseMul,
        'npu_compatible': True,
        'description': 'Element-wise multiplication (weighted message)',
        'used_by': ['SAGE', 'GAT', 'GCN'],
        'reuse_rate': 1.0,
    },
    'P4_ELEWISE_DIV': {
        'class': P4_ElewiseDiv,
        'npu_compatible': True,
        'description': 'Element-wise division (normalization)',
        'used_by': ['SAGE'],
        'reuse_rate': 0.33,
    },
    'P5_ELEWISE_ACT': {
        'class': P5_ElewiseAct,
        'npu_compatible': True,
        'description': 'Element-wise activation (ReLU/ELU)',
        'used_by': ['SAGE', 'GAT', 'GCN'],
        'reuse_rate': 1.0,
    },
    'P6_GAT_EDGE_ATT': {
        'class': P6_GATEdgeAtt,
        'npu_compatible': True,
        'description': 'GAT attention score computation',
        'used_by': ['GAT'],
        'reuse_rate': 0.33,
    },
    'P7_GAT_SOFTMAX': {
        'class': P7_GATSoftmax,
        'npu_compatible': False,
        'description': 'GAT edge softmax normalization',
        'used_by': ['GAT'],
        'reuse_rate': 0.33,
    },
}

# Profiling priority list (7 core primitives to profile)
PROFILING_PRIMITIVES = [
    'P1_MATMUL',
    'P2_GATHER',
    'P3_SCATTER_ADD',
    'P4_ELEWISE_MUL',
    'P5_ELEWISE_ACT',
    'P6_GAT_EDGE_ATT',
    'P7_GAT_SOFTMAX',
]


def get_primitive_info():
    """Get summary information about all primitives."""
    info = []
    for name, props in PRIMITIVE_REGISTRY.items():
        info.append({
            'name': name,
            'npu_compatible': props['npu_compatible'],
            'description': props['description'],
            'used_by': props['used_by'],
            'reuse_rate': props['reuse_rate'],
        })
    return info


def print_primitive_summary():
    """Print a summary table of all primitives."""
    print("=" * 80)
    print("GNN Common Primitive Library Summary")
    print("=" * 80)
    print(f"{'Primitive':<20} {'NPU':<5} {'Used By':<20} {'Description'}")
    print("-" * 80)
    for name in PROFILING_PRIMITIVES:
        props = PRIMITIVE_REGISTRY[name]
        npu = '✅' if props['npu_compatible'] else '❌'
        used_by = ', '.join(props['used_by'])
        print(f"{name:<20} {npu:<5} {used_by:<20} {props['description']}")
    print("=" * 80)
    print(f"Total primitives to profile: {len(PROFILING_PRIMITIVES)}")
    print(f"NPU compatible: {sum(1 for n in PROFILING_PRIMITIVES if PRIMITIVE_REGISTRY[n]['npu_compatible'])}")


if __name__ == '__main__':
    print_primitive_summary()
