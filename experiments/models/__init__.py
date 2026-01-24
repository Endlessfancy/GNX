"""
GNN Model Decomposition Module

This module provides both stage-based and primitive-based implementations
for GraphSAGE, GAT, and GCN models.

Structure:
    - Stage-based (original):
        - Model_sage.py: GraphSAGE 7-stage decomposition
        - Model_gat.py: GAT 7-stage decomposition
        - Model_gcn.py: GCN 6-stage decomposition

    - Primitive-based (optimized):
        - primitives.py: 7 universal primitives (P1-P7)
        - model_sage_primitives.py: SAGE using primitives
        - model_gat_primitives.py: GAT using primitives
        - model_gcn_primitives.py: GCN using primitives

Usage:
    # Stage-based (original API)
    from models.Model_sage import SAGEStage1_Gather, SAGEStage3_ReduceSum

    # Primitive-based (new API)
    from models.primitives import P1_Matmul, P2_Gather, P3_ScatterAdd
    from models.model_sage_primitives import GraphSAGEPrimitive
"""

# Primitives (new API)
from .primitives import (
    P1_Matmul, P1_Matmul_Dual,
    P2_Gather,
    P3_ScatterAdd, P3_ScatterAdd_Count,
    P4_ElewiseMul, P4_ElewiseDiv,
    P5_ElewiseAct,
    P6_GATEdgeAtt,
    P7_GATSoftmax,
    Preprocess_GCNNorm,
    PRIMITIVE_REGISTRY,
    PROFILING_PRIMITIVES,
)

# Primitive-based models
from .model_sage_primitives import (
    GraphSAGELayerPrimitive,
    GraphSAGEPrimitive,
)

from .model_gat_primitives import (
    GATLayerPrimitive,
    GATPrimitive,
)

from .model_gcn_primitives import (
    GCNLayerPrimitive,
    GCNPrimitive,
)

# Stage-based models (original API, for backward compatibility)
from .Model_sage import (
    SAGEStage1_Gather,
    SAGEStage2_Message,
    SAGEStage3_ReduceSum,
    SAGEStage4_ReduceCount,
    SAGEStage5_Normalize,
    SAGEStage6_Transform,
    SAGEStage7_Activate,
)

from .Model_gat import (
    GATStage1_Linear,
    GATStage2_GatherBoth,
    GATStage3_AttentionScore,
    GATStage4_AttentionSoftmax,
    GATStage5_MessageWeighted,
    GATStage6_ReduceSum,
    GATStage7_Activate,
    GATLayerDecomposed,
)

from .Model_gcn import (
    GCNStage1_ComputeNorm,
    GCNStage2_Gather,
    GCNStage3_Message,
    GCNStage4_ReduceSum,
    GCNStage5_Transform,
    GCNStage6_Activate,
    GCNLayerDecomposed,
)

__all__ = [
    # Primitives
    'P1_Matmul', 'P1_Matmul_Dual',
    'P2_Gather',
    'P3_ScatterAdd', 'P3_ScatterAdd_Count',
    'P4_ElewiseMul', 'P4_ElewiseDiv',
    'P5_ElewiseAct',
    'P6_GATEdgeAtt',
    'P7_GATSoftmax',
    'Preprocess_GCNNorm',
    'PRIMITIVE_REGISTRY',
    'PROFILING_PRIMITIVES',

    # Primitive-based models
    'GraphSAGELayerPrimitive', 'GraphSAGEPrimitive',
    'GATLayerPrimitive', 'GATPrimitive',
    'GCNLayerPrimitive', 'GCNPrimitive',

    # Stage-based models (backward compatibility)
    'SAGEStage1_Gather', 'SAGEStage2_Message', 'SAGEStage3_ReduceSum',
    'SAGEStage4_ReduceCount', 'SAGEStage5_Normalize', 'SAGEStage6_Transform',
    'SAGEStage7_Activate',
    'GATStage1_Linear', 'GATStage2_GatherBoth', 'GATStage3_AttentionScore',
    'GATStage4_AttentionSoftmax', 'GATStage5_MessageWeighted',
    'GATStage6_ReduceSum', 'GATStage7_Activate', 'GATLayerDecomposed',
    'GCNStage1_ComputeNorm', 'GCNStage2_Gather', 'GCNStage3_Message',
    'GCNStage4_ReduceSum', 'GCNStage5_Transform', 'GCNStage6_Activate',
    'GCNLayerDecomposed',
]
