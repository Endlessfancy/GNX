"""
Fused Baseline Models for GNN Profiling

These models fuse all stages into a single block for each GNN type,
serving as fair baselines for comparing against the multi-device pipeline.

Supported models:
- GraphSAGE: FusedBlock0_7 (7 stages)
- GCN: FusedGCN (6 stages)
- GAT: FusedGAT (7 stages)
"""

from .Model_sage import FusedBlock0_7, FusedBlock0, FusedBlock1
from .Model_gcn import FusedGCN
from .Model_gat import FusedGAT

MODEL_REGISTRY = {
    'graphsage': FusedBlock0_7,
    'gcn': FusedGCN,
    'gat': FusedGAT,
}

__all__ = [
    'FusedBlock0_7', 'FusedBlock0', 'FusedBlock1',
    'FusedGCN',
    'FusedGAT',
    'MODEL_REGISTRY'
]
