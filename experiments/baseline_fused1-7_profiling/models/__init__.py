"""
Fused 1-7 Baseline Models

These models fuse all 7 stages of GraphSAGE into a single block,
serving as a fair baseline for comparing against the multi-device pipeline.
"""

from .Model_sage import FusedBlock0_7, FusedBlock0, FusedBlock1

MODEL_REGISTRY = {
    'graphsage': FusedBlock0_7,
}

__all__ = ['FusedBlock0_7', 'FusedBlock0', 'FusedBlock1', 'MODEL_REGISTRY']
