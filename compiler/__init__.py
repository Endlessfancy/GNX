"""
GNN Compiler Package
Pipeline-Aware Graph Neural Network Compiler for Heterogeneous Platforms
"""

from .compiler import GNNCompiler
from .utils import CompilerConfig, ProfilingLoader
from .core import (
    GraphPartitioner,
    PEPGenerator,
    CostEstimator,
    GlobalOptimizer,
    ModelCodegen
)

__version__ = '1.0.0'

__all__ = [
    'GNNCompiler',
    'CompilerConfig',
    'ProfilingLoader',
    'GraphPartitioner',
    'PEPGenerator',
    'CostEstimator',
    'GlobalOptimizer',
    'ModelCodegen'
]
