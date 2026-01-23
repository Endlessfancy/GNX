"""
GNN Models for Baseline Profiling

Complete 1-layer GNN models using PyG's native implementations:
- GraphSAGE1Layer: SAGEConv + ReLU
- GCN1Layer: GCNConv + ReLU
- GAT1Layer: GATConv + ReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv


class GraphSAGE1Layer(nn.Module):
    """Single layer GraphSAGE using PyG's SAGEConv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x, edge_index))


class GCN1Layer(nn.Module):
    """Single layer GCN using PyG's GCNConv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x, edge_index))


class GAT1Layer(nn.Module):
    """Single layer GAT using PyG's GATConv"""
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1):
        super().__init__()
        self.conv = GATConv(in_channels, out_channels, heads=heads, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x, edge_index))


# Model registry for easy access
MODEL_REGISTRY = {
    'graphsage': GraphSAGE1Layer,
    'gcn': GCN1Layer,
    'gat': GAT1Layer,
}


def get_model(model_name: str, in_channels: int, out_channels: int) -> nn.Module:
    """Get model by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](in_channels, out_channels)
