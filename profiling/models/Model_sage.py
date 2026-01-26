import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Flickr
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, Size, SparseTensor

# Optional imports for ONNX export and OpenVINO
try:
    import torch.onnx
    import pandas as pd
    import subprocess
    from openvino.runtime import Core
    from openvino.runtime import Tensor
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# Define the SAGE model
class sage_full_10(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        feature_sizes = [in_channels] + [hidden_channels] * (num_layers)
        self.feature_sizes = feature_sizes
        for idx in range(num_layers):
            self.convs.append(SAGEConv(feature_sizes[idx], feature_sizes[idx + 1]))

    def forward(self, x, edge_index):
        for i in range(10):
            for conv in self.convs:
                x = conv(x, edge_index)
                x = x.relu()
        return x
    
class sage_propagate_10(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="mean"):
        super().__init__(aggr=aggr)

    def forward(self, x: torch.Tensor, edge_index: Adj, size: Size = None) -> torch.Tensor:
        return self.propagate(edge_index, x=x, size=size)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: torch.Tensor) -> torch.Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

class sage_linear_10(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias_l=True, bias_r=False):
        super().__init__()
        self.lin_l = Linear(in_channels, out_channels, bias=bias_l)  # For neighbor features
        self.lin_r = Linear(in_channels, out_channels, bias=bias_r)  # For root node features
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, message: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        for i in range(10):
            # Apply linear transformation to aggregated neighbor features
            out = self.lin_l(message)

            # # Add the transformed root node (self-node) features
            out += self.lin_r(x)

        return out


# ====================================================================
# 7-Stage Decomposition of GraphSAGE
# ====================================================================

class SAGEStage1_Gather(torch.nn.Module):
    """
    Stage 1: GATHER - Neighbor feature gathering
    Input: x [num_nodes, feat_dim], edge_index [2, num_edges]
    Output: x_j [num_edges, feat_dim] - features of source nodes for each edge
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Extract source node features for each edge
        # edge_index[0] contains source node indices
        return x[edge_index[0]]


class SAGEStage2_Message(torch.nn.Module):
    """
    Stage 2: MESSAGE - Message computation (identity for basic SAGE)
    Input: x_j [num_edges, feat_dim]
    Output: messages [num_edges, feat_dim]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_j: torch.Tensor) -> torch.Tensor:
        # For basic GraphSAGE, message function is identity
        return x_j


class SAGEStage3_ReduceSum(torch.nn.Module):
    """
    Stage 3: REDUCE_SUM - Sum aggregation
    Input: messages [num_edges, feat_dim], edge_index [2, num_edges], num_nodes
    Output: sum_agg [num_nodes, feat_dim]

    Note: num_nodes_static parameter added for NPU compatibility.
    When exporting for NPU, set num_nodes_static in __init__ to avoid dynamic shapes.
    """
    def __init__(self, num_nodes_static=None):
        super().__init__()
        self.num_nodes_static = num_nodes_static  # For NPU static compilation

    def forward(self, messages: torch.Tensor, edge_index: torch.Tensor, num_nodes: int = None) -> torch.Tensor:
        # Use static num_nodes if provided (NPU mode), otherwise use forward parameter
        actual_num_nodes = self.num_nodes_static if self.num_nodes_static is not None else num_nodes

        # Initialize output tensor
        out = torch.zeros(actual_num_nodes, messages.size(1), dtype=messages.dtype, device=messages.device)

        # Aggregate messages to target nodes using scatter_add
        # edge_index[1] contains target node indices
        target_nodes = edge_index[1]
        # Expand index to match message dimensions for scatter_add
        index_expanded = target_nodes.unsqueeze(1).expand(-1, messages.size(1))
        out = out.scatter_add(0, index_expanded, messages)

        return out


class SAGEStage4_ReduceCount(torch.nn.Module):
    """
    Stage 4: REDUCE_COUNT - Count neighbors
    Input: edge_index [2, num_edges], num_nodes
    Output: count [num_nodes]
    """
    def __init__(self):
        super().__init__()

    def forward(self, edge_index: torch.Tensor, num_nodes: int, num_edges: int) -> torch.Tensor:
        # Count number of neighbors for each node
        target_nodes = edge_index[1]
        ones = torch.ones(num_edges, dtype=torch.float32, device=edge_index.device)
        count = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
        count = scatter_add(ones, target_nodes, dim=0, out=count)
        return count


class SAGEStage5_Normalize(torch.nn.Module):
    """
    Stage 5: NORMALIZE - Compute mean by dividing sum by count
    Input: sum_agg [num_nodes, feat_dim], count [num_nodes]
    Output: mean_agg [num_nodes, feat_dim]
    """
    def __init__(self):
        super().__init__()

    def forward(self, sum_agg: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        # Avoid division by zero for isolated nodes
        count = torch.clamp(count, min=1)
        # Normalize: divide sum by count
        mean_agg = sum_agg / count.unsqueeze(-1)
        return mean_agg


class SAGEStage6_Transform(torch.nn.Module):
    """
    Stage 6: TRANSFORM - Linear transformations
    Input: mean_agg [num_nodes, in_dim], x [num_nodes, in_dim]
    Output: out [num_nodes, out_dim]
    Combines: lin_l(mean_agg) + lin_r(x)
    """
    def __init__(self, in_channels: int, out_channels: int, bias_l=True, bias_r=False):
        super().__init__()
        self.lin_l = Linear(in_channels, out_channels, bias=bias_l)  # For neighbor features
        self.lin_r = Linear(in_channels, out_channels, bias=bias_r)  # For root node features
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, mean_agg: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Transform aggregated neighbor features
        out = self.lin_l(mean_agg)
        # Add transformed self-node features
        out = out + self.lin_r(x)
        return out


class SAGEStage7_Activate(torch.nn.Module):
    """
    Stage 7: ACTIVATE - ReLU activation
    Input: out [num_nodes, out_dim]
    Output: activated [num_nodes, out_dim]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)
