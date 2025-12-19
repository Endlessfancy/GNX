"""
Model Export Utilities for Executer

This module provides standalone model export functionality without external dependencies.
Simplified from executor copy/pep_model_exporter.py for self-contained operation.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch_geometric.nn.dense.linear import Linear
from torch_scatter import scatter_add


# ====================================================================
# GraphSAGE 7-Stage Modules
# ====================================================================

class SAGEStage1_Gather(torch.nn.Module):
    """
    Stage 1: GATHER - Neighbor feature gathering
    Input: x [num_nodes, feat_dim], edge_index [2, num_edges]
    Output: x_j [num_edges, feat_dim]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return x[edge_index[0]]


class SAGEStage2_Message(torch.nn.Module):
    """
    Stage 2: MESSAGE - Message computation
    Input: x_j [num_edges, feat_dim]
    Output: messages [num_edges, feat_dim]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


class SAGEStage3_ReduceSum(torch.nn.Module):
    """
    Stage 3: REDUCE_SUM - Sum aggregation
    Input: messages [num_edges, feat_dim], edge_index [2, num_edges], num_nodes
    Output: sum_agg [num_nodes, feat_dim]
    """
    def __init__(self, num_nodes_static=None):
        super().__init__()
        self.num_nodes_static = num_nodes_static

    def forward(self, messages: torch.Tensor, edge_index: torch.Tensor, num_nodes: int = None) -> torch.Tensor:
        actual_num_nodes = self.num_nodes_static if self.num_nodes_static is not None else num_nodes
        out = torch.zeros(actual_num_nodes, messages.size(1), dtype=messages.dtype, device=messages.device)
        target_nodes = edge_index[1]
        out.index_add_(0, target_nodes, messages)
        return out


class SAGEStage4_ReduceCount(torch.nn.Module):
    """
    Stage 4: REDUCE_COUNT - Count neighbors
    Input: edge_index [2, num_edges], num_nodes, num_edges
    Output: count [num_nodes]
    """
    def __init__(self):
        super().__init__()

    def forward(self, edge_index: torch.Tensor, num_nodes: int, num_edges: int) -> torch.Tensor:
        target_nodes = edge_index[1]
        ones = torch.ones(num_edges, dtype=torch.float32, device=edge_index.device)
        count = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
        count = scatter_add(ones, target_nodes, dim=0, out=count)
        return count


class SAGEStage5_Normalize(torch.nn.Module):
    """
    Stage 5: NORMALIZE - Compute mean
    Input: sum_agg [num_nodes, feat_dim], count [num_nodes]
    Output: mean_agg [num_nodes, feat_dim]
    """
    def __init__(self):
        super().__init__()

    def forward(self, sum_agg: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        count = torch.clamp(count, min=1)
        mean_agg = sum_agg / count.unsqueeze(-1)
        return mean_agg


class SAGEStage6_Transform(torch.nn.Module):
    """
    Stage 6: TRANSFORM - Linear transformations
    Input: mean_agg [num_nodes, in_dim], x [num_nodes, in_dim]
    Output: out [num_nodes, out_dim]
    """
    def __init__(self, in_channels: int, out_channels: int, bias_l=True, bias_r=False):
        super().__init__()
        self.lin_l = Linear(in_channels, out_channels, bias=bias_l)
        self.lin_r = Linear(in_channels, out_channels, bias=bias_r)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, mean_agg: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.lin_l(mean_agg)
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


# ====================================================================
# Combined Stage Model
# ====================================================================

class CombinedStagesModel(nn.Module):
    """
    Combined multi-stage model for sequential execution.
    Simplified version without complex dependency analysis.
    """

    def __init__(self, stages: List[nn.Module], stage_indices: List[int]):
        """
        Args:
            stages: List of stage modules
            stage_indices: Which stages are combined (e.g., [1, 2, 3])
        """
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.stage_indices = stage_indices
        self.name = f"stages_{'_'.join(map(str, stage_indices))}"

    def forward(self, *args):
        """Execute stages sequentially"""
        # For simplicity, we execute based on first stage
        first_stage = self.stage_indices[0]

        if first_stage == 1:
            # Full pipeline: stages 1-7
            return self._execute_full_pipeline(args)
        else:
            raise NotImplementedError(f"Starting from stage {first_stage} not yet supported in simplified version")

    def _execute_full_pipeline(self, args):
        """Execute full 7-stage pipeline"""
        x, edge_index = args[0], args[1]
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # Stage 1: Gather
        if 1 in self.stage_indices:
            x_j = self.stages[self.stage_indices.index(1)](x, edge_index)
        else:
            return x  # Not executing stage 1

        # Stage 2: Message
        if 2 in self.stage_indices:
            messages = self.stages[self.stage_indices.index(2)](x_j)
        else:
            messages = x_j

        # Stage 3: ReduceSum
        if 3 in self.stage_indices:
            sum_agg = self.stages[self.stage_indices.index(3)](messages, edge_index, num_nodes)
        else:
            return messages

        # Stage 4: ReduceCount
        if 4 in self.stage_indices:
            count = self.stages[self.stage_indices.index(4)](edge_index, num_nodes, num_edges)
        else:
            return sum_agg

        # Stage 5: Normalize
        if 5 in self.stage_indices:
            mean_agg = self.stages[self.stage_indices.index(5)](sum_agg, count)
        else:
            return count

        # Stage 6: Transform
        if 6 in self.stage_indices:
            transformed = self.stages[self.stage_indices.index(6)](mean_agg, x)
        else:
            return mean_agg

        # Stage 7: Activate
        if 7 in self.stage_indices:
            output = self.stages[self.stage_indices.index(7)](transformed)
            return output
        else:
            return transformed


# ====================================================================
# Model Exporter
# ====================================================================

class SimpleModelExporter:
    """Simplified model exporter for basic ONNX export"""

    def __init__(self):
        self.stage_modules = None

    def initialize_stages(self, in_dim: int = 500, hid_dim: int = 500, out_dim: int = 256):
        """Initialize all 7 stage modules"""
        self.stage_modules = [
            SAGEStage1_Gather(),
            SAGEStage2_Message(),
            SAGEStage3_ReduceSum(),
            SAGEStage4_ReduceCount(),
            SAGEStage5_Normalize(),
            SAGEStage6_Transform(in_dim, out_dim),  # Fixed: only pass in_channels and out_channels
            SAGEStage7_Activate()
        ]

    def export_combined_model(self, device: str, stages: List[int],
                            output_path: str, num_nodes: int, num_edges: int,
                            num_features: int, dynamic: bool = True):
        """
        Export a combined model for given stages

        Args:
            device: Device name (CPU/GPU/NPU)
            stages: List of stage indices (e.g., [1, 2, 3, 4, 5, 6, 7])
            output_path: Path to save ONNX model
            num_nodes: Number of nodes in graph
            num_edges: Number of edges in graph
            num_features: Number of input features
            dynamic: Whether to export dynamic model
        """
        # Initialize stages if not done
        if self.stage_modules is None:
            self.initialize_stages(in_dim=num_features, hid_dim=num_features, out_dim=256)

        # Get stage modules
        stage_list = [self.stage_modules[s - 1] for s in stages]

        # Create combined model
        combined_model = CombinedStagesModel(stage_list, stages)
        combined_model.eval()

        # Generate dummy inputs
        dummy_x = torch.randn(num_nodes, num_features)
        dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges))
        dummy_inputs = (dummy_x, dummy_edge_index)

        # Export to ONNX
        print(f"  Exporting {device} model (stages {stages}) to {output_path}")

        # Set dynamic axes if needed
        if dynamic:
            dynamic_axes = {
                'x': {0: 'num_nodes'},
                'edge_index': {1: 'num_edges'},
                'output': {0: 'num_nodes'}
            }
        else:
            dynamic_axes = None

        # Export
        torch.onnx.export(
            combined_model,
            dummy_inputs,
            output_path,
            input_names=['x', 'edge_index'],  # Use meaningful names
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=18,
            do_constant_folding=False,
            verbose=False
        )

        # Verify model size
        file_size = os.path.getsize(output_path)
        if file_size < 200:
            raise RuntimeError(f"Exported model is too small ({file_size} bytes), export may have failed")

        print(f"  ✓ Model exported successfully ({file_size / 1024 / 1024:.2f} MB)")

        # Try to verify
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path)

            # Test inference
            onnx_inputs = {
                'input_0': dummy_x.numpy(),
                'input_1': dummy_edge_index.numpy()
            }
            onnx_output = session.run(None, onnx_inputs)[0]

            # PyTorch output
            with torch.no_grad():
                pytorch_output = combined_model(*dummy_inputs).numpy()

            max_error = np.max(np.abs(pytorch_output - onnx_output))
            print(f"  Verification max error: {max_error:.2e}")

        except ImportError:
            print(f"  ⚠ ONNX Runtime not available, skipping verification")
        except Exception as e:
            print(f"  ⚠ Verification failed: {e}")


# ====================================================================
# Convenience Functions
# ====================================================================

def export_model_for_pep_block(pep_block: Dict, output_path: str,
                               num_nodes: int, num_edges: int, num_features: int):
    """
    Export model for a PEP block

    Args:
        pep_block: PEP block dict with 'devices', 'stages', 'ratios'
        output_path: Path to save ONNX model
        num_nodes: Number of nodes
        num_edges: Number of edges
        num_features: Number of input features
    """
    exporter = SimpleModelExporter()

    devices = pep_block['devices']
    stages = pep_block['stages']

    # For simplicity, use first device
    device = devices[0] if isinstance(devices, list) else devices

    # Export model
    exporter.export_combined_model(
        device=device,
        stages=stages,
        output_path=output_path,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_features=num_features,
        dynamic=True
    )


def check_and_export_model(model_path: str, pep_block: Dict,
                          num_nodes: int, num_edges: int, num_features: int) -> bool:
    """
    Check if model exists and is valid, export if needed

    Returns:
        True if model is ready (existed or newly exported)
    """
    # Check if model exists and is not a placeholder
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        if size >= 200:  # Real model
            print(f"  Model exists: {model_path} ({size / 1024 / 1024:.2f} MB)")
            return True
        else:
            print(f"  Model is placeholder ({size} bytes), re-exporting...")
    else:
        print(f"  Model not found: {model_path}, exporting...")

    # Export model
    try:
        export_model_for_pep_block(pep_block, model_path, num_nodes, num_edges, num_features)
        return True
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False


# ====================================================================
# Example Usage
# ====================================================================

if __name__ == "__main__":
    print("Model Export Utils - Test")
    print("=" * 70)

    # Example: Export a model for stages 1-7
    exporter = SimpleModelExporter()
    exporter.export_combined_model(
        device="CPU",
        stages=[1, 2, 3, 4, 5, 6, 7],
        output_path="test_model.onnx",
        num_nodes=1000,
        num_edges=5000,
        num_features=500,
        dynamic=True
    )

    print("\n✓ Test completed!")
