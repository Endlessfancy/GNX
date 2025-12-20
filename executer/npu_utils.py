"""
NPU Utility Functions

Helper functions for NPU inference, including padding/unpadding for static models.
"""

import torch
from typing import Tuple, Optional


def pad_tensor_for_npu(tensor: torch.Tensor, target_size: int, dim: int = 0) -> Tuple[torch.Tensor, int]:
    """
    Pad a tensor to a target size for NPU static model.

    Args:
        tensor: Input tensor to pad
        target_size: Target size for the specified dimension
        dim: Dimension to pad (default: 0)

    Returns:
        (padded_tensor, original_size)
    """
    original_size = tensor.size(dim)

    if original_size >= target_size:
        # No padding needed
        return tensor, original_size

    # Calculate padding amount
    padding_size = target_size - original_size

    # Create padding shape
    padding_shape = list(tensor.shape)
    padding_shape[dim] = padding_size

    # Create zero padding
    padding = torch.zeros(*padding_shape, dtype=tensor.dtype, device=tensor.device)

    # Concatenate along specified dimension
    if dim == 0:
        padded_tensor = torch.cat([tensor, padding], dim=0)
    elif dim == 1:
        padded_tensor = torch.cat([tensor, padding], dim=1)
    else:
        raise NotImplementedError(f"Padding for dim={dim} not implemented")

    return padded_tensor, original_size


def unpad_tensor_from_npu(tensor: torch.Tensor, original_size: int, dim: int = 0) -> torch.Tensor:
    """
    Remove padding from NPU output tensor.

    Args:
        tensor: Padded tensor from NPU
        original_size: Original size before padding
        dim: Dimension that was padded (default: 0)

    Returns:
        Unpadded tensor
    """
    if dim == 0:
        return tensor[:original_size]
    elif dim == 1:
        return tensor[:, :original_size]
    else:
        raise NotImplementedError(f"Unpadding for dim={dim} not implemented")


def get_npu_static_size(model_info: dict, key: str = 'static_num_nodes') -> Optional[int]:
    """
    Get the static size requirement from NPU model info.

    Args:
        model_info: Model metadata dictionary
        key: Key to look up ('static_num_nodes', 'static_num_edges', etc.)

    Returns:
        Static size if specified, None otherwise
    """
    return model_info.get(key, None)


def needs_npu_padding(actual_size: int, npu_static_size: Optional[int]) -> bool:
    """
    Check if padding is needed for NPU inference.

    Args:
        actual_size: Actual data size
        npu_static_size: NPU model's static size requirement

    Returns:
        True if padding is needed
    """
    if npu_static_size is None:
        return False

    return actual_size < npu_static_size


def pad_graph_data_for_npu(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    target_num_nodes: int,
    target_num_edges: int
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Pad graph data (features and edges) for NPU static model.

    Args:
        x: Node features [num_nodes, feat_dim]
        edge_index: Edge indices [2, num_edges]
        target_num_nodes: Target number of nodes
        target_num_edges: Target number of edges

    Returns:
        (x_padded, edge_index_padded, original_num_nodes, original_num_edges)
    """
    original_num_nodes = x.size(0)
    original_num_edges = edge_index.size(1)

    # Pad features
    x_padded, _ = pad_tensor_for_npu(x, target_num_nodes, dim=0)

    # Pad edges
    if original_num_edges < target_num_edges:
        padding_edges = torch.zeros(2, target_num_edges - original_num_edges,
                                    dtype=edge_index.dtype, device=edge_index.device)
        edge_index_padded = torch.cat([edge_index, padding_edges], dim=1)
    else:
        edge_index_padded = edge_index

    return x_padded, edge_index_padded, original_num_nodes, original_num_edges
