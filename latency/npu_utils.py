"""
NPU Utility Functions for Latency Testing

Helper functions for NPU inference with static shape models.
Includes padding/unpadding for graph data (numpy-based for OpenVINO).

NPU models require fixed input shapes. When actual data is smaller than
the static shape, we pad inputs before inference and unpad outputs after.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class NPUPaddingInfo:
    """Track padding information for unpadding outputs."""
    original_num_nodes: int
    original_num_edges: int
    padded_num_nodes: int
    padded_num_edges: int


def pad_array_for_npu(
    array: np.ndarray,
    target_size: int,
    dim: int = 0
) -> Tuple[np.ndarray, int]:
    """
    Pad a numpy array to a target size for NPU static model.

    Args:
        array: Input array to pad
        target_size: Target size for the specified dimension
        dim: Dimension to pad (default: 0)

    Returns:
        (padded_array, original_size)
    """
    original_size = array.shape[dim]

    if original_size >= target_size:
        # No padding needed, but may need to truncate if larger
        if original_size > target_size:
            if dim == 0:
                return array[:target_size], original_size
            elif dim == 1:
                return array[:, :target_size], original_size
        return array, original_size

    # Calculate padding amount
    padding_size = target_size - original_size

    # Create padding shape
    padding_shape = list(array.shape)
    padding_shape[dim] = padding_size

    # Create zero padding with same dtype
    padding = np.zeros(padding_shape, dtype=array.dtype)

    # Concatenate along specified dimension
    padded_array = np.concatenate([array, padding], axis=dim)

    return padded_array, original_size


def unpad_array_from_npu(
    array: np.ndarray,
    original_size: int,
    dim: int = 0
) -> np.ndarray:
    """
    Remove padding from NPU output array.

    Args:
        array: Padded array from NPU
        original_size: Original size before padding
        dim: Dimension that was padded (default: 0)

    Returns:
        Unpadded array
    """
    if dim == 0:
        return array[:original_size]
    elif dim == 1:
        return array[:, :original_size]
    else:
        raise NotImplementedError(f"Unpadding for dim={dim} not implemented")


def pad_graph_data_for_npu(
    x: np.ndarray,
    edge_index: np.ndarray,
    target_num_nodes: int,
    target_num_edges: int
) -> Tuple[np.ndarray, np.ndarray, NPUPaddingInfo]:
    """
    Pad graph data (features and edges) for NPU static model.

    Args:
        x: Node features [num_nodes, feat_dim], dtype float32
        edge_index: Edge indices [2, num_edges], dtype int64
        target_num_nodes: Target number of nodes (static shape)
        target_num_edges: Target number of edges (static shape)

    Returns:
        (x_padded, edge_index_padded, padding_info)
    """
    original_num_nodes = x.shape[0]
    original_num_edges = edge_index.shape[1]

    # Pad features (dim 0)
    x_padded, _ = pad_array_for_npu(x, target_num_nodes, dim=0)

    # Pad edges (dim 1) - use zeros which point to node 0
    # This is safe because padding nodes have zero features
    if original_num_edges < target_num_edges:
        padding_edges = np.zeros(
            (2, target_num_edges - original_num_edges),
            dtype=edge_index.dtype
        )
        edge_index_padded = np.concatenate([edge_index, padding_edges], axis=1)
    else:
        # Truncate if necessary
        edge_index_padded = edge_index[:, :target_num_edges]

    padding_info = NPUPaddingInfo(
        original_num_nodes=original_num_nodes,
        original_num_edges=original_num_edges,
        padded_num_nodes=target_num_nodes,
        padded_num_edges=target_num_edges
    )

    return x_padded, edge_index_padded, padding_info


def unpad_graph_output_from_npu(
    output: np.ndarray,
    padding_info: NPUPaddingInfo
) -> np.ndarray:
    """
    Remove padding from NPU output.

    Args:
        output: Output array from NPU [padded_num_nodes, out_dim]
        padding_info: Padding information from pad_graph_data_for_npu

    Returns:
        Unpadded output [original_num_nodes, out_dim]
    """
    return unpad_array_from_npu(output, padding_info.original_num_nodes, dim=0)


def pad_intermediate_for_npu(
    data: np.ndarray,
    target_num_nodes: int,
    dim: int = 0
) -> Tuple[np.ndarray, int]:
    """
    Pad intermediate stage output for NPU input.

    Used when previous block outputs fewer nodes than NPU static shape.

    Args:
        data: Intermediate data (e.g., aggregated features)
        target_num_nodes: NPU static node count
        dim: Dimension to pad

    Returns:
        (padded_data, original_size)
    """
    return pad_array_for_npu(data, target_num_nodes, dim=dim)


def needs_npu_padding(
    actual_nodes: int,
    actual_edges: int,
    static_nodes: int,
    static_edges: int
) -> bool:
    """
    Check if padding is needed for NPU inference.

    Args:
        actual_nodes: Actual number of nodes
        actual_edges: Actual number of edges
        static_nodes: NPU model's static node count
        static_edges: NPU model's static edge count

    Returns:
        True if any dimension needs padding
    """
    return actual_nodes != static_nodes or actual_edges != static_edges


def prepare_npu_inputs(
    inputs: Dict[str, np.ndarray],
    static_num_nodes: int,
    static_num_edges: int
) -> Tuple[Dict[str, np.ndarray], Optional[NPUPaddingInfo]]:
    """
    Prepare inputs for NPU by padding to static shape.

    Handles common input formats:
    - {'x': features, 'edge_index': edges}  (Stage 1-5)
    - {'mean_agg': agg, 'x': features}      (Stage 6-7)
    - Single tensor input

    Args:
        inputs: Input dictionary
        static_num_nodes: NPU static node count
        static_num_edges: NPU static edge count

    Returns:
        (padded_inputs, padding_info)
    """
    padded_inputs = {}
    padding_info = None

    # Detect input format
    has_edge_index = 'edge_index' in inputs
    has_x = 'x' in inputs

    if has_edge_index and has_x:
        # Full graph input (Stage 1-5)
        x = inputs['x']
        edge_index = inputs['edge_index']

        x_padded, edge_index_padded, padding_info = pad_graph_data_for_npu(
            x, edge_index, static_num_nodes, static_num_edges
        )

        padded_inputs['x'] = x_padded
        padded_inputs['edge_index'] = edge_index_padded

        # Copy any other inputs unchanged
        for key, value in inputs.items():
            if key not in ['x', 'edge_index']:
                padded_inputs[key] = value

    elif has_x and 'mean_agg' in inputs:
        # Stage 6-7 input format
        x = inputs['x']
        mean_agg = inputs['mean_agg']

        original_num_nodes = x.shape[0]

        x_padded, _ = pad_array_for_npu(x, static_num_nodes, dim=0)
        mean_agg_padded, _ = pad_array_for_npu(mean_agg, static_num_nodes, dim=0)

        padded_inputs['x'] = x_padded
        padded_inputs['mean_agg'] = mean_agg_padded

        padding_info = NPUPaddingInfo(
            original_num_nodes=original_num_nodes,
            original_num_edges=0,  # Not applicable
            padded_num_nodes=static_num_nodes,
            padded_num_edges=0
        )

    else:
        # Generic case: pad all arrays in dim 0 by node count
        first_key = list(inputs.keys())[0]
        original_size = inputs[first_key].shape[0]

        for key, value in inputs.items():
            if isinstance(value, np.ndarray):
                if value.ndim >= 1 and value.shape[0] == original_size:
                    # Likely a node-indexed tensor
                    padded, _ = pad_array_for_npu(value, static_num_nodes, dim=0)
                    padded_inputs[key] = padded
                else:
                    padded_inputs[key] = value
            else:
                padded_inputs[key] = value

        padding_info = NPUPaddingInfo(
            original_num_nodes=original_size,
            original_num_edges=0,
            padded_num_nodes=static_num_nodes,
            padded_num_edges=0
        )

    return padded_inputs, padding_info


def unpad_npu_outputs(
    outputs: Dict[str, np.ndarray],
    padding_info: NPUPaddingInfo
) -> Dict[str, np.ndarray]:
    """
    Remove padding from all NPU outputs.

    Args:
        outputs: Output dictionary from NPU
        padding_info: Padding information

    Returns:
        Unpadded outputs
    """
    unpadded_outputs = {}

    for key, value in outputs.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            # Check if first dim matches padded node count
            if value.shape[0] == padding_info.padded_num_nodes:
                unpadded_outputs[key] = unpad_array_from_npu(
                    value, padding_info.original_num_nodes, dim=0
                )
            else:
                unpadded_outputs[key] = value
        else:
            unpadded_outputs[key] = value

    return unpadded_outputs


if __name__ == "__main__":
    # Test padding utilities
    print("Testing NPU padding utilities...")

    # Test data
    num_nodes = 1000
    num_edges = 5000
    feat_dim = 500

    x = np.random.randn(num_nodes, feat_dim).astype(np.float32)
    edge_index = np.random.randint(0, num_nodes, (2, num_edges)).astype(np.int64)

    # Static shape (larger than actual)
    static_nodes = 1500
    static_edges = 8000

    # Test pad_graph_data_for_npu
    x_padded, edge_padded, info = pad_graph_data_for_npu(
        x, edge_index, static_nodes, static_edges
    )

    print(f"Original: x={x.shape}, edge_index={edge_index.shape}")
    print(f"Padded: x={x_padded.shape}, edge_index={edge_padded.shape}")
    print(f"Padding info: {info}")

    # Test unpadding
    output = np.random.randn(static_nodes, 256).astype(np.float32)
    output_unpadded = unpad_graph_output_from_npu(output, info)
    print(f"Output: {output.shape} -> {output_unpadded.shape}")

    # Verify data integrity
    assert np.allclose(x, x_padded[:num_nodes])
    assert np.allclose(edge_index, edge_padded[:, :num_edges])
    assert output_unpadded.shape[0] == num_nodes

    print("\nAll tests passed!")
