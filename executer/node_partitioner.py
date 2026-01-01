"""
Node-Based Graph Partitioner for Data Parallelism

This module provides node-based partitioning for GNN inference with heterogeneous devices.
Key features:
- Partition by destination nodes (each device owns a subset of nodes)
- No synchronization needed between stages (sum_agg and count naturally match)
- NPU padding support (pad to full graph size for static models)
- Full graph features (x_full) support for correct GNN execution
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


class NodeBasedPartitioner:
    """
    Node-based graph partitioner for data parallelism.

    Key principle: Partition by destination nodes.
    - Each device owns a subset of nodes
    - Edges assigned to device that owns the destination node
    - Ensures sum_agg and count from same set of edges (no sync needed)
    """

    @staticmethod
    def partition_by_nodes(
        edge_index: torch.Tensor,
        num_nodes: int,
        split_ratios: List[float],
        x: Optional[torch.Tensor] = None
    ) -> List[Dict]:
        """
        Partition graph by destination nodes.

        【阶段2优化】向量化预计算所有partition边界，减少GIL持有时间

        Args:
            edge_index: [2, num_edges] - edge connectivity
            num_nodes: Total number of nodes
            split_ratios: List of ratios for each partition (e.g., [0.3, 0.7])
            x: Optional [num_nodes, feat_dim] - node features

        Returns:
            List of partition dictionaries, each containing:
            - 'node_range': (start_node, end_node) - nodes owned by this device
            - 'edge_index': Edge indices for edges with dest in node_range
            - 'edge_mask': Boolean mask for edges in this partition
            - 'num_edges': Number of edges in this partition
            - 'num_nodes': Number of nodes owned by this device
            - 'x_full': Full graph features (for Stage 1 Gather)
            - 'x_owned': Features for owned nodes only (for Stage 6 Transform)
        """
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)

        num_edges = edge_index.size(1)
        dest_nodes = edge_index[1]  # 【阶段2优化】提前提取，避免重复索引

        # 【阶段2优化】向量化预计算所有partition的node ranges
        node_boundaries = NodeBasedPartitioner._compute_node_boundaries(num_nodes, split_ratios)

        # 【阶段2优化】预分配partitions列表
        partitions = []

        # 遍历每个partition（通常只有2-3个，循环开销较小）
        for i, (start_node, end_node) in enumerate(node_boundaries):
            partition_nodes = end_node - start_node

            # 【阶段2优化】向量化的edge filtering（已经是tensor操作，释放GIL）
            edge_mask = (dest_nodes >= start_node) & (dest_nodes < end_node)
            partition_edges = edge_index[:, edge_mask]
            partition_num_edges = partition_edges.size(1)

            # Prepare partition info
            partition_info = {
                'node_range': (start_node, end_node),
                'edge_index': partition_edges,
                'edge_mask': edge_mask,
                'num_edges': partition_num_edges,
                'num_nodes': partition_nodes,
                'total_num_nodes': num_nodes,
                'node_indices': torch.arange(start_node, end_node, dtype=torch.long)
            }

            # Add node features if provided
            if x is not None:
                partition_info['x_full'] = x  # Full graph features (for Gather stage)
                partition_info['x_owned'] = x[start_node:end_node]  # Owned nodes only (for Transform stage)

            # Statistics - 【阶段2优化】torch.unique释放GIL，保留原逻辑
            if partition_num_edges > 0:
                src_nodes = partition_edges[0]
                unique_src_nodes = torch.unique(src_nodes)
                partition_info['num_unique_src_nodes'] = unique_src_nodes.size(0)
            else:
                partition_info['num_unique_src_nodes'] = 0

            partitions.append(partition_info)

        return partitions

    @staticmethod
    def _compute_node_boundaries(num_nodes: int, split_ratios: List[float]) -> List[Tuple[int, int]]:
        """
        【阶段2优化】向量化计算所有partition的node boundaries

        预先计算所有partition的[start, end)范围，避免循环中重复计算

        Args:
            num_nodes: 总节点数
            split_ratios: 分割比例列表

        Returns:
            [(start_0, end_0), (start_1, end_1), ...] 每个partition的节点范围
        """
        # 向量化计算每个partition的节点数
        partition_sizes = [int(num_nodes * ratio) for ratio in split_ratios]

        # 调整最后一个partition以包含剩余节点
        partition_sizes[-1] = num_nodes - sum(partition_sizes[:-1])

        # 计算累积边界
        boundaries = []
        start_node = 0
        for size in partition_sizes:
            end_node = start_node + size
            boundaries.append((start_node, end_node))
            start_node = end_node

        return boundaries

    @staticmethod
    def apply_npu_padding(
        partition: Dict,
        full_num_nodes: int,
        full_num_edges: int,
        full_feat_dim: Optional[int] = None
    ) -> Dict:
        """
        Apply padding to a partition for NPU static model inference.

        Strategy: Pad to full graph size (not multi-tier).
        - Pad edge_index to full_num_edges
        - Pad node features to full_num_nodes
        - Padded edges point to node 0 (with zero messages)
        - Padded node features are zeros

        Args:
            partition: Partition dictionary from partition_by_nodes()
            full_num_nodes: Original graph's number of nodes
            full_num_edges: Original graph's number of edges
            full_feat_dim: Feature dimension (required if padding features)

        Returns:
            Padded partition dictionary with additional fields:
            - 'edge_index_padded': [2, full_num_edges]
            - 'x_full_padded': [full_num_nodes, feat_dim] (if x exists)
            - 'padding_info': Details about padding applied
        """
        partition_num_edges = partition['num_edges']
        partition_num_nodes = partition['num_nodes']

        # Calculate padding amounts
        edge_padding = full_num_edges - partition_num_edges
        node_padding = full_num_nodes - partition_num_nodes

        if edge_padding < 0 or node_padding < 0:
            raise ValueError(f"Partition larger than full graph! "
                           f"Partition: {partition_num_edges} edges, {partition_num_nodes} nodes; "
                           f"Full: {full_num_edges} edges, {full_num_nodes} nodes")

        # Pad edge_index
        edge_index = partition['edge_index']
        if edge_padding > 0:
            # Pad with edges pointing to node 0 (dummy edges)
            padding_edges = torch.zeros(2, edge_padding, dtype=torch.long)
            edge_index_padded = torch.cat([edge_index, padding_edges], dim=1)
        else:
            edge_index_padded = edge_index

        padded_partition = partition.copy()
        padded_partition['edge_index_padded'] = edge_index_padded

        # Pad node features if present
        # Note: x_full is already full-sized, no need to pad
        # But x_owned needs padding
        if 'x_owned' in partition and partition['x_owned'] is not None:
            x = partition['x_owned']
            if full_feat_dim is None:
                full_feat_dim = x.size(1)

            if node_padding > 0:
                # Pad with zero features
                padding_features = torch.zeros(node_padding, full_feat_dim,
                                             dtype=x.dtype, device=x.device)
                x_owned_padded = torch.cat([x, padding_features], dim=0)
            else:
                x_owned_padded = x

            padded_partition['x_owned_padded'] = x_owned_padded

        # Store padding metadata
        padded_partition['padding_info'] = {
            'original_num_edges': partition_num_edges,
            'original_num_nodes': partition_num_nodes,
            'padded_num_edges': full_num_edges,
            'padded_num_nodes': full_num_nodes,
            'edge_padding_count': edge_padding,
            'node_padding_count': node_padding,
            'edge_padding_ratio': edge_padding / full_num_edges if full_num_edges > 0 else 0,
            'node_padding_ratio': node_padding / full_num_nodes if full_num_nodes > 0 else 0
        }

        return padded_partition

    @staticmethod
    def partition_and_pad(
        edge_index: torch.Tensor,
        num_nodes: int,
        split_ratios: List[float],
        device_types: List[str],
        x: Optional[torch.Tensor] = None
    ) -> List[Dict]:
        """
        Complete workflow: partition by nodes and apply padding for NPU devices.

        Args:
            edge_index: [2, num_edges]
            num_nodes: Total number of nodes
            split_ratios: Partition ratios
            device_types: List of device types (e.g., ['NPU', 'GPU'])
            x: Optional node features

        Returns:
            List of partition dictionaries (with padding applied to NPU partitions)
        """
        # Step 1: Partition by nodes
        partitions = NodeBasedPartitioner.partition_by_nodes(
            edge_index, num_nodes, split_ratios, x
        )

        # Step 2: Apply padding to NPU partitions
        full_num_edges = edge_index.size(1)
        full_feat_dim = x.size(1) if x is not None else None

        padded_partitions = []
        for partition, device_type in zip(partitions, device_types):
            if device_type.upper() == 'NPU':
                # Apply padding for NPU
                padded_partition = NodeBasedPartitioner.apply_npu_padding(
                    partition, num_nodes, full_num_edges, full_feat_dim
                )
                padded_partitions.append(padded_partition)
            else:
                # No padding for CPU/GPU
                padded_partitions.append(partition)

        return padded_partitions

    @staticmethod
    def merge_outputs(outputs: List[torch.Tensor], partitions: List[Dict]) -> torch.Tensor:
        """
        Merge outputs from node-based partitions.

        For node-based partitioning, each device owns disjoint nodes.
        This function correctly handles:
        - Full-sized outputs (when model uses x_full)
        - Partition-sized outputs
        - NPU padded outputs

        Args:
            outputs: List of output tensors from each partition
            partitions: List of partition info

        Returns:
            Merged output tensor [total_num_nodes, ...]
        """
        actual_outputs = []

        for output, partition in zip(outputs, partitions):
            node_range = partition['node_range']
            start_node, end_node = node_range
            partition_num_nodes = partition['num_nodes']
            total_num_nodes = partition.get('total_num_nodes', output.size(0))

            # Case 1: Output is full-sized (from using x_full in Gather)
            if output.size(0) == total_num_nodes and total_num_nodes > partition_num_nodes:
                # Extract only owned nodes
                actual_output = output[start_node:end_node]

            # Case 2: NPU output with padding
            elif 'padding_info' in partition:
                # Extract actual nodes (remove padding)
                actual_num_nodes = partition['padding_info']['original_num_nodes']
                actual_output = output[:actual_num_nodes]

            # Case 3: Output is already partition-sized
            else:
                actual_output = output

            actual_outputs.append(actual_output)

        # Concatenate in node_range order
        merged = torch.cat(actual_outputs, dim=0)
        return merged
