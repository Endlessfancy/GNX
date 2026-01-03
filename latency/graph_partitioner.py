"""
Graph Partitioner with 1-hop Halo Node Expansion

Provides graph partitioning for data parallel GNN execution.
Each partition contains:
- owned_nodes: Primary nodes assigned to this partition
- halo_nodes: 1-hop neighbors from other partitions (read-only)
- local_edge_index: Edges remapped to local node IDs

This enables correct GNN aggregation while splitting data across devices.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PartitionData:
    """Data for a single graph partition."""
    partition_id: int

    # Node data
    x_local: np.ndarray           # [num_local_nodes, feat_dim] - owned + halo features
    num_owned: int                 # Number of owned nodes (output dimension)
    num_halo: int                  # Number of halo nodes

    # Edge data
    edge_index_local: np.ndarray  # [2, num_local_edges] - remapped edge indices
    num_edges: int

    # Mapping info
    owned_mask: np.ndarray        # [num_local_nodes] bool - True for owned nodes
    global_to_local: Dict[int, int]  # Global node ID -> local node ID
    local_to_global: np.ndarray   # [num_local_nodes] - local ID -> global ID


class HaloPartitioner:
    """
    1-hop Halo Node Graph Partitioner.

    Splits a graph into N partitions where each partition contains:
    1. Owned nodes: Linearly assigned nodes [start, end)
    2. Halo nodes: 1-hop neighbors of owned nodes from other partitions

    This ensures correct message passing at partition boundaries.
    """

    def __init__(self, num_partitions: int = 2, ratios: Optional[List[float]] = None):
        """
        Initialize partitioner.

        Args:
            num_partitions: Number of partitions to create
            ratios: Partition size ratios (default: equal split)
        """
        self.num_partitions = num_partitions

        if ratios is None:
            self.ratios = [1.0 / num_partitions] * num_partitions
        else:
            # Normalize ratios
            total = sum(ratios)
            self.ratios = [r / total for r in ratios]

    def partition(self,
                  x: np.ndarray,
                  edge_index: np.ndarray) -> List[PartitionData]:
        """
        Partition graph into subgraphs with halo expansion.

        Args:
            x: Node features [num_nodes, feat_dim]
            edge_index: Edge indices [2, num_edges] (source, target)

        Returns:
            List of PartitionData, one per partition
        """
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        # Step 1: Assign owned nodes to each partition
        owned_ranges = self._compute_owned_ranges(num_nodes)

        # Step 2: Build partition data with halo expansion
        partitions = []
        for pid in range(self.num_partitions):
            partition = self._build_partition(
                pid, x, edge_index, owned_ranges
            )
            partitions.append(partition)

        return partitions

    def _compute_owned_ranges(self, num_nodes: int) -> List[Tuple[int, int]]:
        """Compute (start, end) range for each partition's owned nodes."""
        ranges = []
        start = 0

        for i, ratio in enumerate(self.ratios):
            if i == self.num_partitions - 1:
                # Last partition gets remaining nodes
                end = num_nodes
            else:
                end = start + int(num_nodes * ratio)
            ranges.append((start, end))
            start = end

        return ranges

    def _build_partition(self,
                         partition_id: int,
                         x: np.ndarray,
                         edge_index: np.ndarray,
                         owned_ranges: List[Tuple[int, int]]) -> PartitionData:
        """Build a single partition with halo expansion."""

        start, end = owned_ranges[partition_id]
        owned_nodes = set(range(start, end))

        # Find halo nodes: sources of edges whose targets are owned but sources are not
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        halo_nodes = set()
        local_edge_src = []
        local_edge_dst = []

        for i in range(edge_index.shape[1]):
            src = int(src_nodes[i])
            dst = int(dst_nodes[i])

            # Edge is relevant if target is owned
            if dst in owned_nodes:
                local_edge_src.append(src)
                local_edge_dst.append(dst)

                # Source is halo if not owned
                if src not in owned_nodes:
                    halo_nodes.add(src)

        # Create node ordering: owned first, then halo
        owned_list = list(range(start, end))  # Ordered owned nodes
        halo_list = sorted(list(halo_nodes))  # Sorted halo nodes
        all_local_nodes = owned_list + halo_list

        # Build mapping
        global_to_local = {g: l for l, g in enumerate(all_local_nodes)}
        local_to_global = np.array(all_local_nodes, dtype=np.int64)

        # Remap edge indices to local IDs
        local_edge_index = np.zeros((2, len(local_edge_src)), dtype=np.int64)
        for i, (src, dst) in enumerate(zip(local_edge_src, local_edge_dst)):
            local_edge_index[0, i] = global_to_local[src]
            local_edge_index[1, i] = global_to_local[dst]

        # Extract local features
        x_local = x[all_local_nodes]

        # Create owned mask
        num_owned = len(owned_list)
        num_halo = len(halo_list)
        num_local = num_owned + num_halo
        owned_mask = np.zeros(num_local, dtype=bool)
        owned_mask[:num_owned] = True

        return PartitionData(
            partition_id=partition_id,
            x_local=x_local,
            num_owned=num_owned,
            num_halo=num_halo,
            edge_index_local=local_edge_index,
            num_edges=local_edge_index.shape[1],
            owned_mask=owned_mask,
            global_to_local=global_to_local,
            local_to_global=local_to_global
        )

    def merge_outputs(self,
                      outputs: List[np.ndarray],
                      partitions: List[PartitionData],
                      total_nodes: int) -> np.ndarray:
        """
        Merge partition outputs back to global ordering.

        Args:
            outputs: List of output arrays from each partition
            partitions: Partition metadata
            total_nodes: Total number of nodes in original graph

        Returns:
            Merged output [total_nodes, out_dim]
        """
        if len(outputs) == 0:
            raise ValueError("No outputs to merge")

        out_dim = outputs[0].shape[1] if len(outputs[0].shape) > 1 else 1
        merged = np.zeros((total_nodes, out_dim), dtype=outputs[0].dtype)

        for out, part in zip(outputs, partitions):
            # Only take owned nodes' outputs (first num_owned rows)
            owned_output = out[:part.num_owned]

            # Map back to global indices
            for local_id in range(part.num_owned):
                global_id = int(part.local_to_global[local_id])
                merged[global_id] = owned_output[local_id]

        return merged


def test_halo_partitioner():
    """Test HaloPartitioner with a simple graph."""
    print("Testing HaloPartitioner...")
    print("=" * 60)

    # Create a simple graph
    # Nodes: 0, 1, 2, 3, 4, 5 (6 nodes)
    # Edges: 0->1, 1->2, 2->3, 3->4, 4->5, 0->2, 1->3, 2->4, 3->5
    num_nodes = 6
    x = np.random.randn(num_nodes, 8).astype(np.float32)
    edge_index = np.array([
        [0, 1, 2, 3, 4, 0, 1, 2, 3],  # src
        [1, 2, 3, 4, 5, 2, 3, 4, 5]   # dst
    ], dtype=np.int64)

    print(f"Original graph: {num_nodes} nodes, {edge_index.shape[1]} edges")
    print(f"Edge list: {list(zip(edge_index[0], edge_index[1]))}")

    # Partition into 2 parts
    partitioner = HaloPartitioner(num_partitions=2)
    partitions = partitioner.partition(x, edge_index)

    for p in partitions:
        print(f"\nPartition {p.partition_id}:")
        print(f"  Owned nodes: {p.num_owned}")
        print(f"  Halo nodes: {p.num_halo}")
        print(f"  Total local nodes: {p.num_owned + p.num_halo}")
        print(f"  Local edges: {p.num_edges}")
        print(f"  x_local shape: {p.x_local.shape}")
        print(f"  Local node IDs (global): {list(p.local_to_global)}")
        print(f"  Owned mask: {p.owned_mask}")

    # Test merge
    fake_outputs = [
        np.arange(p.num_owned + p.num_halo).reshape(-1, 1).astype(np.float32)
        for p in partitions
    ]
    merged = partitioner.merge_outputs(fake_outputs, partitions, num_nodes)
    print(f"\nMerged output shape: {merged.shape}")
    print(f"Merged values: {merged.flatten()}")

    print("\n" + "=" * 60)
    print("HaloPartitioner test passed!")


if __name__ == "__main__":
    test_halo_partitioner()
