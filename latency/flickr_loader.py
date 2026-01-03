"""
Flickr Data Loader for Latency Testing

Loads Flickr dataset and partitions into subgraphs for PEP testing.
Uses the same data loading logic as executer.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add parent directories to path for imports
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir / 'compiler'))
sys.path.insert(0, str(_parent_dir / 'executer'))

import torch
from utils.graph_loader import GraphLoader


class FlickrSubgraphLoader:
    """
    Load Flickr dataset and partition into subgraphs for latency testing.

    Features:
    - Loads Flickr (89,250 nodes, 899,756 edges, 500 features)
    - Partitions into configurable number of subgraphs
    - Identifies ghost nodes for each partition
    - Provides max size for NPU static shape
    """

    def __init__(self, num_subgraphs: int = 8, cache_dir: Path = Path('data')):
        """
        Initialize loader.

        Args:
            num_subgraphs: Number of subgraphs to partition into
            cache_dir: Directory for caching dataset
        """
        self.num_subgraphs = num_subgraphs

        print(f"Loading Flickr dataset...")

        # 1. Load Flickr dataset
        loader = GraphLoader(cache_dir=cache_dir)
        self.full_data = loader.load_flickr()

        self.num_nodes = self.full_data.num_nodes
        self.num_edges = self.full_data.num_edges
        self.feature_dim = self.full_data.x.shape[1]

        # 2. Create uniform partition config
        self.partition_config = self._create_partition_config()

        # 3. Build partition mapping
        self.partition_mapping = self._build_partition_mapping()

        # 4. Build ghost node mapping
        self.ghost_node_mapping = self._build_ghost_node_mapping()

        # 5. Calculate max sizes
        self._max_nodes, self._max_edges = self._calculate_max_sizes()

        print(f"Partitioned into {num_subgraphs} subgraphs:")
        print(f"  Max nodes per subgraph: {self._max_nodes}")
        print(f"  Max edges per subgraph: {self._max_edges}")

    def _create_partition_config(self) -> Dict:
        """
        Create uniform partition configuration.

        Returns:
            partition_config: {
                'k': num_subgraphs,
                'num_subgraphs': num_subgraphs,
                'subgraphs': [{'id': i, 'n': nodes_per_partition}, ...]
            }
        """
        nodes_per_partition = self.num_nodes // self.num_subgraphs
        remainder = self.num_nodes % self.num_subgraphs

        subgraphs = []
        for i in range(self.num_subgraphs):
            # Distribute remainder across first partitions
            n = nodes_per_partition + (1 if i < remainder else 0)
            subgraphs.append({
                'id': i,
                'n': n
            })

        return {
            'k': self.num_subgraphs,
            'num_subgraphs': self.num_subgraphs,
            'subgraphs': subgraphs
        }

    def _build_partition_mapping(self) -> Dict[int, torch.Tensor]:
        """
        Build mapping from subgraph ID to owned node IDs.

        Returns:
            {subgraph_id: tensor of node IDs}
        """
        partition_mapping = {}
        node_ptr = 0

        for sg_config in self.partition_config['subgraphs']:
            sg_id = sg_config['id']
            n = sg_config['n']

            # Assign node ID range: [node_ptr, node_ptr + n)
            partition_mapping[sg_id] = torch.arange(node_ptr, node_ptr + n)
            node_ptr += n

        return partition_mapping

    def _build_ghost_node_mapping(self) -> Dict[int, Dict]:
        """
        Identify ghost nodes for each subgraph.

        Ghost nodes: nodes not owned by this subgraph but are neighbors
        of nodes in this subgraph.

        Returns:
            {subgraph_id: {'ghost_nodes': tensor, 'num_ghosts': int}}
        """
        edge_index = self.full_data.edge_index
        ghost_mapping = {}

        for sg_id in range(self.num_subgraphs):
            owned_nodes = self.partition_mapping[sg_id]

            # Find edges where target is in owned_nodes
            mask = torch.isin(edge_index[1], owned_nodes)
            incoming_edges = edge_index[:, mask]

            # Source nodes not in owned_nodes = ghost nodes
            src_nodes = incoming_edges[0]
            ghost_mask = ~torch.isin(src_nodes, owned_nodes)
            ghost_nodes = src_nodes[ghost_mask].unique()

            ghost_mapping[sg_id] = {
                'ghost_nodes': ghost_nodes,
                'num_ghosts': len(ghost_nodes)
            }

        return ghost_mapping

    def _calculate_max_sizes(self) -> Tuple[int, int]:
        """
        Calculate maximum nodes and edges across all subgraphs.

        Returns:
            (max_nodes, max_edges): with 10% safety margin
        """
        max_nodes = 0
        max_edges = 0

        edge_index = self.full_data.edge_index

        for sg_id in range(self.num_subgraphs):
            owned_nodes = self.partition_mapping[sg_id]
            num_owned = len(owned_nodes)
            num_ghosts = self.ghost_node_mapping[sg_id]['num_ghosts']

            total_nodes = num_owned + num_ghosts

            # Count edges for this subgraph
            mask = torch.isin(edge_index[1], owned_nodes)
            num_edges = mask.sum().item()

            max_nodes = max(max_nodes, total_nodes)
            max_edges = max(max_edges, num_edges)

        # Add 10% safety margin
        max_nodes = int(max_nodes * 1.1)
        max_edges = int(max_edges * 1.1)

        return max_nodes, max_edges

    def get_subgraph(self, subgraph_id: int) -> Dict:
        """
        Get data for a specific subgraph.

        Args:
            subgraph_id: Subgraph ID (0 to num_subgraphs-1)

        Returns:
            {
                'edge_index': [2, m] local edge indices
                'x': [n + num_ghosts, 500] node features
                'num_nodes': total nodes (owned + ghost)
                'num_edges': number of edges
                'num_owned': number of owned nodes
                'num_ghosts': number of ghost nodes
            }
        """
        if subgraph_id < 0 or subgraph_id >= self.num_subgraphs:
            raise ValueError(f"Invalid subgraph_id: {subgraph_id}")

        owned_nodes = self.partition_mapping[subgraph_id]
        ghost_nodes = self.ghost_node_mapping[subgraph_id]['ghost_nodes']

        # Combine owned and ghost nodes
        all_nodes = torch.cat([owned_nodes, ghost_nodes])

        # Build global to local ID mapping
        node_mapping = {int(nid): i for i, nid in enumerate(all_nodes)}

        # Extract edges for this subgraph
        edge_index = self.full_data.edge_index
        mask = torch.isin(edge_index[1], owned_nodes)
        subgraph_edges = edge_index[:, mask]

        # Map global IDs to local IDs
        local_edge_index = torch.zeros_like(subgraph_edges)
        for i in range(subgraph_edges.shape[1]):
            src_global = int(subgraph_edges[0, i])
            dst_global = int(subgraph_edges[1, i])
            local_edge_index[0, i] = node_mapping[src_global]
            local_edge_index[1, i] = node_mapping[dst_global]

        # Extract features
        x = self.full_data.x[all_nodes]

        return {
            'edge_index': local_edge_index,
            'x': x,
            'num_nodes': len(all_nodes),
            'num_edges': local_edge_index.shape[1],
            'num_owned': len(owned_nodes),
            'num_ghosts': len(ghost_nodes)
        }

    def get_max_size(self) -> Tuple[int, int]:
        """
        Get maximum subgraph size (for NPU static shape).

        Returns:
            (max_nodes, max_edges)
        """
        return self._max_nodes, self._max_edges

    def get_all_subgraphs(self) -> List[Dict]:
        """
        Get data for all subgraphs.

        Returns:
            List of subgraph data dictionaries
        """
        return [self.get_subgraph(i) for i in range(self.num_subgraphs)]

    def print_stats(self):
        """Print statistics for all subgraphs."""
        print(f"\nFlickr Subgraph Statistics ({self.num_subgraphs} partitions):")
        print("-" * 60)
        print(f"{'ID':<4} {'Owned':<10} {'Ghost':<10} {'Total':<10} {'Edges':<10}")
        print("-" * 60)

        total_owned = 0
        total_edges = 0

        for sg_id in range(self.num_subgraphs):
            sg = self.get_subgraph(sg_id)
            total_owned += sg['num_owned']
            total_edges += sg['num_edges']

            print(f"{sg_id:<4} {sg['num_owned']:<10} {sg['num_ghosts']:<10} "
                  f"{sg['num_nodes']:<10} {sg['num_edges']:<10}")

        print("-" * 60)
        print(f"Total owned nodes: {total_owned}")
        print(f"Total edges: {total_edges}")
        print(f"Max size (with margin): {self._max_nodes} nodes, {self._max_edges} edges")


if __name__ == "__main__":
    # Test the loader
    loader = FlickrSubgraphLoader(num_subgraphs=8)
    loader.print_stats()

    # Test getting a subgraph
    print("\nTesting subgraph 0:")
    sg0 = loader.get_subgraph(0)
    print(f"  x shape: {sg0['x'].shape}")
    print(f"  edge_index shape: {sg0['edge_index'].shape}")
    print(f"  num_nodes: {sg0['num_nodes']}")
    print(f"  num_edges: {sg0['num_edges']}")
