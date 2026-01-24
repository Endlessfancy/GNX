"""
METIS-based Graph Partitioner for Reddit2 Dataset

This script partitions the Reddit2 dataset using METIS min-cut algorithm
to generate subgraph partition candidates with minimized halo nodes.

Requirements from requirement.md:
- Generate discrete set of subgraph partition candidates
- K_min: constrained by max subgraph size (100k nodes)
- K_max: constrained by halo node ratio limit (30%)
- For each K, use METIS to get optimal partition minimizing halo nodes

Reddit2 Dataset:
- Nodes: ~232,965
- Edges: ~23,000,000 (23M)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import pymetis
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class PartitionResult:
    """Result of a single partition configuration"""
    k: int  # Number of partitions
    node_assignments: np.ndarray  # node_id -> partition_id
    subgraph_stats: List[Dict]  # Per-subgraph statistics
    total_halo_nodes: int
    halo_ratio: float  # total_halo_nodes / (k * avg_partition_size)
    edge_cut: int  # Number of cut edges
    is_valid: bool  # Meets constraints


class Reddit2Partitioner:
    """
    METIS-based partitioner for Reddit2 dataset.

    Generates partition candidates for K in [K_min, K_max] where:
    - K_min: ceil(num_nodes / max_nodes_per_subgraph)
    - K_max: limited by halo_ratio_limit
    """

    def __init__(
        self,
        max_nodes_per_subgraph: int = 100000,
        halo_ratio_limit: float = 0.30,
        cache_dir: Path = Path('data')
    ):
        """
        Args:
            max_nodes_per_subgraph: Maximum nodes allowed per subgraph
            halo_ratio_limit: Maximum halo node ratio (halo_nodes / owned_nodes)
            cache_dir: Directory for caching dataset
        """
        self.max_nodes_per_subgraph = max_nodes_per_subgraph
        self.halo_ratio_limit = halo_ratio_limit
        self.cache_dir = cache_dir

        # Load Reddit2 dataset
        self.data = self._load_reddit2()
        self.num_nodes = self.data.num_nodes
        self.num_edges = self.data.num_edges

        # Build adjacency list for METIS
        self.adj_list = self._build_adjacency_list()

        # Calculate K bounds
        self.k_min = max(1, int(np.ceil(self.num_nodes / max_nodes_per_subgraph)))

        print(f"\nReddit2 Dataset Statistics:")
        print(f"  Nodes: {self.num_nodes:,}")
        print(f"  Edges: {self.num_edges:,}")
        print(f"  Features: {self.data.x.shape[1]}")
        print(f"\nPartitioning Constraints:")
        print(f"  Max nodes per subgraph: {max_nodes_per_subgraph:,}")
        print(f"  Halo ratio limit: {halo_ratio_limit*100:.0f}%")
        print(f"  K_min (based on size): {self.k_min}")

    def _load_reddit2(self):
        """Load Reddit2 dataset using PyG"""
        try:
            from torch_geometric.datasets import Reddit2
            from torch_geometric.data.data import Data

            # Handle PyTorch 2.6+ serialization
            try:
                torch.serialization.add_safe_globals([Data])
            except AttributeError:
                pass

            print("Loading Reddit2 dataset...")
            dataset = Reddit2(root=str(self.cache_dir / 'Reddit2'))
            return dataset[0]
        except ImportError:
            raise ImportError("torch_geometric required. Install: pip install torch_geometric")

    def _build_adjacency_list(self) -> List[np.ndarray]:
        """
        Build adjacency list format required by pymetis.

        Returns:
            List where adj_list[i] contains neighbors of node i
        """
        print("Building adjacency list for METIS...")
        edge_index = self.data.edge_index.numpy()

        # Build neighbor lists
        neighbors = defaultdict(list)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            neighbors[src].append(dst)

        # Convert to list of numpy arrays (pymetis format)
        adj_list = []
        for node_id in range(self.num_nodes):
            adj_list.append(np.array(neighbors[node_id], dtype=np.int32))

        return adj_list

    def partition(self, k: int) -> PartitionResult:
        """
        Partition graph into k subgraphs using METIS.

        Args:
            k: Number of partitions

        Returns:
            PartitionResult with statistics
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        if k == 1:
            # Special case: no partitioning needed
            node_assignments = np.zeros(self.num_nodes, dtype=np.int32)
            return self._compute_partition_stats(node_assignments, k, edge_cut=0)

        # Run METIS partitioning
        # pymetis.part_graph returns (edge_cut, partition_assignment)
        edge_cut, node_assignments = pymetis.part_graph(k, adjacency=self.adj_list)
        node_assignments = np.array(node_assignments, dtype=np.int32)

        return self._compute_partition_stats(node_assignments, k, edge_cut)

    def _compute_partition_stats(
        self,
        node_assignments: np.ndarray,
        k: int,
        edge_cut: int
    ) -> PartitionResult:
        """
        Compute detailed statistics for a partition.

        Halo nodes (ghost nodes): Nodes not owned by a subgraph but are
        neighbors of nodes in that subgraph.
        """
        edge_index = self.data.edge_index.numpy()

        # Group nodes by partition
        partition_nodes = defaultdict(set)
        for node_id, part_id in enumerate(node_assignments):
            partition_nodes[part_id].add(node_id)

        subgraph_stats = []
        total_halo_nodes = 0

        for part_id in range(k):
            owned_nodes = partition_nodes[part_id]
            num_owned = len(owned_nodes)

            # Find halo nodes: neighbors of owned nodes that are not owned
            halo_nodes = set()
            internal_edges = 0
            boundary_edges = 0

            for node_id in owned_nodes:
                for neighbor in self.adj_list[node_id]:
                    if neighbor in owned_nodes:
                        internal_edges += 1
                    else:
                        boundary_edges += 1
                        halo_nodes.add(neighbor)

            num_halo = len(halo_nodes)
            total_nodes = num_owned + num_halo
            halo_ratio = num_halo / num_owned if num_owned > 0 else 0

            subgraph_stats.append({
                'partition_id': part_id,
                'num_owned': num_owned,
                'num_halo': num_halo,
                'total_nodes': total_nodes,
                'internal_edges': internal_edges,
                'boundary_edges': boundary_edges,
                'halo_ratio': halo_ratio
            })

            total_halo_nodes += num_halo

        # Overall halo ratio: average across partitions
        avg_owned = self.num_nodes / k
        overall_halo_ratio = total_halo_nodes / (k * avg_owned) if k > 0 else 0

        # Check if partition meets constraints
        max_owned = max(s['num_owned'] for s in subgraph_stats)
        max_halo_ratio = max(s['halo_ratio'] for s in subgraph_stats)

        is_valid = (
            max_owned <= self.max_nodes_per_subgraph and
            max_halo_ratio <= self.halo_ratio_limit
        )

        return PartitionResult(
            k=k,
            node_assignments=node_assignments,
            subgraph_stats=subgraph_stats,
            total_halo_nodes=total_halo_nodes,
            halo_ratio=overall_halo_ratio,
            edge_cut=edge_cut,
            is_valid=is_valid
        )

    def find_partition_candidates(
        self,
        k_max: int = 20,
        verbose: bool = True
    ) -> List[PartitionResult]:
        """
        Generate partition candidates for K in [K_min, K_max].

        Args:
            k_max: Maximum K to try
            verbose: Print progress

        Returns:
            List of PartitionResults, filtering for valid ones
        """
        results = []

        print(f"\n{'='*80}")
        print(f"Generating Partition Candidates: K in [{self.k_min}, {k_max}]")
        print(f"{'='*80}")

        for k in range(self.k_min, k_max + 1):
            if verbose:
                print(f"\nPartitioning with K={k}...")

            result = self.partition(k)
            results.append(result)

            if verbose:
                self._print_result_summary(result)

        # Filter valid results
        valid_results = [r for r in results if r.is_valid]

        print(f"\n{'='*80}")
        print(f"Summary: {len(valid_results)}/{len(results)} valid partitions")
        print(f"{'='*80}")

        return results

    def _print_result_summary(self, result: PartitionResult):
        """Print summary of a partition result"""
        max_owned = max(s['num_owned'] for s in result.subgraph_stats)
        min_owned = min(s['num_owned'] for s in result.subgraph_stats)
        max_halo = max(s['num_halo'] for s in result.subgraph_stats)
        max_halo_ratio = max(s['halo_ratio'] for s in result.subgraph_stats)

        valid_str = "VALID" if result.is_valid else "INVALID"

        print(f"  K={result.k}: {valid_str}")
        print(f"    Nodes per subgraph: {min_owned:,} - {max_owned:,}")
        print(f"    Max halo nodes: {max_halo:,} (ratio: {max_halo_ratio*100:.1f}%)")
        print(f"    Edge cut: {result.edge_cut:,}")
        print(f"    Overall halo ratio: {result.halo_ratio*100:.1f}%")

    def print_detailed_stats(self, result: PartitionResult):
        """Print detailed statistics for each subgraph"""
        print(f"\n{'='*80}")
        print(f"Detailed Statistics for K={result.k}")
        print(f"{'='*80}")
        print(f"\n{'ID':<4} {'Owned':<10} {'Halo':<10} {'Total':<10} {'Internal':<12} {'Boundary':<12} {'Halo%':<8}")
        print("-" * 80)

        for s in result.subgraph_stats:
            print(f"{s['partition_id']:<4} {s['num_owned']:<10,} {s['num_halo']:<10,} "
                  f"{s['total_nodes']:<10,} {s['internal_edges']:<12,} "
                  f"{s['boundary_edges']:<12,} {s['halo_ratio']*100:<7.1f}%")

        print("-" * 80)
        print(f"Total owned: {self.num_nodes:,}")
        print(f"Total halo: {result.total_halo_nodes:,}")
        print(f"Edge cut: {result.edge_cut:,}")
        print(f"Valid: {result.is_valid}")

    def save_results(self, results: List[PartitionResult], output_path: Path):
        """Save partition results to JSON"""
        output_data = {
            'dataset': 'Reddit2',
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'constraints': {
                'max_nodes_per_subgraph': self.max_nodes_per_subgraph,
                'halo_ratio_limit': self.halo_ratio_limit
            },
            'partitions': []
        }

        for r in results:
            partition_data = {
                'k': r.k,
                'is_valid': r.is_valid,
                'edge_cut': r.edge_cut,
                'overall_halo_ratio': r.halo_ratio,
                'total_halo_nodes': r.total_halo_nodes,
                'subgraphs': []
            }

            for s in r.subgraph_stats:
                partition_data['subgraphs'].append({
                    'id': s['partition_id'],
                    'num_owned': s['num_owned'],
                    'num_halo': s['num_halo'],
                    'total_nodes': s['total_nodes'],
                    'internal_edges': s['internal_edges'],
                    'boundary_edges': s['boundary_edges'],
                    'halo_ratio': s['halo_ratio']
                })

            output_data['partitions'].append(partition_data)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main():
    """Main function to run partitioning experiments"""

    # Configuration
    MAX_NODES_PER_SUBGRAPH = 100000  # 100k max
    HALO_RATIO_LIMIT = 0.30  # 30% max
    K_MAX = 20  # Try up to 20 partitions

    output_dir = Path(__file__).parent

    # Create partitioner
    partitioner = Reddit2Partitioner(
        max_nodes_per_subgraph=MAX_NODES_PER_SUBGRAPH,
        halo_ratio_limit=HALO_RATIO_LIMIT,
        cache_dir=output_dir / 'data'
    )

    # Generate partition candidates
    results = partitioner.find_partition_candidates(k_max=K_MAX, verbose=True)

    # Print detailed stats for valid partitions
    print("\n" + "="*80)
    print("VALID PARTITION DETAILS")
    print("="*80)

    valid_results = [r for r in results if r.is_valid]
    for result in valid_results:
        partitioner.print_detailed_stats(result)

    # Summary table
    print("\n" + "="*80)
    print("PARTITION SUMMARY TABLE")
    print("="*80)
    print(f"\n{'K':<4} {'Valid':<8} {'Node Range':<20} {'Edge Range':<25} {'Max Halo%':<10} {'EdgeCut':<12}")
    print("-" * 90)

    for r in results:
        min_owned = min(s['num_owned'] for s in r.subgraph_stats)
        max_owned = max(s['num_owned'] for s in r.subgraph_stats)
        min_edges = min(s['internal_edges'] + s['boundary_edges'] for s in r.subgraph_stats)
        max_edges = max(s['internal_edges'] + s['boundary_edges'] for s in r.subgraph_stats)
        max_halo_ratio = max(s['halo_ratio'] for s in r.subgraph_stats)

        valid_str = "Yes" if r.is_valid else "No"
        node_range = f"{min_owned:,} - {max_owned:,}"
        edge_range = f"{min_edges:,} - {max_edges:,}"

        print(f"{r.k:<4} {valid_str:<8} {node_range:<20} {edge_range:<25} {max_halo_ratio*100:<9.1f}% {r.edge_cut:<12,}")

    # Save results
    output_path = output_dir / 'reddit2_partition_results.json'
    partitioner.save_results(results, output_path)

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if valid_results:
        # Find best partition (minimize halo ratio while meeting constraints)
        best = min(valid_results, key=lambda r: r.halo_ratio)
        print(f"\nBest partition: K={best.k}")
        print(f"  Overall halo ratio: {best.halo_ratio*100:.1f}%")
        print(f"  Edge cut: {best.edge_cut:,}")

        # Also show options with different trade-offs
        print(f"\nAll valid partitions (K values): {[r.k for r in valid_results]}")
    else:
        print("\nNo valid partitions found with current constraints!")
        print("Consider relaxing constraints:")
        print(f"  - Increase max_nodes_per_subgraph (current: {MAX_NODES_PER_SUBGRAPH:,})")
        print(f"  - Increase halo_ratio_limit (current: {HALO_RATIO_LIMIT*100:.0f}%)")


if __name__ == "__main__":
    main()
