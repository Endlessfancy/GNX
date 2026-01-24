"""
METIS-based Graph Partitioner for ogbn-products Dataset

This script partitions the ogbn-products dataset using METIS min-cut algorithm
to generate subgraph partition candidates with minimized halo nodes.

ogbn-products Dataset (from OGB):
- Nodes: ~2,449,029
- Edges: ~61,859,140
- Features: 100

Constraints:
- Max nodes per subgraph: 100,000
- Halo ratio limit: 30%
- K_min = ceil(2,449,029 / 100,000) = 25
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


class OGBNProductsPartitioner:
    """
    METIS-based partitioner for ogbn-products dataset.

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

        # Load ogbn-products dataset
        self.data = self._load_ogbn_products()
        self.num_nodes = self.data['num_nodes']
        self.num_edges = self.data['num_edges']
        self.edge_index = self.data['edge_index']

        # Build adjacency list for METIS
        self.adj_list = self._build_adjacency_list()

        # Calculate K bounds
        self.k_min = max(1, int(np.ceil(self.num_nodes / max_nodes_per_subgraph)))

        print(f"\nogbn-products Dataset Statistics:")
        print(f"  Nodes: {self.num_nodes:,}")
        print(f"  Edges: {self.num_edges:,}")
        print(f"\nPartitioning Constraints:")
        print(f"  Max nodes per subgraph: {max_nodes_per_subgraph:,}")
        print(f"  Halo ratio limit: {halo_ratio_limit*100:.0f}%")
        print(f"  K_min (based on size): {self.k_min}")

    def _load_ogbn_products(self):
        """Load ogbn-products dataset using OGB"""
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
            import os

            print("Loading ogbn-products dataset...")

            # Use existing data location if available
            existing_path = Path('/home/haoyang/private/GNX/data/ogb')
            if existing_path.exists():
                root = str(existing_path)
            else:
                root = str(self.cache_dir)

            # Suppress download prompt by checking if data exists
            dataset = PygNodePropPredDataset(name='ogbn-products', root=root)
            data = dataset[0]

            return {
                'num_nodes': data.num_nodes,
                'num_edges': data.edge_index.shape[1],
                'edge_index': data.edge_index.numpy(),
                'x': data.x
            }
        except ImportError:
            raise ImportError("ogb required. Install: pip install ogb")

    def _build_adjacency_list(self) -> List[np.ndarray]:
        """
        Build adjacency list format required by pymetis.

        Returns:
            List where adj_list[i] contains neighbors of node i
        """
        print("Building adjacency list for METIS...")
        edge_index = self.edge_index

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
        print(f"  Running METIS for K={k}...")
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
        k_min: int = None,
        k_max: int = None,
        verbose: bool = True
    ) -> List[PartitionResult]:
        """
        Generate partition candidates for K in [K_min, K_max].

        Args:
            k_min: Minimum K to try (default: self.k_min)
            k_max: Maximum K to try
            verbose: Print progress

        Returns:
            List of PartitionResults
        """
        if k_min is None:
            k_min = self.k_min
        if k_max is None:
            k_max = k_min + 20  # Default range

        results = []

        print(f"\n{'='*80}")
        print(f"Generating Partition Candidates: K in [{k_min}, {k_max}]")
        print(f"{'='*80}")

        for k in range(k_min, k_max + 1):
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
            'dataset': 'ogbn-products',
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

    def generate_markdown_report(self, results: List[PartitionResult], output_path: Path):
        """Generate detailed markdown report"""
        lines = []
        lines.append("# ogbn-products Dataset METIS Partition Analysis\n")

        # Dataset info
        lines.append("## Dataset Information\n")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| Dataset | ogbn-products |")
        lines.append(f"| Nodes | {self.num_nodes:,} |")
        lines.append(f"| Edges | {self.num_edges:,} |")
        lines.append(f"| Average Degree | ~{self.num_edges * 2 / self.num_nodes:.1f} |")
        lines.append("")

        # Constraints
        lines.append("## Partition Constraints\n")
        lines.append("| Constraint | Value |")
        lines.append("|------------|-------|")
        lines.append(f"| Max nodes per subgraph | {self.max_nodes_per_subgraph:,} |")
        lines.append(f"| Halo ratio limit | {self.halo_ratio_limit*100:.0f}% |")
        lines.append(f"| K_min (based on size) | {self.k_min} |")
        lines.append("")

        # Summary
        valid_results = [r for r in results if r.is_valid]
        lines.append("## Summary\n")
        if valid_results:
            lines.append(f"**Result: {len(valid_results)} valid partitions found**\n")
            lines.append(f"Valid K values: {[r.k for r in valid_results]}\n")
        else:
            lines.append("**Result: No valid partitions found**\n")
        lines.append("")

        # Overview table
        lines.append("## Partition Results Overview\n")
        lines.append("| K | Valid | Node Range | Max Halo Ratio | Edge Cut | Overall Halo Ratio |")
        lines.append("|---|-------|------------|----------------|----------|-------------------|")

        for r in results:
            min_owned = min(s['num_owned'] for s in r.subgraph_stats)
            max_owned = max(s['num_owned'] for s in r.subgraph_stats)
            max_halo_ratio = max(s['halo_ratio'] for s in r.subgraph_stats)
            valid_str = "Yes" if r.is_valid else "No"
            node_range = f"{min_owned:,} - {max_owned:,}"
            lines.append(f"| {r.k} | {valid_str} | {node_range} | {max_halo_ratio*100:.1f}% | {r.edge_cut:,} | {r.halo_ratio*100:.1f}% |")

        lines.append("")

        # Detailed subgraph stats for each K
        lines.append("## Detailed Subgraph Statistics\n")

        for r in results:
            lines.append(f"### K={r.k}\n")
            lines.append("| ID | Owned | Halo | Total Nodes | Internal Edges | Boundary Edges | Halo% |")
            lines.append("|----|-------|------|-------------|----------------|----------------|-------|")

            for s in r.subgraph_stats:
                lines.append(f"| {s['partition_id']} | {s['num_owned']:,} | {s['num_halo']:,} | {s['total_nodes']:,} | {s['internal_edges']:,} | {s['boundary_edges']:,} | {s['halo_ratio']*100:.1f}% |")

            status = "✓ Valid" if r.is_valid else "✗ Invalid"
            lines.append(f"\n- **Edge cut**: {r.edge_cut:,} | **Overall halo ratio**: {r.halo_ratio*100:.1f}% | **Status**: {status}\n")
            lines.append("---\n")

        # Conclusions
        lines.append("## Conclusions\n")
        if valid_results:
            best = min(valid_results, key=lambda r: r.halo_ratio)
            lines.append(f"- **Best partition**: K={best.k} with {best.halo_ratio*100:.1f}% overall halo ratio")
            lines.append(f"- **Valid K range**: {min(r.k for r in valid_results)} to {max(r.k for r in valid_results)}")
        else:
            lines.append("- No valid partitions found with current constraints")
            lines.append("- Consider relaxing halo ratio limit or max nodes per subgraph")

        lines.append("\n---")
        lines.append("*Generated by METIS partition analysis*")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"Markdown report saved to: {output_path}")


def main():
    """Main function to run partitioning experiments"""

    # Configuration
    MAX_NODES_PER_SUBGRAPH = 100000  # 100k max
    HALO_RATIO_LIMIT = 0.30  # 30% max

    output_dir = Path(__file__).parent

    # Create partitioner
    partitioner = OGBNProductsPartitioner(
        max_nodes_per_subgraph=MAX_NODES_PER_SUBGRAPH,
        halo_ratio_limit=HALO_RATIO_LIMIT,
        cache_dir=output_dir / 'data'
    )

    # K_min based on node constraint
    k_min = partitioner.k_min
    # Test K from k_min to k_min + 15 (or adjust based on dataset size)
    k_max = k_min + 15

    # For very large datasets, we might want to test a specific range
    # Let's test from K=2 up to K=k_min+10 to see the full picture
    k_start = 2
    k_end = min(k_min + 10, 50)  # Cap at 50 partitions

    print(f"\nTesting K from {k_start} to {k_end}")

    # Generate partition candidates
    results = partitioner.find_partition_candidates(k_min=k_start, k_max=k_end, verbose=True)

    # Save JSON results
    json_path = output_dir / 'ogbn_products_partition_results.json'
    partitioner.save_results(results, json_path)

    # Generate markdown report
    md_path = output_dir / 'ogbn_products_partition.md'
    partitioner.generate_markdown_report(results, md_path)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    valid_results = [r for r in results if r.is_valid]
    if valid_results:
        best = min(valid_results, key=lambda r: r.halo_ratio)
        print(f"\nBest partition: K={best.k}")
        print(f"  Overall halo ratio: {best.halo_ratio*100:.1f}%")
        print(f"  Edge cut: {best.edge_cut:,}")
        print(f"\nAll valid partitions (K values): {[r.k for r in valid_results]}")
    else:
        print("\nNo valid partitions found with current constraints!")
        print("Consider relaxing constraints:")
        print(f"  - Increase max_nodes_per_subgraph (current: {MAX_NODES_PER_SUBGRAPH:,})")
        print(f"  - Increase halo_ratio_limit (current: {HALO_RATIO_LIMIT*100:.0f}%)")


if __name__ == "__main__":
    main()
