#!/usr/bin/env python3
"""
Phase 1: Graph Partition

METIS k-way graph partitioning with NPU padding.

Usage:
    python phase1_partition.py --dataset flickr --k-set 8,9,10,11,12
    python phase1_partition.py --dataset flickr --output output/phase1_partition.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add compiler directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import CompilerConfig, GraphLoader
from core import GraphPartitioner


# ============================================================================
# Serialization Helpers
# ============================================================================

def subgraph_to_dict(sg):
    """Subgraph -> dict for JSON serialization"""
    return {
        'id': sg.id,
        'n': sg.n,
        'm': sg.m,
        'n_pad': sg.n_pad,
        'm_pad': sg.m_pad,
        'cut_edges': sg.cut_edges
    }


# ============================================================================
# Phase 1 Main
# ============================================================================

def run_phase1(dataset_name: str, k_set: list, output_path: Path,
               use_metis: bool = True, npu_padding: int = 1000):
    """
    Run Phase 1: Graph Partition

    Args:
        dataset_name: Dataset name (flickr, reddit, yelp, synthetic)
        k_set: List of K values to try
        output_path: Output JSON path
        use_metis: Whether to use METIS
        npu_padding: NPU padding multiple

    Returns:
        Phase 1 result dict
    """
    print("=" * 60)
    print("Phase 1: Graph Partition")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset: {dataset_name}...")
    t0 = time.perf_counter()
    graph_loader = GraphLoader()
    graph_data = graph_loader.load_dataset(dataset_name)
    load_time = time.perf_counter() - t0

    print(f"  Nodes: {graph_data.num_nodes:,}")
    print(f"  Edges: {graph_data.num_edges:,}")
    print(f"  Load time: {load_time:.2f}s")

    # Initialize partitioner
    partitioner = GraphPartitioner(
        padding_multiple=npu_padding,
        use_metis=use_metis
    )

    # Run partition for all K values
    print(f"\nPartitioning for K = {k_set}...")
    t0 = time.perf_counter()
    partition_results = partitioner.partition_multiple_k(
        graph_data.edge_index,
        k_set
    )
    partition_time = time.perf_counter() - t0
    print(f"  Total partition time: {partition_time:.2f}s")

    # Build output
    result = {
        'phase': 1,
        'dataset': dataset_name,
        'num_nodes': graph_data.num_nodes,
        'num_edges': graph_data.num_edges,
        'use_metis': use_metis,
        'npu_padding_multiple': npu_padding,
        'partition_time_sec': partition_time,
        'k_results': {}
    }

    for k, subgraphs in partition_results.items():
        result['k_results'][str(k)] = [subgraph_to_dict(sg) for sg in subgraphs]

    # Print summary
    print(f"\n{'K':<6} {'Subgraphs':<12} {'Avg Nodes':<12} {'Total Cut':<12}")
    print("-" * 45)
    for k, subgraphs in partition_results.items():
        avg_n = sum(sg.n for sg in subgraphs) / len(subgraphs)
        total_cut = sum(sg.cut_edges for sg in subgraphs) // 2
        print(f"{k:<6} {len(subgraphs):<12} {avg_n:<12.0f} {total_cut:<12,}")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"\nPhase 1 output saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Graph Partition")
    parser.add_argument('--dataset', type=str, default='flickr',
                        help='Dataset name (flickr, reddit, yelp)')
    parser.add_argument('--k-set', type=str, default='8,9,10,11,12,13,14,15',
                        help='Comma-separated K values')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--no-metis', action='store_true',
                        help='Use random partition instead of METIS')
    parser.add_argument('--npu-padding', type=int, default=1000,
                        help='NPU padding multiple')
    args = parser.parse_args()

    k_set = [int(k.strip()) for k in args.k_set.split(',')]

    output_path = Path(args.output) if args.output else \
        Path(__file__).parent / 'output' / 'phase1_partition.json'

    run_phase1(
        dataset_name=args.dataset,
        k_set=k_set,
        output_path=output_path,
        use_metis=not args.no_metis,
        npu_padding=args.npu_padding
    )


if __name__ == '__main__':
    main()
