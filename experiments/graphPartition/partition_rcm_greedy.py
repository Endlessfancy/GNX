#!/usr/bin/env python3
"""
RC-M + Greedy Edge-Balanced Partition

A fast graph partitioning method for Edge GNN Inference:
1. RCM (Reverse Cuthill-McKee) reordering for locality optimization
2. Greedy partitioning based on edge count for load balancing

Compared to METIS:
- Much faster: O(n+m) vs O(m log n)
- Good load balance (edges per partition)
- Reduced halo due to RCM locality

Usage:
    python partition_rcm_greedy.py --flickr
    python partition_rcm_greedy.py --reddit2
    python partition_rcm_greedy.py --ogbn
    python partition_rcm_greedy.py --all
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

import torch

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / 'data'
RESULTS_DIR = SCRIPT_DIR

# K values per dataset
K_VALUES_FLICKR = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
K_VALUES_REDDIT2 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
K_VALUES_OGBN = [20, 40, 60, 80, 100, 150, 200]


# ============================================================================
# Dataset Loading
# ============================================================================

def load_flickr():
    """Load Flickr dataset"""
    from torch_geometric.datasets import Flickr
    from torch_geometric.data.data import Data
    try:
        torch.serialization.add_safe_globals([Data])
    except AttributeError:
        pass
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = Flickr(root=str(DATA_DIR / 'Flickr'))
    return dataset[0]


def load_reddit2():
    """Load Reddit2 dataset"""
    from torch_geometric.datasets import Reddit2
    from torch_geometric.data.data import Data
    try:
        torch.serialization.add_safe_globals([Data])
    except AttributeError:
        pass
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = Reddit2(root=str(DATA_DIR / 'Reddit2'))
    return dataset[0]


def load_ogbn_products():
    """Load ogbn-products dataset"""
    from ogb.nodeproppred import PygNodePropPredDataset
    import os
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ogbn_dir = DATA_DIR / 'ogbn_products'
    raw_dir = ogbn_dir / 'raw'
    os.makedirs(str(raw_dir), exist_ok=True)
    dataset = PygNodePropPredDataset(name='ogbn-products', root=str(DATA_DIR))
    return dataset[0]


# ============================================================================
# RCM + Greedy Partition Implementation
# ============================================================================

def build_csr_adjacency(edge_index: np.ndarray, num_nodes: int) -> csr_matrix:
    """Build CSR adjacency matrix from edge_index"""
    row = edge_index[0]
    col = edge_index[1]
    data = np.ones(len(row), dtype=np.int32)
    adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adj


def rcm_greedy_partition(edge_index: np.ndarray, num_nodes: int, k: int) -> Dict:
    """
    RC-M + Greedy Edge-Balanced Partition

    Args:
        edge_index: [2, num_edges] edge index array
        num_nodes: number of nodes
        k: number of partitions

    Returns:
        Dictionary with partition results
    """
    result = {
        'k': k,
        'num_nodes': num_nodes,
        'num_edges': edge_index.shape[1],
        'timings': {}
    }

    # Step 1: Build CSR adjacency matrix
    t0 = time.perf_counter()
    adj = build_csr_adjacency(edge_index, num_nodes)
    result['timings']['build_csr'] = time.perf_counter() - t0

    # Step 2: RCM reordering
    t0 = time.perf_counter()
    perm = reverse_cuthill_mckee(adj)
    result['timings']['rcm_reorder'] = time.perf_counter() - t0

    # Create inverse permutation (new_id -> original_id is perm, original_id -> new_id is inv_perm)
    inv_perm = np.argsort(perm)

    # Step 3: Calculate degrees in new ordering
    t0 = time.perf_counter()
    degrees = np.array(adj.sum(axis=1)).flatten()
    reordered_degrees = degrees[perm]

    # Step 4: Greedy partitioning by edge count
    total_edges = adj.nnz
    target = total_edges / k

    partition_boundaries = [0]  # Start indices in reordered space
    current_edges = 0

    for i, deg in enumerate(reordered_degrees):
        current_edges += deg
        if current_edges >= target and len(partition_boundaries) < k:
            partition_boundaries.append(i + 1)
            current_edges = 0

    # Ensure we have exactly k partitions
    if len(partition_boundaries) < k + 1:
        partition_boundaries.append(num_nodes)

    result['timings']['greedy_partition'] = time.perf_counter() - t0
    result['timings']['total'] = sum(result['timings'].values())

    # Step 5: Compute partition statistics
    t0 = time.perf_counter()

    # Create node assignments (in original node ID space)
    node_assignments = np.zeros(num_nodes, dtype=np.int32)
    for part_id in range(k):
        start = partition_boundaries[part_id]
        end = partition_boundaries[part_id + 1] if part_id + 1 < len(partition_boundaries) else num_nodes
        # Nodes in this partition (in original ID space)
        original_nodes = perm[start:end]
        node_assignments[original_nodes] = part_id

    # Compute statistics for each partition
    subgraphs = []
    total_edge_cut = 0
    total_halo = 0

    # Build neighbor lists for halo computation
    neighbors = defaultdict(set)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        neighbors[src].add(dst)

    for part_id in range(k):
        owned_nodes = set(np.where(node_assignments == part_id)[0])

        # Count halo nodes (neighbors outside partition)
        halo_nodes = set()
        boundary_edges = 0
        internal_edges = 0

        for node in owned_nodes:
            for neighbor in neighbors[node]:
                if neighbor not in owned_nodes:
                    halo_nodes.add(neighbor)
                    boundary_edges += 1
                else:
                    internal_edges += 1

        subgraphs.append({
            'id': part_id,
            'num_owned': len(owned_nodes),
            'num_halo': len(halo_nodes),
            'total_nodes': len(owned_nodes) + len(halo_nodes),
            'internal_edges': internal_edges,
            'boundary_edges': boundary_edges,
            'halo_ratio': len(halo_nodes) / len(owned_nodes) if len(owned_nodes) > 0 else 0
        })

        total_edge_cut += boundary_edges
        total_halo += len(halo_nodes)

    # Edge cut is counted twice (once per endpoint), divide by 2
    result['edge_cut'] = total_edge_cut // 2
    result['total_halo_nodes'] = total_halo
    result['overall_halo_ratio'] = total_halo / num_nodes
    result['subgraphs'] = subgraphs
    result['timings']['compute_stats'] = time.perf_counter() - t0

    return result


def naive_partition(edge_index: np.ndarray, num_nodes: int, k: int) -> Dict:
    """
    Naive equal-node partition (baseline)
    Simply divide nodes into k equal parts by ID
    """
    result = {
        'k': k,
        'num_nodes': num_nodes,
        'num_edges': edge_index.shape[1],
        'timings': {}
    }

    t0 = time.perf_counter()

    # Simple division by node ID
    nodes_per_part = num_nodes // k
    node_assignments = np.zeros(num_nodes, dtype=np.int32)
    for i in range(num_nodes):
        node_assignments[i] = min(i // nodes_per_part, k - 1)

    result['timings']['partition'] = time.perf_counter() - t0

    # Compute statistics
    t0 = time.perf_counter()

    neighbors = defaultdict(set)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        neighbors[src].add(dst)

    subgraphs = []
    total_edge_cut = 0
    total_halo = 0

    for part_id in range(k):
        owned_nodes = set(np.where(node_assignments == part_id)[0])

        halo_nodes = set()
        boundary_edges = 0
        internal_edges = 0

        for node in owned_nodes:
            for neighbor in neighbors[node]:
                if neighbor not in owned_nodes:
                    halo_nodes.add(neighbor)
                    boundary_edges += 1
                else:
                    internal_edges += 1

        subgraphs.append({
            'id': part_id,
            'num_owned': len(owned_nodes),
            'num_halo': len(halo_nodes),
            'total_nodes': len(owned_nodes) + len(halo_nodes),
            'internal_edges': internal_edges,
            'boundary_edges': boundary_edges,
            'halo_ratio': len(halo_nodes) / len(owned_nodes) if len(owned_nodes) > 0 else 0
        })

        total_edge_cut += boundary_edges
        total_halo += len(halo_nodes)

    result['edge_cut'] = total_edge_cut // 2
    result['total_halo_nodes'] = total_halo
    result['overall_halo_ratio'] = total_halo / num_nodes
    result['subgraphs'] = subgraphs
    result['timings']['compute_stats'] = time.perf_counter() - t0
    result['timings']['total'] = sum(result['timings'].values())

    return result


# ============================================================================
# Testing Functions
# ============================================================================

def test_dataset(dataset_name: str, load_func, k_values: List[int]) -> Dict:
    """Test partitioning methods on a dataset"""
    print(f"\n{'='*70}")
    print(f"Testing: {dataset_name}")
    print(f"{'='*70}")

    # Load data
    print(f"\nLoading {dataset_name}...")
    t0 = time.perf_counter()
    data = load_func()
    load_time = time.perf_counter() - t0

    num_nodes = data.num_nodes
    num_edges = data.num_edges
    edge_index = data.edge_index.numpy()

    print(f"  Nodes: {num_nodes:,}, Edges: {num_edges:,}")
    print(f"  Load time: {load_time:.2f}s")

    results = {
        'dataset': dataset_name,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'load_time': load_time,
        'k_values': k_values,
        'rcm_greedy': {},
        'naive': {}
    }

    # Test each K value
    print(f"\n{'K':<6} {'Method':<12} {'Time(s)':<10} {'EdgeCut':<12} {'Halo Ratio':<12} {'Edge CV%':<10}")
    print("-" * 70)

    for k in k_values:
        # RCM + Greedy
        rcm_result = rcm_greedy_partition(edge_index, num_nodes, k)
        results['rcm_greedy'][str(k)] = rcm_result

        # Naive partition
        naive_result = naive_partition(edge_index, num_nodes, k)
        results['naive'][str(k)] = naive_result

        # Calculate edge CV% for load balance
        rcm_edges = [s['internal_edges'] + s['boundary_edges'] for s in rcm_result['subgraphs']]
        rcm_cv = np.std(rcm_edges) / np.mean(rcm_edges) * 100 if np.mean(rcm_edges) > 0 else 0

        naive_edges = [s['internal_edges'] + s['boundary_edges'] for s in naive_result['subgraphs']]
        naive_cv = np.std(naive_edges) / np.mean(naive_edges) * 100 if np.mean(naive_edges) > 0 else 0

        print(f"{k:<6} {'RCM+Greedy':<12} {rcm_result['timings']['total']:<10.3f} "
              f"{rcm_result['edge_cut']:<12,} {rcm_result['overall_halo_ratio']:<12.2%} {rcm_cv:<10.1f}")
        print(f"{'':<6} {'Naive':<12} {naive_result['timings']['total']:<10.3f} "
              f"{naive_result['edge_cut']:<12,} {naive_result['overall_halo_ratio']:<12.2%} {naive_cv:<10.1f}")

    return results


def compare_with_metis(results: Dict, metis_file: Path):
    """Compare RCM+Greedy results with METIS results"""
    if not metis_file.exists():
        print(f"\nMETIS results not found: {metis_file}")
        return

    with open(metis_file) as f:
        metis_data = json.load(f)

    print(f"\n{'='*70}")
    print("Comparison with METIS")
    print(f"{'='*70}")
    print(f"\n{'K':<6} {'METIS Cut':<12} {'RCM Cut':<12} {'Naive Cut':<12} {'RCM/METIS':<12}")
    print("-" * 60)

    for partition in metis_data.get('partitions', []):
        k = partition['k']
        if k < 2:
            continue

        metis_cut = partition['edge_cut']

        if str(k) in results['rcm_greedy']:
            rcm_cut = results['rcm_greedy'][str(k)]['edge_cut']
            naive_cut = results['naive'][str(k)]['edge_cut']
            ratio = rcm_cut / metis_cut if metis_cut > 0 else 0
            print(f"{k:<6} {metis_cut:<12,} {rcm_cut:<12,} {naive_cut:<12,} {ratio:<12.2f}x")


def save_results(results: Dict, filename: str):
    """Save results to JSON"""
    filepath = RESULTS_DIR / filename

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to: {filepath}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RC-M + Greedy Partition")
    parser.add_argument("--flickr", action="store_true", help="Test Flickr")
    parser.add_argument("--reddit2", action="store_true", help="Test Reddit2")
    parser.add_argument("--ogbn", action="store_true", help="Test ogbn-products")
    parser.add_argument("--all", action="store_true", help="Test all datasets")
    args = parser.parse_args()

    if args.all:
        args.flickr = True
        args.reddit2 = True
        args.ogbn = True

    if not any([args.flickr, args.reddit2, args.ogbn]):
        parser.print_help()
        return

    print("=" * 70)
    print("RC-M + Greedy Edge-Balanced Partition")
    print("=" * 70)

    all_results = {}

    if args.flickr:
        results = test_dataset('Flickr', load_flickr, K_VALUES_FLICKR)
        all_results['flickr'] = results
        save_results(results, 'rcm_greedy_flickr.json')
        compare_with_metis(results, RESULTS_DIR / 'partition_results.json')

    if args.reddit2:
        results = test_dataset('Reddit2', load_reddit2, K_VALUES_REDDIT2)
        all_results['reddit2'] = results
        save_results(results, 'rcm_greedy_reddit2.json')
        compare_with_metis(results, RESULTS_DIR / 'reddit2_partition_results.json')

    if args.ogbn:
        results = test_dataset('ogbn-products', load_ogbn_products, K_VALUES_OGBN)
        all_results['ogbn'] = results
        save_results(results, 'rcm_greedy_ogbn.json')
        compare_with_metis(results, RESULTS_DIR / 'ogbn_products_partition_results.json')

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nRCM + Greedy vs Naive Partition:")
    print("- RCM reordering improves locality (reduces halo)")
    print("- Greedy by edges ensures load balance")
    print("- Much faster than METIS (linear time)")
    print("\nTrade-off: Edge cut is higher than METIS, but much lower than Naive")


if __name__ == '__main__':
    main()
