#!/usr/bin/env python3
"""
Graph Partition Time Measurement Script

Measures the time to partition graphs for:
- Flickr dataset
- Reddit2 dataset
- ogbn-products dataset

Measures:
1. Dataset loading time
2. Adjacency list building time
3. METIS partitioning time (for various K values)
4. Statistics computation time

Usage:
    python measure_partition_time.py --all
    python measure_partition_time.py --flickr
    python measure_partition_time.py --reddit2
    python measure_partition_time.py --ogbn
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
GRAPH_PARTITION_DIR = PROJECT_ROOT / 'experiments' / 'graphPartition'
sys.path.insert(0, str(PROJECT_ROOT))

import torch

# Try to import pymetis
try:
    import pymetis
    PYMETIS_AVAILABLE = True
except ImportError:
    PYMETIS_AVAILABLE = False
    print("WARNING: pymetis not available. Install: pip install pymetis")

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = SCRIPT_DIR / 'results'
DATA_DIR = GRAPH_PARTITION_DIR / 'data'

# Partition K values per dataset
K_VALUES_FLICKR = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
K_VALUES_REDDIT2 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
K_VALUES_OGBN = [20, 40, 60, 80, 100, 150, 200]

# Number of repetitions for timing
NUM_REPEATS = 3


# ============================================================================
# Dataset Loading Functions
# ============================================================================

def load_flickr(cache_dir: Path):
    """Load Flickr dataset"""
    from torch_geometric.datasets import Flickr
    from torch_geometric.data.data import Data

    try:
        torch.serialization.add_safe_globals([Data])
    except AttributeError:
        pass

    # Ensure directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = Flickr(root=str(cache_dir / 'Flickr'))
    return dataset[0]


def load_reddit2(cache_dir: Path):
    """Load Reddit2 dataset"""
    from torch_geometric.datasets import Reddit2
    from torch_geometric.data.data import Data

    try:
        torch.serialization.add_safe_globals([Data])
    except AttributeError:
        pass

    # Ensure directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = Reddit2(root=str(cache_dir / 'Reddit2'))
    return dataset[0]


def load_ogbn_products(cache_dir: Path):
    """Load ogbn-products dataset"""
    from ogb.nodeproppred import PygNodePropPredDataset
    import os

    # Ensure all directories exist (Windows compatibility)
    ogbn_dir = cache_dir / 'ogbn_products'
    raw_dir = ogbn_dir / 'raw'
    os.makedirs(str(raw_dir), exist_ok=True)

    dataset = PygNodePropPredDataset(name='ogbn-products', root=str(cache_dir))
    return dataset[0]


# ============================================================================
# Partition Functions
# ============================================================================

def build_adjacency_list(data) -> List[np.ndarray]:
    """Build adjacency list for pymetis"""
    edge_index = data.edge_index.numpy()
    num_nodes = data.num_nodes

    neighbors = defaultdict(list)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        neighbors[src].append(dst)

    adj_list = []
    for node_id in range(num_nodes):
        adj_list.append(np.array(neighbors[node_id], dtype=np.int32))

    return adj_list


def run_metis_partition(adj_list: List[np.ndarray], k: int):
    """Run METIS partitioning"""
    if k == 1:
        return 0, np.zeros(len(adj_list), dtype=np.int32)

    edge_cut, node_assignments = pymetis.part_graph(k, adjacency=adj_list)
    return edge_cut, np.array(node_assignments, dtype=np.int32)


def compute_partition_stats(data, node_assignments: np.ndarray, adj_list: List[np.ndarray], k: int):
    """Compute partition statistics"""
    partition_nodes = defaultdict(set)
    for node_id, part_id in enumerate(node_assignments):
        partition_nodes[part_id].add(node_id)

    total_halo = 0
    for part_id in range(k):
        owned_nodes = partition_nodes[part_id]
        halo_nodes = set()
        for node_id in owned_nodes:
            for neighbor in adj_list[node_id]:
                if neighbor not in owned_nodes:
                    halo_nodes.add(neighbor)
        total_halo += len(halo_nodes)

    return total_halo


# ============================================================================
# Timing Functions
# ============================================================================

def measure_dataset_time(dataset_name: str, load_func, cache_dir: Path, k_values: List[int]) -> Dict:
    """Measure time for a single dataset"""
    print(f"\n{'='*70}")
    print(f"Measuring Partition Time for: {dataset_name}")
    print(f"K values: {k_values}")
    print(f"{'='*70}")

    results = {
        'dataset': dataset_name,
        'num_repeats': NUM_REPEATS,
        'k_values': k_values
    }

    # ===== 1. Dataset Loading Time =====
    print(f"\n[1] Loading {dataset_name} dataset...")
    load_times = []
    data = None

    for i in range(NUM_REPEATS):
        start = time.perf_counter()
        data = load_func(cache_dir)
        load_time = time.perf_counter() - start
        load_times.append(load_time)
        print(f"    Run {i+1}: {load_time:.3f}s")

    results['load_time_s'] = {
        'mean': np.mean(load_times),
        'std': np.std(load_times),
        'min': np.min(load_times),
        'max': np.max(load_times)
    }

    num_nodes = data.num_nodes
    num_edges = data.num_edges
    results['num_nodes'] = num_nodes
    results['num_edges'] = num_edges

    print(f"    Nodes: {num_nodes:,}, Edges: {num_edges:,}")
    print(f"    Mean: {results['load_time_s']['mean']:.3f}s")

    # ===== 2. Adjacency List Building Time =====
    print(f"\n[2] Building adjacency list...")
    adj_times = []
    adj_list = None

    for i in range(NUM_REPEATS):
        start = time.perf_counter()
        adj_list = build_adjacency_list(data)
        adj_time = time.perf_counter() - start
        adj_times.append(adj_time)
        print(f"    Run {i+1}: {adj_time:.3f}s")

    results['adjacency_build_time_s'] = {
        'mean': np.mean(adj_times),
        'std': np.std(adj_times),
        'min': np.min(adj_times),
        'max': np.max(adj_times)
    }
    print(f"    Mean: {results['adjacency_build_time_s']['mean']:.3f}s")

    # ===== 3. METIS Partitioning Time =====
    if not PYMETIS_AVAILABLE:
        print("\n[3] METIS partitioning skipped (pymetis not available)")
        results['partition_times'] = {}
        return results

    print(f"\n[3] Running METIS partitioning...")
    results['partition_times'] = {}

    for k in k_values:
        print(f"\n    K={k}:")
        partition_times = []
        stats_times = []

        for i in range(NUM_REPEATS):
            # Partition time
            start = time.perf_counter()
            edge_cut, node_assignments = run_metis_partition(adj_list, k)
            partition_time = time.perf_counter() - start
            partition_times.append(partition_time)

            # Stats computation time
            start = time.perf_counter()
            total_halo = compute_partition_stats(data, node_assignments, adj_list, k)
            stats_time = time.perf_counter() - start
            stats_times.append(stats_time)

            print(f"      Run {i+1}: partition={partition_time:.3f}s, stats={stats_time:.3f}s")

        results['partition_times'][str(k)] = {
            'partition': {
                'mean': np.mean(partition_times),
                'std': np.std(partition_times),
                'min': np.min(partition_times),
                'max': np.max(partition_times)
            },
            'stats_computation': {
                'mean': np.mean(stats_times),
                'std': np.std(stats_times),
                'min': np.min(stats_times),
                'max': np.max(stats_times)
            },
            'total': {
                'mean': np.mean(partition_times) + np.mean(stats_times),
            },
            'edge_cut': int(edge_cut),
            'total_halo': int(total_halo)
        }

        print(f"      Mean: partition={np.mean(partition_times):.3f}s, stats={np.mean(stats_times):.3f}s")

    # ===== Summary =====
    total_time = results['load_time_s']['mean'] + results['adjacency_build_time_s']['mean']
    if k_values:
        k_example = str(k_values[-1])
        total_time += results['partition_times'][k_example]['total']['mean']

    results['total_pipeline_time_s'] = total_time

    print(f"\n[Summary] {dataset_name}")
    print(f"    Load:      {results['load_time_s']['mean']:.3f}s")
    print(f"    Adj Build: {results['adjacency_build_time_s']['mean']:.3f}s")
    if k_values and PYMETIS_AVAILABLE:
        print(f"    Partition (K={k_values[-1]}): {results['partition_times'][str(k_values[-1])]['total']['mean']:.3f}s")
    print(f"    Total:     {total_time:.3f}s")

    return results


def save_results(results: Dict, filename: str):
    """Save results to JSON"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Measure Graph Partition Time")
    parser.add_argument("--flickr", action="store_true", help="Measure Flickr")
    parser.add_argument("--reddit2", action="store_true", help="Measure Reddit2")
    parser.add_argument("--ogbn", action="store_true", help="Measure ogbn-products")
    parser.add_argument("--all", action="store_true", help="Measure all datasets")
    args = parser.parse_args()

    if args.all:
        args.flickr = True
        args.reddit2 = True
        args.ogbn = True

    if not any([args.flickr, args.reddit2, args.ogbn]):
        parser.print_help()
        return

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {DATA_DIR}")

    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'k_values_flickr': K_VALUES_FLICKR,
        'k_values_reddit2': K_VALUES_REDDIT2,
        'k_values_ogbn': K_VALUES_OGBN,
        'num_repeats': NUM_REPEATS,
        'datasets': {}
    }

    # Measure each dataset
    if args.flickr:
        results = measure_dataset_time('Flickr', load_flickr, DATA_DIR, K_VALUES_FLICKR)
        all_results['datasets']['flickr'] = results
        save_results(results, 'partition_time_flickr.json')

    if args.reddit2:
        results = measure_dataset_time('Reddit2', load_reddit2, DATA_DIR, K_VALUES_REDDIT2)
        all_results['datasets']['reddit2'] = results
        save_results(results, 'partition_time_reddit2.json')

    if args.ogbn:
        results = measure_dataset_time('ogbn-products', load_ogbn_products, DATA_DIR, K_VALUES_OGBN)
        all_results['datasets']['ogbn_products'] = results
        save_results(results, 'partition_time_ogbn.json')

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Dataset':<20} {'Nodes':<12} {'Edges':<15} {'Load(s)':<10} {'AdjBuild(s)':<12} {'Total(s)':<10}")
    print("-" * 90)

    for name, data in all_results['datasets'].items():
        print(f"{data['dataset']:<20} {data['num_nodes']:<12,} {data['num_edges']:<15,} "
              f"{data['load_time_s']['mean']:<10.3f} {data['adjacency_build_time_s']['mean']:<12.3f} "
              f"{data['total_pipeline_time_s']:<10.3f}")

    # Save combined results
    if len(all_results['datasets']) > 1:
        save_results(all_results, 'partition_time_all.json')


if __name__ == '__main__':
    main()
