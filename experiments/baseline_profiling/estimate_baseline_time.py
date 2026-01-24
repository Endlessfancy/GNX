#!/usr/bin/env python3
"""
Baseline Time Estimation - Single Device Sequential Execution

Estimates total inference time for complete 1-layer GNN models running
on a single device (CPU or GPU) without pipeline parallelism.

Reads:
- Subgraph partition data from graphPartition/*.json
- Baseline profiling results from results/cpugpu_results.json

Uses 2D interpolation (nodes, edges) to estimate latency for each subgraph.

Usage:
    python estimate_baseline_time.py                    # All datasets, all models
    python estimate_baseline_time.py --dataset reddit2  # Single dataset
    python estimate_baseline_time.py --model graphsage  # Single model
    python estimate_baseline_time.py --k 10             # Specific partition size
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'results'
PARTITION_DIR = SCRIPT_DIR.parent / 'graphPartition'

# Partition files
PARTITION_FILES = {
    'flickr': PARTITION_DIR / 'partition_results.json',
    'reddit2': PARTITION_DIR / 'reddit2_partition_results.json',
    'ogbn_products': PARTITION_DIR / 'ogbn_products_partition_results.json',
}

# Models
MODEL_NAMES = ['graphsage', 'gcn', 'gat']

# ============================================================================
# Data Loading
# ============================================================================

def load_profiling_results():
    """Load baseline profiling results"""
    results = {}

    # Try to load cpugpu_results.json first
    cpugpu_file = RESULTS_DIR / 'cpugpu_results.json'
    if cpugpu_file.exists():
        with open(cpugpu_file, 'r') as f:
            results.update(json.load(f))

    # Also load individual model results if available
    for model in MODEL_NAMES:
        model_file = RESULTS_DIR / f'{model}_cpugpu.json'
        if model_file.exists():
            with open(model_file, 'r') as f:
                results.update(json.load(f))

    return results


def load_partition_data(dataset_name):
    """Load partition data for a dataset"""
    filepath = PARTITION_FILES.get(dataset_name.lower())
    if not filepath or not filepath.exists():
        print(f"Partition file not found: {filepath}")
        return None

    with open(filepath, 'r') as f:
        return json.load(f)


# ============================================================================
# Interpolation Model
# ============================================================================

class LatencyInterpolator:
    """Latency estimation using nearest neighbor interpolation with linear scaling.

    For each query (nodes, edges), find similar data points and interpolate.
    This is more robust to noisy data than pure regression.
    """

    def __init__(self, profiling_results, model_name, device):
        self.model_name = model_name
        self.device = device
        self.data = []

        # Extract data points for this model and device
        for key, result in profiling_results.items():
            parts = key.split(',')
            if len(parts) != 4:
                continue

            m, n, e, d = parts
            if m != model_name or d != device:
                continue

            if result.get('failed', True):
                continue

            self.data.append({
                'nodes': int(n),
                'edges': int(e),
                'time': result['mean']
            })

        if len(self.data) < 3:
            print(f"  Warning: Not enough data for {model_name}/{device}")
            self.model = None
            return

        # Store data range
        nodes = np.array([d['nodes'] for d in self.data])
        edges = np.array([d['edges'] for d in self.data])

        self.min_nodes = nodes.min()
        self.max_nodes = nodes.max()
        self.min_edges = edges.min()
        self.max_edges = edges.max()
        self.num_points = len(self.data)

        # Compute average time per edge (for scaling)
        times = np.array([d['time'] for d in self.data])
        self.avg_time_per_edge = np.median(times / edges)

        self.model = True  # Mark as ready

    def predict(self, nodes, edges):
        """Predict latency using weighted nearest neighbor interpolation"""
        if self.model is None:
            return -1

        # Find closest data points based on normalized distance
        distances = []
        for d in self.data:
            # Normalize by range to make node and edge distances comparable
            node_dist = abs(d['nodes'] - nodes) / max(1, self.max_nodes - self.min_nodes)
            edge_dist = abs(d['edges'] - edges) / max(1, self.max_edges - self.min_edges)
            dist = np.sqrt(node_dist**2 + edge_dist**2)
            distances.append((dist, d))

        # Sort by distance and take top 3
        distances.sort(key=lambda x: x[0])
        nearest = distances[:3]

        # Weighted average (inverse distance weighting)
        total_weight = 0
        weighted_time = 0

        for dist, d in nearest:
            # Scale the reference time based on edge ratio
            # Time scales roughly linearly with edges for GNN
            edge_ratio = edges / max(1, d['edges'])
            scaled_time = d['time'] * edge_ratio

            # Weight by inverse distance (add small epsilon to avoid div by zero)
            weight = 1 / (dist + 0.01)
            weighted_time += weight * scaled_time
            total_weight += weight

        result = weighted_time / total_weight if total_weight > 0 else 1.0
        return max(1.0, result)

    def get_info(self):
        """Return model info for debugging"""
        if self.model is None:
            return "No model"
        return f"KNN interpolation from {self.num_points} points"


# ============================================================================
# Time Estimation
# ============================================================================

def estimate_dataset_time(dataset_name, partition_data, profiling_results,
                          model_name='graphsage', k=None, device='CPU'):
    """Estimate total time for a dataset on a single device"""

    # Build interpolator
    interpolator = LatencyInterpolator(profiling_results, model_name, device)

    if interpolator.model is None:
        return None

    results = []

    for partition in partition_data['partitions']:
        if k is not None and partition['k'] != k:
            continue

        k_val = partition['k']
        subgraphs = partition['subgraphs']

        # Calculate time for each subgraph
        subgraph_times = []
        for sg in subgraphs:
            nodes = sg['total_nodes']
            edges = sg['internal_edges']
            t = interpolator.predict(nodes, edges)
            subgraph_times.append({
                'id': sg['id'],
                'nodes': nodes,
                'edges': edges,
                'time_ms': t
            })

        # Total time = sum of all subgraph times (sequential execution)
        total_time = sum(s['time_ms'] for s in subgraph_times)

        results.append({
            'k': k_val,
            'num_subgraphs': len(subgraphs),
            'total_time_ms': total_time,
            'avg_time_ms': total_time / len(subgraphs),
            'subgraphs': subgraph_times
        })

    return results


def print_summary(dataset_name, results, model_name, device):
    """Print summary table"""
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name.upper()} | Model: {model_name.upper()} | Device: {device}")
    print(f"{'='*70}")

    print(f"\n{'K':<4} | {'Subgraphs':<10} | {'Total Time (ms)':<15} | {'Avg/Subgraph':<12} | {'Throughput':<12}")
    print("-" * 70)

    for r in results:
        throughput = 1000 / r['total_time_ms'] if r['total_time_ms'] > 0 else 0
        print(f"{r['k']:<4} | {r['num_subgraphs']:<10} | {r['total_time_ms']:<15.2f} | "
              f"{r['avg_time_ms']:<12.2f} | {throughput:.4f} infer/s")

    # Find optimal K (minimum total time)
    best = min(results, key=lambda x: x['total_time_ms'])
    print(f"\nOptimal K={best['k']}: {best['total_time_ms']:.2f} ms total")


def save_results(dataset_name, results, model_name, device, partition_data):
    """Save all K results to JSON file"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Find optimal K
    best = min(results, key=lambda x: x['total_time_ms'])

    # Build output data structure
    output = {
        'dataset': dataset_name,
        'model': model_name,
        'device': device,
        'total_nodes': partition_data['num_nodes'],
        'total_edges': partition_data['num_edges'],
        'optimal_k': best['k'],
        'optimal_total_time_ms': best['total_time_ms'],
        'all_k_results': []
    }

    # Add all K results
    for r in results:
        k_result = {
            'k': r['k'],
            'num_subgraphs': r['num_subgraphs'],
            'total_time_ms': r['total_time_ms'],
            'avg_time_ms': r['avg_time_ms'],
            'throughput_per_sec': 1000 / r['total_time_ms'] if r['total_time_ms'] > 0 else 0,
            'subgraphs': r['subgraphs']
        }
        output['all_k_results'].append(k_result)

    # Save to file
    output_file = RESULTS_DIR / f'baseline_time_{dataset_name}_{model_name}_{device}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Also save a summary of optimal K separately
    optimal_file = RESULTS_DIR / f'baseline_optimal_{dataset_name}_{model_name}_{device}.json'
    optimal_output = {
        'dataset': dataset_name,
        'model': model_name,
        'device': device,
        'optimal_k': best['k'],
        'optimal_total_time_ms': best['total_time_ms'],
        'optimal_avg_time_ms': best['avg_time_ms'],
        'optimal_throughput_per_sec': 1000 / best['total_time_ms'] if best['total_time_ms'] > 0 else 0,
        'subgraphs': best['subgraphs']
    }
    with open(optimal_file, 'w', encoding='utf-8') as f:
        json.dump(optimal_output, f, indent=2)

    print(f"Optimal K results saved to: {optimal_file}")

    return output


def print_detailed(dataset_name, results, model_name, device, k):
    """Print detailed breakdown for specific K"""
    result = next((r for r in results if r['k'] == k), None)
    if not result:
        print(f"No results for K={k}")
        return

    print(f"\n{'='*70}")
    print(f"Detailed Breakdown: {dataset_name.upper()} K={k}")
    print(f"Model: {model_name.upper()} | Device: {device}")
    print(f"{'='*70}")

    print(f"\n{'ID':<4} | {'Nodes':<10} | {'Edges':<12} | {'Time (ms)':<12}")
    print("-" * 50)

    for sg in result['subgraphs']:
        print(f"{sg['id']:<4} | {sg['nodes']:<10} | {sg['edges']:<12} | {sg['time_ms']:<12.2f}")

    print("-" * 50)
    print(f"{'Total':<4} | {'':<10} | {'':<12} | {result['total_time_ms']:<12.2f}")


def compare_all_models(dataset_name, partition_data, profiling_results, k=10):
    """Compare all models and devices for a specific K"""
    print(f"\n{'='*80}")
    print(f"Model Comparison: {dataset_name.upper()} K={k}")
    print(f"{'='*80}")

    print(f"\n{'Model':<12} | {'CPU (ms)':<12} | {'GPU (ms)':<12} | {'Speedup':<10}")
    print("-" * 55)

    for model in MODEL_NAMES:
        cpu_results = estimate_dataset_time(dataset_name, partition_data,
                                            profiling_results, model, k, 'CPU')
        gpu_results = estimate_dataset_time(dataset_name, partition_data,
                                            profiling_results, model, k, 'GPU')

        cpu_time = cpu_results[0]['total_time_ms'] if cpu_results else -1
        gpu_time = gpu_results[0]['total_time_ms'] if gpu_results else -1

        speedup = cpu_time / gpu_time if gpu_time > 0 and cpu_time > 0 else 0

        cpu_str = f"{cpu_time:.2f}" if cpu_time > 0 else "N/A"
        gpu_str = f"{gpu_time:.2f}" if gpu_time > 0 else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"

        print(f"{model.upper():<12} | {cpu_str:<12} | {gpu_str:<12} | {speedup_str:<10}")


def generate_csv_report(profiling_results, output_file=None):
    """Generate comprehensive CSV report for all datasets"""
    if output_file is None:
        output_file = RESULTS_DIR / 'baseline_time_estimation.csv'

    lines = ['dataset,k,model,device,total_time_ms,num_subgraphs,avg_time_ms']

    for dataset_name in PARTITION_FILES.keys():
        partition_data = load_partition_data(dataset_name)
        if not partition_data:
            continue

        for model in MODEL_NAMES:
            for device in ['CPU', 'GPU']:
                results = estimate_dataset_time(dataset_name, partition_data,
                                               profiling_results, model, None, device)
                if not results:
                    continue

                for r in results:
                    lines.append(f"{dataset_name},{r['k']},{model},{device},"
                               f"{r['total_time_ms']:.2f},{r['num_subgraphs']},"
                               f"{r['avg_time_ms']:.2f}")

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nCSV report saved to: {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Baseline Time Estimation')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['flickr', 'reddit2', 'ogbn_products'],
                       help='Dataset to analyze (default: all)')
    parser.add_argument('--model', type=str, default='graphsage',
                       choices=MODEL_NAMES,
                       help='Model to use (default: graphsage)')
    parser.add_argument('--device', type=str, default='CPU',
                       choices=['CPU', 'GPU'],
                       help='Device to estimate (default: CPU)')
    parser.add_argument('--k', type=int, default=None,
                       help='Specific partition K value (default: all)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all models for K=10')
    parser.add_argument('--csv', action='store_true',
                       help='Generate CSV report for all datasets')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed subgraph breakdown')
    parser.add_argument('--save', action='store_true',
                       help='Save results to JSON files')
    parser.add_argument('--save-all', action='store_true',
                       help='Save results for all datasets, models, and devices')

    args = parser.parse_args()

    # Load profiling results
    print("Loading profiling results...")
    profiling_results = load_profiling_results()

    if not profiling_results:
        print("ERROR: No profiling results found. Run profiling first.")
        return

    print(f"Loaded {len(profiling_results)} profiling data points")

    # Generate CSV if requested
    if args.csv:
        generate_csv_report(profiling_results)
        return

    # Save all results for all datasets, models, and devices
    if args.save_all:
        print("\nSaving results for all datasets, models, and devices...")
        all_results = []
        for dataset_name in PARTITION_FILES.keys():
            partition_data = load_partition_data(dataset_name)
            if not partition_data:
                continue

            for model in MODEL_NAMES:
                for device in ['CPU', 'GPU']:
                    results = estimate_dataset_time(dataset_name, partition_data,
                                                   profiling_results, model, None, device)
                    if results:
                        output = save_results(dataset_name, results, model, device, partition_data)
                        all_results.append(output)

        # Save combined summary
        summary_file = RESULTS_DIR / 'baseline_all_results_summary.json'
        summary = []
        for r in all_results:
            summary.append({
                'dataset': r['dataset'],
                'model': r['model'],
                'device': r['device'],
                'optimal_k': r['optimal_k'],
                'optimal_total_time_ms': r['optimal_total_time_ms']
            })
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\nCombined summary saved to: {summary_file}")
        return

    # Determine datasets to process
    datasets = [args.dataset] if args.dataset else list(PARTITION_FILES.keys())

    for dataset_name in datasets:
        partition_data = load_partition_data(dataset_name)
        if not partition_data:
            continue

        print(f"\nDataset: {partition_data['dataset']}")
        print(f"  Total nodes: {partition_data['num_nodes']:,}")
        print(f"  Total edges: {partition_data['num_edges']:,}")

        # Compare all models
        if args.compare:
            k_val = args.k if args.k else 10
            compare_all_models(dataset_name, partition_data, profiling_results, k_val)
            continue

        # Estimate time
        results = estimate_dataset_time(dataset_name, partition_data, profiling_results,
                                       args.model, args.k, args.device)

        if not results:
            print(f"  No results for {args.model}/{args.device}")
            continue

        # Print summary
        print_summary(dataset_name, results, args.model, args.device)

        # Save results if requested
        if args.save:
            save_results(dataset_name, results, args.model, args.device, partition_data)

        # Print detailed if requested
        if args.detailed and args.k:
            print_detailed(dataset_name, results, args.model, args.device, args.k)


if __name__ == '__main__':
    main()
