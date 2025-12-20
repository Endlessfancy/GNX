"""
CPU-only Baseline Test for GraphSAGE on Flickr Dataset

This script runs the complete GraphSAGE model on CPU without any partitioning,
serving as a baseline for comparison with pipeline/data parallel approaches.
"""

import sys
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr
from torch_geometric.nn import SAGEConv
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class GraphSAGE(torch.nn.Module):
    """
    Standard GraphSAGE model with 2 layers

    Architecture:
    - Layer 1: input_dim (500) -> hidden_dim (256)
    - Layer 2: hidden_dim (256) -> output_dim (256)
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Layer 2
        x = self.conv2(x, edge_index)

        return x


def load_flickr_dataset():
    """Load Flickr dataset"""
    print("Loading Flickr dataset...")
    dataset = Flickr(root='/tmp/Flickr')
    data = dataset[0]

    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features: {data.num_features}")

    return data


def run_cpu_baseline(num_runs=10, warmup_runs=2):
    """
    Run GraphSAGE inference on CPU

    Args:
        num_runs: Number of inference runs for timing
        warmup_runs: Number of warmup runs (not counted in timing)

    Returns:
        Dictionary with timing results
    """
    print("="*80)
    print("CPU-only Baseline Test")
    print("="*80)
    print()

    # Load data
    data = load_flickr_dataset()

    # Create model
    print("\nInitializing GraphSAGE model...")
    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=256,
        out_channels=256
    )

    # Force CPU
    device = torch.device('cpu')
    model = model.to(device)
    data = data.to(device)

    model.eval()

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")

    # Warmup runs
    print(f"\nRunning {warmup_runs} warmup iterations...")
    with torch.no_grad():
        for i in range(warmup_runs):
            _ = model(data.x, data.edge_index)
            print(f"  Warmup {i+1}/{warmup_runs} completed")

    # Timed runs
    print(f"\nRunning {num_runs} timed iterations...")
    times = []

    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            output = model(data.x, data.edge_index)
            end_time = time.time()

            elapsed_ms = (end_time - start_time) * 1000
            times.append(elapsed_ms)
            print(f"  Run {i+1}/{num_runs}: {elapsed_ms:.2f}ms")

    # Calculate statistics
    times_tensor = torch.tensor(times)
    mean_time = times_tensor.mean().item()
    std_time = times_tensor.std().item()
    min_time = times_tensor.min().item()
    max_time = times_tensor.max().item()

    # Print results
    print("\n" + "="*80)
    print("Results")
    print("="*80)
    print(f"Output shape: {output.shape}")
    print(f"\nTiming Statistics ({num_runs} runs):")
    print(f"  Mean:   {mean_time:.2f} ms")
    print(f"  Std:    {std_time:.2f} ms")
    print(f"  Min:    {min_time:.2f} ms")
    print(f"  Max:    {max_time:.2f} ms")
    print()

    # Save results
    results = {
        'device': 'CPU',
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_features': data.num_features,
        'output_shape': list(output.shape),
        'num_runs': num_runs,
        'times_ms': times,
        'mean_time_ms': mean_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
    }

    return results


def save_results(results, output_file='cpu_baseline_results.txt'):
    """Save results to file"""
    output_path = Path(__file__).parent / output_file

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CPU-only Baseline Results\n")
        f.write("="*80 + "\n\n")

        f.write("Dataset:\n")
        f.write(f"  Name: Flickr\n")
        f.write(f"  Nodes: {results['num_nodes']:,}\n")
        f.write(f"  Edges: {results['num_edges']:,}\n")
        f.write(f"  Features: {results['num_features']}\n\n")

        f.write("Model:\n")
        f.write(f"  Architecture: GraphSAGE (2 layers)\n")
        f.write(f"  Hidden dim: 256\n")
        f.write(f"  Output dim: 256\n\n")

        f.write("Performance:\n")
        f.write(f"  Device: {results['device']}\n")
        f.write(f"  Number of runs: {results['num_runs']}\n")
        f.write(f"  Mean time: {results['mean_time_ms']:.2f} ms\n")
        f.write(f"  Std time: {results['std_time_ms']:.2f} ms\n")
        f.write(f"  Min time: {results['min_time_ms']:.2f} ms\n")
        f.write(f"  Max time: {results['max_time_ms']:.2f} ms\n\n")

        f.write("Individual Run Times (ms):\n")
        for i, t in enumerate(results['times_ms'], 1):
            f.write(f"  Run {i}: {t:.2f}\n")

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # Run baseline test
    results = run_cpu_baseline(num_runs=10, warmup_runs=2)

    # Save results
    save_results(results)

    print("\n" + "="*80)
    print("CPU Baseline Test Completed!")
    print("="*80)
