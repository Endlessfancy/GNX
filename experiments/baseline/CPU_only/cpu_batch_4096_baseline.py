"""
CPU Batch-4096 Baseline Test for GraphSAGE (1 Layer) on Flickr Dataset

This script partitions the Flickr dataset into batches of 4096 target nodes,
finds 1-hop halo nodes (neighbors not in current batch), and runs GraphSAGE
first layer inference on each batch sequentially. Measures total execution time.
"""

import sys
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import k_hop_subgraph
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model with 1 layer only
    """
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x


def partition_graph_into_batches(data, batch_size=4096):
    """
    Partition graph nodes into batches with halo nodes.

    For each batch:
    - Target nodes: batch_size nodes (or remaining nodes for last batch)
    - Halo nodes: neighbors of target nodes that are NOT in target nodes

    Args:
        data: PyG Data object
        batch_size: Number of target nodes per batch (default: 4096)

    Returns:
        List of dictionaries containing batch information:
        - target_nodes: tensor of target node indices
        - halo_nodes: tensor of halo node indices
        - subgraph_nodes: all nodes in subgraph (target + halo)
        - subgraph_edge_index: edge index for subgraph
        - subgraph_x: features for subgraph nodes
        - target_mask: boolean mask for target nodes in subgraph
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # Create node order (just sequential for simplicity)
    all_nodes = torch.arange(num_nodes)

    # Split into batches
    batches = []
    num_batches = (num_nodes + batch_size - 1) // batch_size

    print(f"\nPartitioning {num_nodes:,} nodes into batches of {batch_size}...")
    print(f"  Expected batches: {num_batches}")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_nodes)

        # Target nodes for this batch
        target_nodes = all_nodes[start_idx:end_idx]
        target_set = set(target_nodes.tolist())

        # Find 1-hop neighbors (since we have 1 GNN layer)
        # k_hop_subgraph returns (subset, edge_index, mapping, edge_mask)
        subgraph_nodes, subgraph_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=target_nodes,
            num_hops=1,  # 1 layer needs 1-hop neighbors
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes
        )

        # Identify halo nodes (in subgraph but not in target)
        subgraph_set = set(subgraph_nodes.tolist())
        halo_nodes = torch.tensor([n for n in subgraph_set if n not in target_set])

        # Create mask for target nodes in the subgraph
        # mapping gives the new indices of target nodes in subgraph
        target_mask = torch.zeros(len(subgraph_nodes), dtype=torch.bool)
        target_mask[mapping] = True

        # Extract features for subgraph
        subgraph_x = data.x[subgraph_nodes]

        batch_info = {
            'batch_idx': batch_idx,
            'target_nodes': target_nodes,
            'halo_nodes': halo_nodes,
            'subgraph_nodes': subgraph_nodes,
            'subgraph_edge_index': subgraph_edge_index,
            'subgraph_x': subgraph_x,
            'target_mask': target_mask,
            'mapping': mapping,  # Maps original target indices to subgraph indices
            'num_target': len(target_nodes),
            'num_halo': len(halo_nodes),
            'num_subgraph_nodes': len(subgraph_nodes),
            'num_subgraph_edges': subgraph_edge_index.size(1),
        }
        batches.append(batch_info)

        print(f"  Batch {batch_idx + 1}/{num_batches}: "
              f"target={len(target_nodes):,}, halo={len(halo_nodes):,}, "
              f"total={len(subgraph_nodes):,}, edges={subgraph_edge_index.size(1):,}")

    return batches


def load_flickr_dataset():
    """Load Flickr dataset"""
    print("Loading Flickr dataset...")
    dataset = Flickr(root='/tmp/Flickr')
    data = dataset[0]

    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features: {data.num_features}")

    return data


def run_batch_baseline(batch_size=4096, num_runs=10, warmup_runs=2):
    """
    Run GraphSAGE inference on CPU with batch partitioning

    Args:
        batch_size: Number of target nodes per batch
        num_runs: Number of full inference runs for timing
        warmup_runs: Number of warmup runs

    Returns:
        Dictionary with timing results
    """
    print("="*80)
    print(f"CPU Batch-{batch_size} Baseline Test (1 Layer, 1-hop)")
    print("="*80)
    print()

    # Load data
    data = load_flickr_dataset()

    # Create model (1 layer only)
    print("\nInitializing GraphSAGE model (1 layer)...")
    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=256
    )

    # Force CPU
    device = torch.device('cpu')
    model = model.to(device)
    data = data.to(device)

    model.eval()

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")

    # Partition graph into batches
    batches = partition_graph_into_batches(data, batch_size=batch_size)
    num_batches = len(batches)

    # Print partition summary
    total_target = sum(b['num_target'] for b in batches)
    total_halo = sum(b['num_halo'] for b in batches)
    total_subgraph = sum(b['num_subgraph_nodes'] for b in batches)
    print(f"\nPartition Summary:")
    print(f"  Total batches: {num_batches}")
    print(f"  Total target nodes: {total_target:,}")
    print(f"  Total halo nodes (with duplicates): {total_halo:,}")
    print(f"  Total subgraph nodes (with duplicates): {total_subgraph:,}")
    print(f"  Expansion ratio: {total_subgraph / total_target:.2f}x")

    # Warmup runs
    print(f"\nRunning {warmup_runs} warmup iterations...")
    with torch.no_grad():
        for warmup_i in range(warmup_runs):
            for batch in batches:
                _ = model(batch['subgraph_x'], batch['subgraph_edge_index'])
            print(f"  Warmup {warmup_i+1}/{warmup_runs} completed")

    # Timed runs
    print(f"\nRunning {num_runs} timed iterations...")
    total_times = []
    batch_times_all = []

    with torch.no_grad():
        for run_i in range(num_runs):
            batch_times = []
            total_start = time.time()

            # Collect outputs for all target nodes
            all_outputs = torch.zeros(data.num_nodes, 256, device=device)

            for batch in batches:
                batch_start = time.time()

                # Run inference on subgraph
                output = model(batch['subgraph_x'], batch['subgraph_edge_index'])

                # Extract only target node outputs
                target_output = output[batch['target_mask']]

                # Store in full output tensor
                all_outputs[batch['target_nodes']] = target_output

                batch_end = time.time()
                batch_times.append((batch_end - batch_start) * 1000)

            total_end = time.time()
            total_elapsed_ms = (total_end - total_start) * 1000
            total_times.append(total_elapsed_ms)
            batch_times_all.append(batch_times)

            print(f"  Run {run_i+1}/{num_runs}: {total_elapsed_ms:.2f}ms "
                  f"(avg per batch: {sum(batch_times)/len(batch_times):.2f}ms)")

    # Calculate statistics
    times_tensor = torch.tensor(total_times)
    mean_time = times_tensor.mean().item()
    std_time = times_tensor.std().item()
    min_time = times_tensor.min().item()
    max_time = times_tensor.max().item()

    # Calculate per-batch statistics
    avg_batch_times = [sum(bt)/len(bt) for bt in batch_times_all]
    mean_batch_time = sum(avg_batch_times) / len(avg_batch_times)

    # Print results
    print("\n" + "="*80)
    print("Results")
    print("="*80)
    print(f"Output shape: {all_outputs.shape}")
    print(f"\nTotal Timing Statistics ({num_runs} runs):")
    print(f"  Mean:   {mean_time:.2f} ms")
    print(f"  Std:    {std_time:.2f} ms")
    print(f"  Min:    {min_time:.2f} ms")
    print(f"  Max:    {max_time:.2f} ms")
    print(f"\nPer-Batch Statistics:")
    print(f"  Number of batches: {num_batches}")
    print(f"  Average time per batch: {mean_batch_time:.2f} ms")
    print()

    # Save results
    results = {
        'device': 'CPU',
        'batch_size': batch_size,
        'num_batches': num_batches,
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_features': data.num_features,
        'output_shape': list(all_outputs.shape),
        'num_runs': num_runs,
        'total_times_ms': total_times,
        'mean_time_ms': mean_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'mean_batch_time_ms': mean_batch_time,
        'batches_info': [
            {
                'batch_idx': b['batch_idx'],
                'num_target': b['num_target'],
                'num_halo': b['num_halo'],
                'num_subgraph_nodes': b['num_subgraph_nodes'],
                'num_subgraph_edges': b['num_subgraph_edges'],
            }
            for b in batches
        ],
    }

    return results


def save_results(results, output_file='cpu_batch_4096_results.txt'):
    """Save results to file"""
    output_path = Path(__file__).parent / output_file

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"CPU Batch-{results['batch_size']} Baseline Results (1 Layer, 1-hop)\n")
        f.write("="*80 + "\n\n")

        f.write("Dataset:\n")
        f.write(f"  Name: Flickr\n")
        f.write(f"  Nodes: {results['num_nodes']:,}\n")
        f.write(f"  Edges: {results['num_edges']:,}\n")
        f.write(f"  Features: {results['num_features']}\n\n")

        f.write("Batch Configuration:\n")
        f.write(f"  Batch size (target nodes): {results['batch_size']}\n")
        f.write(f"  Number of batches: {results['num_batches']}\n\n")

        f.write("Model:\n")
        f.write(f"  Architecture: GraphSAGE (1 layer)\n")
        f.write(f"  Hidden dim: 256\n\n")

        f.write("Performance:\n")
        f.write(f"  Device: {results['device']}\n")
        f.write(f"  Number of runs: {results['num_runs']}\n")
        f.write(f"  Mean total time: {results['mean_time_ms']:.2f} ms\n")
        f.write(f"  Std time: {results['std_time_ms']:.2f} ms\n")
        f.write(f"  Min time: {results['min_time_ms']:.2f} ms\n")
        f.write(f"  Max time: {results['max_time_ms']:.2f} ms\n")
        f.write(f"  Mean per-batch time: {results['mean_batch_time_ms']:.2f} ms\n\n")

        f.write("Individual Run Times (ms):\n")
        for i, t in enumerate(results['total_times_ms'], 1):
            f.write(f"  Run {i}: {t:.2f}\n")

        f.write("\nBatch Details:\n")
        for b in results['batches_info']:
            f.write(f"  Batch {b['batch_idx']+1}: "
                   f"target={b['num_target']}, halo={b['num_halo']}, "
                   f"total={b['num_subgraph_nodes']}, edges={b['num_subgraph_edges']}\n")

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # Run batch baseline test
    results = run_batch_baseline(batch_size=4096, num_runs=10, warmup_runs=2)

    # Save results
    save_results(results)

    print("\n" + "="*80)
    print("CPU Batch-4096 Baseline Test (1 Layer, 1-hop) Completed!")
    print("="*80)
