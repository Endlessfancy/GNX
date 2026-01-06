"""
CPU Batch-4096 Baseline Test for GraphSAGE (1 Layer) on Flickr Dataset
Using OpenVINO IR model for inference.

This script partitions the Flickr dataset into batches of 4096 target nodes,
finds 1-hop halo nodes (neighbors not in current batch), and runs GraphSAGE
first layer inference on each batch using OpenVINO. Measures total execution time.

For NPU: Uses HETERO:NPU,CPU mode for automatic fallback of unsupported operations.
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Flickr
from torch_geometric.utils import k_hop_subgraph
from pathlib import Path

# OpenVINO imports
import openvino as ov
from openvino.runtime import Core

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# ====================================================================
# GraphSAGE 7-Stage Modules (for ONNX export)
# ====================================================================

class SAGEStage1_Gather(torch.nn.Module):
    """Stage 1: GATHER - Neighbor feature gathering"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return x[edge_index[0]]


class SAGEStage2_Message(torch.nn.Module):
    """Stage 2: MESSAGE - Identity for mean aggregator"""
    def __init__(self):
        super().__init__()

    def forward(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


class SAGEStage3_ReduceSum(torch.nn.Module):
    """Stage 3: REDUCE_SUM - Sum aggregation using scatter_add for ONNX compatibility"""
    def __init__(self):
        super().__init__()

    def forward(self, messages: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        target_nodes = edge_index[1]
        feat_dim = messages.size(1)
        index_expanded = target_nodes.unsqueeze(1).expand(-1, feat_dim)
        out = torch.zeros(num_nodes, feat_dim, dtype=messages.dtype, device=messages.device)
        out = out.scatter_add(0, index_expanded, messages)
        return out


class SAGEStage4_ReduceCount(torch.nn.Module):
    """Stage 4: REDUCE_COUNT - Count neighbors using scatter_add for ONNX compatibility"""
    def __init__(self):
        super().__init__()

    def forward(self, edge_index: torch.Tensor, num_nodes: int, num_edges: int) -> torch.Tensor:
        target_nodes = edge_index[1]
        ones = torch.ones(num_edges, 1, device=edge_index.device)
        index_expanded = target_nodes.unsqueeze(1)
        count = torch.zeros(num_nodes, 1, device=edge_index.device)
        count = count.scatter_add(0, index_expanded, ones)
        return count.squeeze(1)


class SAGEStage5_Normalize(torch.nn.Module):
    """Stage 5: NORMALIZE - Compute mean"""
    def __init__(self):
        super().__init__()

    def forward(self, sum_agg: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        count = torch.clamp(count, min=1)
        mean_agg = sum_agg / count.unsqueeze(-1)
        return mean_agg


class SAGEStage6_Transform(torch.nn.Module):
    """Stage 6: TRANSFORM - Linear transformations"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.zeros_(self.lin_l.bias)
        nn.init.xavier_uniform_(self.lin_r.weight)

    def forward(self, mean_agg: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.lin_l(mean_agg) + self.lin_r(x)


class SAGEStage7_Activate(torch.nn.Module):
    """Stage 7: ACTIVATE - ReLU activation"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


class GraphSAGEFullModel(nn.Module):
    """
    Complete GraphSAGE 1-layer model (stages 1-7) for ONNX export.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.stage1 = SAGEStage1_Gather()
        self.stage2 = SAGEStage2_Message()
        self.stage3 = SAGEStage3_ReduceSum()
        self.stage4 = SAGEStage4_ReduceCount()
        self.stage5 = SAGEStage5_Normalize()
        self.stage6 = SAGEStage6_Transform(in_channels, out_channels)
        self.stage7 = SAGEStage7_Activate()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # Stage 1: Gather
        x_j = self.stage1(x, edge_index)

        # Stage 2: Message (identity)
        messages = self.stage2(x_j)

        # Stage 3: ReduceSum
        sum_agg = self.stage3(messages, edge_index, num_nodes)

        # Stage 4: ReduceCount
        count = self.stage4(edge_index, num_nodes, num_edges)

        # Stage 5: Normalize
        mean_agg = self.stage5(sum_agg, count)

        # Stage 6: Transform
        transformed = self.stage6(mean_agg, x)

        # Stage 7: Activate
        output = self.stage7(transformed)

        return output


def export_onnx_model(num_nodes: int, num_edges: int, num_features: int,
                      out_channels: int = 256, output_path: str = "graphsage_layer1.onnx",
                      static: bool = False):
    """
    Export GraphSAGE full model (stages 1-7) to ONNX format.
    Skips export if model already exists.
    """
    # Check if model already exists
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 1000:  # Valid model should be > 1KB
            shape_type = "static" if static else "dynamic"
            print(f"  ONNX model exists ({shape_type} shape): {output_path}")
            return output_path

    model = GraphSAGEFullModel(num_features, out_channels)
    model.eval()

    # Dummy inputs
    dummy_x = torch.randn(num_nodes, num_features)
    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Dynamic axes only for non-static export
    dynamic_axes = None if static else {
        'x': {0: 'num_nodes'},
        'edge_index': {1: 'num_edges'},
        'output': {0: 'num_nodes'}
    }

    # Export
    torch.onnx.export(
        model,
        (dummy_x, dummy_edge_index),
        output_path,
        input_names=['x', 'edge_index'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        verbose=False
    )

    shape_type = "static" if static else "dynamic"
    print(f"  ONNX model exported ({shape_type} shape): {output_path}")
    return output_path


def compile_openvino_model(onnx_path: str, device: str = "CPU"):
    """
    Compile ONNX model with OpenVINO.
    For NPU: automatically uses HETERO:NPU,CPU for fallback.
    """
    core = Core()
    model = core.read_model(onnx_path)

    if device == "NPU":
        # NPU: use HETERO plugin for automatic CPU fallback
        target_device = "HETERO:NPU,CPU"
        print(f"  Using HETERO mode: {target_device}")
    else:
        target_device = device

    compiled_model = core.compile_model(model, target_device)
    print(f"  OpenVINO model compiled for {target_device}")
    return compiled_model


def partition_graph_into_batches(data, batch_size=4096):
    """
    Partition graph nodes into batches with halo nodes.
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    all_nodes = torch.arange(num_nodes)
    batches = []
    num_batches = (num_nodes + batch_size - 1) // batch_size

    print(f"\nPartitioning {num_nodes:,} nodes into batches of {batch_size}...")
    print(f"  Expected batches: {num_batches}")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_nodes)

        target_nodes = all_nodes[start_idx:end_idx]
        target_set = set(target_nodes.tolist())

        # Find 1-hop neighbors
        subgraph_nodes, subgraph_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=target_nodes,
            num_hops=1,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes
        )

        # Identify halo nodes
        subgraph_set = set(subgraph_nodes.tolist())
        halo_nodes = torch.tensor([n for n in subgraph_set if n not in target_set])

        # Create mask for target nodes
        target_mask = torch.zeros(len(subgraph_nodes), dtype=torch.bool)
        target_mask[mapping] = True

        # Extract features
        subgraph_x = data.x[subgraph_nodes]

        batch_info = {
            'batch_idx': batch_idx,
            'target_nodes': target_nodes,
            'halo_nodes': halo_nodes,
            'subgraph_nodes': subgraph_nodes,
            'subgraph_edge_index': subgraph_edge_index,
            'subgraph_x': subgraph_x,
            'target_mask': target_mask,
            'mapping': mapping,
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


def run_openvino_inference(compiled_model, x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """
    Run inference with OpenVINO compiled model.
    """
    infer_request = compiled_model.create_infer_request()

    # Set inputs
    infer_request.set_tensor('x', ov.Tensor(x.astype(np.float32)))
    infer_request.set_tensor('edge_index', ov.Tensor(edge_index.astype(np.int64)))

    # Inference
    infer_request.infer()

    # Get output
    output = infer_request.get_output_tensor(0).data.copy()

    return output


def run_batch_baseline(batch_size=4096, num_runs=10, warmup_runs=2, device="CPU"):
    """
    Run GraphSAGE inference with OpenVINO using batch partitioning.
    For NPU: automatically uses HETERO:NPU,CPU for fallback.
    """
    print("="*80)
    print(f"{device} Batch-{batch_size} Baseline Test (1 Layer, 1-hop) - OpenVINO")
    if device == "NPU":
        print("  (Using HETERO:NPU,CPU for automatic fallback)")
    print("="*80)
    print()

    # Load data
    data = load_flickr_dataset()

    # Move data to CPU
    data = data.to('cpu')

    # Partition graph first
    batches = partition_graph_into_batches(data, batch_size=batch_size)
    num_batches = len(batches)

    is_npu = (device == "NPU")

    # Create model directory
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)

    if is_npu:
        # NPU: export static shape model for each batch
        print(f"\nExporting {num_batches} static ONNX models for NPU...")
        compiled_models = []

        for batch in batches:
            batch_idx = batch['batch_idx']
            num_nodes = batch['num_subgraph_nodes']
            num_edges = batch['num_subgraph_edges']

            onnx_path = model_dir / f"graphsage_npu_batch{batch_idx}_n{num_nodes}_e{num_edges}.onnx"

            # Export static model for this batch
            export_onnx_model(
                num_nodes=num_nodes,
                num_edges=num_edges,
                num_features=data.num_features,
                out_channels=256,
                output_path=str(onnx_path),
                static=True
            )

            # Compile with OpenVINO (HETERO:NPU,CPU)
            compiled_model = compile_openvino_model(str(onnx_path), device)
            compiled_models.append(compiled_model)

        print(f"  All {num_batches} models compiled")

    else:
        # CPU/GPU: single dynamic shape model
        print("\nExporting ONNX model (dynamic shape)...")
        onnx_path = model_dir / f"graphsage_{device.lower()}_dynamic.onnx"

        export_onnx_model(
            num_nodes=10000,
            num_edges=50000,
            num_features=data.num_features,
            out_channels=256,
            output_path=str(onnx_path),
            static=False
        )

        # Compile with OpenVINO
        print(f"\nCompiling OpenVINO model for {device}...")
        compiled_model = compile_openvino_model(str(onnx_path), device)
        compiled_models = [compiled_model] * num_batches

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
    for warmup_i in range(warmup_runs):
        for i, batch in enumerate(batches):
            x_np = batch['subgraph_x'].numpy()
            edge_index_np = batch['subgraph_edge_index'].numpy()
            _ = run_openvino_inference(compiled_models[i], x_np, edge_index_np)
        print(f"  Warmup {warmup_i+1}/{warmup_runs} completed")

    # Timed runs
    print(f"\nRunning {num_runs} timed iterations...")
    total_times = []
    batch_times_all = []

    for run_i in range(num_runs):
        batch_times = []
        total_start = time.time()

        # Collect outputs for all target nodes
        all_outputs = np.zeros((data.num_nodes, 256), dtype=np.float32)

        for i, batch in enumerate(batches):
            batch_start = time.time()

            # Prepare inputs
            x_np = batch['subgraph_x'].numpy()
            edge_index_np = batch['subgraph_edge_index'].numpy()

            # Run inference
            output = run_openvino_inference(compiled_models[i], x_np, edge_index_np)

            # Extract target node outputs
            target_mask = batch['target_mask'].numpy()
            target_output = output[target_mask]

            # Store in full output tensor
            target_nodes = batch['target_nodes'].numpy()
            all_outputs[target_nodes] = target_output

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

    # Per-batch statistics
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
        'device': device,
        'runtime': 'OpenVINO',
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
        f.write(f"{results['device']} Batch-{results['batch_size']} Baseline Results (1 Layer, 1-hop) - OpenVINO\n")
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
        f.write(f"  Hidden dim: 256\n")
        f.write(f"  Runtime: {results['runtime']}\n\n")

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
    import argparse

    parser = argparse.ArgumentParser(description='Batch-4096 Baseline with OpenVINO')
    parser.add_argument('--device', type=str, default='CPU',
                        choices=['CPU', 'GPU', 'NPU'],
                        help='OpenVINO device (CPU, GPU, or NPU)')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size (target nodes per batch)')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of timed runs')
    parser.add_argument('--warmup_runs', type=int, default=2,
                        help='Number of warmup runs')
    args = parser.parse_args()

    # Run batch baseline test
    results = run_batch_baseline(
        batch_size=args.batch_size,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        device=args.device
    )

    # Save results
    output_file = f"{args.device.lower()}_batch_{args.batch_size}_results.txt"
    save_results(results, output_file)

    print("\n" + "="*80)
    print(f"{args.device} Batch-{args.batch_size} Baseline Test Completed!")
    print("="*80)
