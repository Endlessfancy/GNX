#!/usr/bin/env python3
"""
GCN NPU Single Test Script - Tests one (nodes, stage) combination with all edges

GCN 6-Stage Decomposition:
    Stage 1: COMPUTE_NORM - NPU SKIP (scatter)
    Stage 2: GATHER       - NPU supported
    Stage 3: MESSAGE      - NPU supported
    Stage 4: REDUCE_SUM   - NPU SKIP (scatter)
    Stage 5: TRANSFORM    - NPU supported
    Stage 6: ACTIVATE     - NPU supported

Usage:
    python gcn_profile_npu.py --nodes 1000 --stage 2
    python gcn_profile_npu.py --nodes 1000 --stage 2 --edges 2000,5000,10000

Exit codes:
    0 - All tests passed
    1 - Some tests failed (partial success)
    2 - Device lost / fatal error (NPU in bad state)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Paths
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "gcn_exported_models"
RESULTS_DIR = SCRIPT_DIR / "gcn_results"
CHECKPOINT_DIR = RESULTS_DIR

# GCN NPU supported stages (skip 1, 4 - scatter operations)
NPU_SUPPORTED_STAGES = [2, 3, 5, 6]

# Test cases from config
def load_test_cases():
    config_path = SCRIPT_DIR / "test_cases.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def get_edges_for_nodes(test_cases, nodes):
    """Get all edge counts for a given node count"""
    edges = []
    for case in test_cases['test_cases']:
        if case['nodes'] == nodes:
            edges.append(case['edges'])
    return sorted(edges)

def generate_dummy_input(stage_id, num_nodes, num_edges, feature_dim=500):
    """Generate dummy input for a GCN stage"""
    torch.manual_seed(42)

    if stage_id == 2:
        # Stage 2: GATHER - x[N, F], edge_index[2, E]
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        return (x, edge_index)
    elif stage_id == 3:
        # Stage 3: MESSAGE - x_j[E, F], norm[E]
        x_j = torch.randn(num_edges, feature_dim)
        norm = torch.rand(num_edges) + 0.1  # avoid zero
        return (x_j, norm)
    elif stage_id == 5:
        # Stage 5: TRANSFORM - agg[N, F]
        agg = torch.randn(num_nodes, feature_dim)
        return (agg,)
    elif stage_id == 6:
        # Stage 6: ACTIVATE - out[N, F']
        out = torch.randn(num_nodes, feature_dim)
        return (out,)
    else:
        raise ValueError(f"Stage {stage_id} not supported for NPU (scatter operation)")

def measure_latency_npu(ir_path, dummy_input, num_warmup=10, num_iterations=50):
    """Measure NPU latency using async API (includes CPU<->NPU data transfer time)"""
    try:
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(ir_path))
        compiled_model = core.compile_model(model, 'NPU')

        # Prepare inputs
        inputs = [t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
                  for t in dummy_input]

        # Create infer request for async inference
        infer_request = compiled_model.create_infer_request()

        # Warmup (set tensor each time to simulate real scenario)
        for _ in range(num_warmup):
            for i in range(len(inputs)):
                infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))
            infer_request.start_async()
            infer_request.wait()

        # Measure using async API (set tensor each time to include CPU->NPU transfer)
        latencies = []
        for _ in range(num_iterations):
            # Re-set input tensors to trigger data transfer
            for i in range(len(inputs)):
                infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))

            start = time.perf_counter()
            infer_request.start_async()
            infer_request.wait()
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'failed': False
        }

    except Exception as e:
        error_msg = str(e)
        is_device_lost = 'DEVICE_LOST' in error_msg
        is_oom = 'OUT_OF_HOST_MEMORY' in error_msg or 'OUT_OF_DEVICE_MEMORY' in error_msg

        return {
            'mean': -1,
            'std': -1,
            'min': -1,
            'max': -1,
            'failed': True,
            'error': error_msg,
            'device_lost': is_device_lost,
            'oom': is_oom
        }

def main():
    parser = argparse.ArgumentParser(description='GCN NPU Single Test')
    parser.add_argument('--nodes', type=int, required=True, help='Number of nodes')
    parser.add_argument('--stage', type=int, required=True, choices=NPU_SUPPORTED_STAGES,
                       help='Stage ID (1/4 skipped for NPU - scatter operations)')
    parser.add_argument('--edges', type=str, default=None,
                       help='Comma-separated edge counts (default: all from config)')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=50, help='Measurement iterations')
    parser.add_argument('--feature-dim', type=int, default=500, help='Feature dimension')

    args = parser.parse_args()

    # Load config
    config = load_test_cases()

    # Get edges to test
    if args.edges:
        edges_list = [int(e.strip()) for e in args.edges.split(',')]
    else:
        edges_list = get_edges_for_nodes(config, args.nodes)

    if not edges_list:
        print(f"ERROR: No edges found for nodes={args.nodes}")
        sys.exit(1)

    print(f"=" * 60)
    print(f"GCN NPU Test: Stage {args.stage}, Nodes={args.nodes}")
    print(f"NPU Skip Stages: 1, 4 (scatter operations)")
    print(f"Edges to test: {edges_list}")
    print(f"=" * 60)

    # Results storage
    results = {}
    failed_count = 0
    device_lost = False

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for edges in edges_list:
        ir_path = MODELS_DIR / f"stage{args.stage}_npu_n{args.nodes}_e{edges}.xml"

        if not ir_path.exists():
            print(f"  [{args.nodes}n, {edges}e] IR not found, skipping")
            continue

        print(f"  [{args.nodes}n, {edges}e] Testing... ", end='', flush=True)

        dummy_input = generate_dummy_input(args.stage, args.nodes, edges, args.feature_dim)
        result = measure_latency_npu(ir_path, dummy_input, args.warmup, args.iterations)

        key = f"{args.nodes},{edges},NPU,{args.stage}"

        if result['failed']:
            failed_count += 1
            print(f"FAILED")

            if result.get('device_lost'):
                print(f"    !! DEVICE_LOST detected - NPU in bad state")
                device_lost = True
                results[key] = result
                break
            elif result.get('oom'):
                print(f"    !! OUT_OF_MEMORY - skipping remaining tests for this stage")
                results[key] = result
                break
            else:
                print(f"    Error: {result.get('error', 'Unknown')[:80]}")
                results[key] = result
        else:
            print(f"{result['mean']:.2f}ms +/- {result['std']:.2f}")
            results[key] = result

    # Save checkpoint
    checkpoint_file = CHECKPOINT_DIR / f"npu_stage{args.stage}_n{args.nodes}.json"

    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results: {len(results) - failed_count} passed, {failed_count} failed")
    print(f"Saved to: {checkpoint_file}")
    print(f"{'=' * 60}")

    # Exit codes
    if device_lost:
        sys.exit(2)  # Fatal error
    elif failed_count > 0:
        sys.exit(1)  # Partial failure
    else:
        sys.exit(0)  # All passed

if __name__ == '__main__':
    main()
