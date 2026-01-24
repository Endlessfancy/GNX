#!/usr/bin/env python3
"""
NPU Isolated Testing - Test one model/node/edge combination at a time

This script tests NPU for a SINGLE model and test case.
Run this in a subprocess to isolate NPU failures.

Note: Complete GNN models include scatter operations which may require
      HETERO:NPU,CPU fallback mode.

Usage:
    python profile_npu_isolated.py --model graphsage --nodes 5000 --edges 50000
    python profile_npu_isolated.py --model gcn --nodes 10000 --edges 100000
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

import torch

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / 'exported_models'
RESULTS_DIR = SCRIPT_DIR / 'results'
TEST_CASES_FILE = SCRIPT_DIR / 'test_cases.json'
FEATURE_DIM = 500


def load_config():
    with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_input(num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """Generate input for complete GNN models"""
    torch.manual_seed(42)
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return (x, edge_index)


def measure_latency_openvino(ir_path, device, dummy_input, num_warmup=10, num_iterations=50):
    """Measure latency using OpenVINO async API for more accurate timing"""
    try:
        import openvino.runtime as ov

        core = ov.Core()
        model = core.read_model(str(ir_path))

        # Try direct device first, fallback to HETERO if needed
        try:
            compiled_model = core.compile_model(model, device)
        except Exception as e:
            if device == 'NPU':
                print(f"  Direct NPU failed, trying HETERO:NPU,CPU...")
                compiled_model = core.compile_model(model, 'HETERO:NPU,CPU')
            else:
                raise e

        if isinstance(dummy_input, tuple):
            inputs = [t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
                     for t in dummy_input]
        else:
            inputs = [dummy_input.numpy()]

        # Create infer request for async inference
        infer_request = compiled_model.create_infer_request()

        # Set input tensors
        for i in range(len(inputs)):
            infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))

        # Warmup
        for _ in range(num_warmup):
            infer_request.start_async()
            infer_request.wait()

        # Measure using async API
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            infer_request.start_async()
            infer_request.wait()
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'failed': False
        }

    except Exception as e:
        return {
            'mean': -1, 'std': -1, 'min': -1, 'max': -1,
            'failed': True, 'error': str(e)
        }


def test_single_case(model_name: str, num_nodes: int, num_edges: int):
    """Test a single model and test case on NPU"""
    config = load_config()
    num_warmup = config['config']['num_warmup']
    num_iterations = config['config']['num_iterations']

    print(f"=" * 60)
    print(f"NPU Testing: {model_name} - {num_nodes} nodes, {num_edges} edges")
    print(f"=" * 60)

    key = f"{model_name},{num_nodes},{num_edges},NPU"
    ir_path = MODELS_DIR / f"{model_name}_npu_n{num_nodes}_e{num_edges}.xml"

    if not ir_path.exists():
        print(f"  IR not found: {ir_path}")
        return {key: {'failed': True, 'error': 'IR not found'}}

    print(f"  Testing... ", end='', flush=True)

    dummy_input = generate_input(num_nodes, num_edges)
    result = measure_latency_openvino(ir_path, 'NPU', dummy_input,
                                      num_warmup, num_iterations)

    if result['failed']:
        print(f"FAILED: {result.get('error', '')[:40]}")
    else:
        print(f"{result['mean']:.2f}ms")

    return {key: result}


def main():
    parser = argparse.ArgumentParser(description='NPU Isolated Testing - Baseline Models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['graphsage', 'gcn', 'gat'],
                       help='Model type to test')
    parser.add_argument('--nodes', type=int, required=True,
                       help='Number of nodes (e.g., 5000, 10000, 50000)')
    parser.add_argument('--edges', type=int, required=True,
                       help='Number of edges (e.g., 50000, 100000)')

    args = parser.parse_args()

    results = test_single_case(args.model, args.nodes, args.edges)

    # Save results for this test case
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f'npu_{args.model}_n{args.nodes}_e{args.edges}.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Return exit code based on results
    key = f"{args.model},{args.nodes},{args.edges},NPU"
    if results.get(key, {}).get('failed', True):
        sys.exit(1)  # Failed
    else:
        sys.exit(0)  # Success


if __name__ == '__main__':
    main()
