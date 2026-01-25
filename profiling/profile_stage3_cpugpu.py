#!/usr/bin/env python3
"""
Stage 3 CPU/GPU Test Script - Tests Stage 3 (REDUCE_SUM) on CPU and GPU only

Stage 3 uses scatter_add which is NOT supported on NPU, so this script
only tests CPU and GPU.

Usage:
    python profile_stage3_cpugpu.py
    python profile_stage3_cpugpu.py --nodes 1000,5000,10000
    python profile_stage3_cpugpu.py --pu CPU
    python profile_stage3_cpugpu.py --pu GPU

Exit codes:
    0 - All tests passed
    1 - Some tests failed
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
MODELS_DIR = SCRIPT_DIR / "exported_models"
RESULTS_DIR = SCRIPT_DIR / "results"

# Stage 3 specific
STAGE_ID = 3
FEATURE_DIM = 500


def load_test_cases():
    config_path = SCRIPT_DIR / "test_cases.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def generate_dummy_input(num_nodes, num_edges, feature_dim=500):
    """Generate dummy input for Stage 3 (REDUCE_SUM)"""
    torch.manual_seed(42)
    # Stage 3: (messages[E,F], edge_index[2,E], num_nodes)
    messages = torch.randn(num_edges, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return (messages, edge_index, torch.tensor(num_nodes))


def measure_latency_openvino(ir_path, pu, dummy_input, num_warmup=10, num_iterations=50):
    """Measure latency using OpenVINO async API (includes CPU↔GPU data transfer time)"""
    try:
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(ir_path))
        compiled_model = core.compile_model(model, pu)

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

        # Measure using async API (set tensor each time to include CPU→GPU transfer)
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
        return {
            'mean': -1,
            'std': -1,
            'min': -1,
            'max': -1,
            'failed': True,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Stage 3 CPU/GPU Test')
    parser.add_argument('--nodes', type=str, default=None,
                        help='Comma-separated node counts (default: all from config)')
    parser.add_argument('--pu', type=str, default='CPU,GPU',
                        help='Processing units to test (default: CPU,GPU)')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=50, help='Measurement iterations')

    args = parser.parse_args()

    # Load config
    config = load_test_cases()
    test_cases = config['test_cases']

    # Filter by nodes if specified
    if args.nodes:
        node_list = [int(n.strip()) for n in args.nodes.split(',')]
        test_cases = [c for c in test_cases if c['nodes'] in node_list]

    # Parse PUs
    pu_list = [p.strip().upper() for p in args.pu.split(',')]

    print("=" * 60)
    print("Stage 3 (REDUCE_SUM) CPU/GPU Test")
    print("=" * 60)
    print(f"Test cases: {len(test_cases)}")
    print(f"Processing units: {pu_list}")
    print("=" * 60)

    # Results storage
    results = {}
    failed_count = 0

    for pu in pu_list:
        ir_path = MODELS_DIR / f"stage{STAGE_ID}_{pu.lower()}.xml"

        if not ir_path.exists():
            print(f"\nWARNING: {pu} IR not found at {ir_path}")
            print(f"  Run first: python profile_stages.py --export-cpugpu")
            continue

        print(f"\n--- Testing on {pu} ---")

        for case in test_cases:
            nodes, edges = case['nodes'], case['edges']
            print(f"  [{nodes}n, {edges}e] Testing... ", end='', flush=True)

            dummy_input = generate_dummy_input(nodes, edges, FEATURE_DIM)
            result = measure_latency_openvino(ir_path, pu, dummy_input, args.warmup, args.iterations)

            key = f"{nodes},{edges},{pu},{STAGE_ID}"

            if result['failed']:
                failed_count += 1
                print(f"FAILED: {result.get('error', 'Unknown')[:60]}")
            else:
                print(f"{result['mean']:.2f}ms +/- {result['std']:.2f}")

            results[key] = result

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "stage3_cpugpu.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results: {len(results) - failed_count} passed, {failed_count} failed")
    print(f"Saved to: {output_file}")
    print("=" * 60)

    # Exit code
    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
