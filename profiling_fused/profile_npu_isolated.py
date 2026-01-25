#!/usr/bin/env python3
"""
NPU Isolated Testing - Test one node size at a time

This script tests NPU FusedBlock1 for a SINGLE node size.
Run this in a subprocess to isolate NPU failures.

Note: Block 1 (stages 5-7) computation is edge-independent.
      We only need one test per node size.

Usage:
    python profile_npu_isolated.py --nodes 5000
    python profile_npu_isolated.py --nodes 50000
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


def generate_block1_input(num_nodes, feature_dim=FEATURE_DIM):
    """Generate input for FusedBlock1 (edge-independent)"""
    torch.manual_seed(42)
    sum_agg = torch.randn(num_nodes, feature_dim)
    count = torch.rand(num_nodes) * 10 + 1.0
    x = torch.randn(num_nodes, feature_dim)
    return (sum_agg, count, x)


def remove_outliers_iqr(data, k=1.5):
    """Remove outliers using IQR method"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]


def measure_latency_openvino(ir_path, device, dummy_input, num_warmup=10, num_iterations=50):
    """Measure latency using OpenVINO async API (includes CPU↔NPU data transfer time)"""
    try:
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(ir_path))
        compiled_model = core.compile_model(model, device)

        if isinstance(dummy_input, tuple):
            inputs = [t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
                     for t in dummy_input]
        else:
            inputs = [dummy_input.numpy()]

        # Create infer request for async inference
        infer_request = compiled_model.create_infer_request()

        # Warmup (set tensor each time to simulate real scenario)
        for _ in range(num_warmup):
            for i in range(len(inputs)):
                infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))
            infer_request.start_async()
            infer_request.wait()

        # Measure using async API (set tensor each time to include CPU→NPU transfer)
        latencies = []
        for _ in range(num_iterations):
            # Re-set input tensors to trigger data transfer
            for i in range(len(inputs)):
                infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))

            start = time.perf_counter()
            infer_request.start_async()
            infer_request.wait()
            latencies.append((time.perf_counter() - start) * 1000)

        # Remove outliers using IQR method
        latencies_clean = remove_outliers_iqr(latencies)

        # Fallback to original if too many removed
        if len(latencies_clean) < len(latencies) * 0.5:
            latencies_clean = latencies

        return {
            'mean': float(np.mean(latencies_clean)),
            'median': float(np.median(latencies_clean)),
            'std': float(np.std(latencies_clean)),
            'min': float(np.min(latencies_clean)),
            'max': float(np.max(latencies_clean)),
            'raw_mean': float(np.mean(latencies)),
            'outliers_removed': len(latencies) - len(latencies_clean),
            'failed': False
        }

    except Exception as e:
        return {
            'mean': -1, 'median': -1, 'std': -1, 'min': -1, 'max': -1,
            'failed': True, 'error': str(e)
        }


def test_single_node_size(target_nodes: int):
    """Test a single node size (Block 1 is edge-independent)"""
    config = load_config()
    num_warmup = config['config']['num_warmup']
    num_iterations = config['config']['num_iterations']

    print(f"=" * 60)
    print(f"NPU Testing: {target_nodes} nodes (edge-independent)")
    print(f"=" * 60)

    key = f"{target_nodes},NPU,block1"
    ir_path = MODELS_DIR / f"block1_fused_npu_n{target_nodes}.xml"

    if not ir_path.exists():
        print(f"  IR not found: {ir_path}")
        return {key: {'failed': True, 'error': 'IR not found'}}

    print(f"  Testing... ", end='', flush=True)

    dummy_input = generate_block1_input(target_nodes)
    result = measure_latency_openvino(ir_path, 'NPU', dummy_input,
                                      num_warmup, num_iterations)

    if result['failed']:
        print(f"FAILED: {result.get('error', '')[:40]}")
    else:
        print(f"{result['mean']:.2f}ms")

    return {key: result}


def main():
    parser = argparse.ArgumentParser(description='NPU Isolated Testing')
    parser.add_argument('--nodes', type=int, required=True,
                       help='Node size to test (e.g., 5000, 10000, 50000)')

    args = parser.parse_args()

    results = test_single_node_size(args.nodes)

    # Save results for this node size
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f'npu_n{args.nodes}.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Return exit code based on results
    key = f"{args.nodes},NPU,block1"
    if results.get(key, {}).get('failed', True):
        sys.exit(1)  # Failed
    else:
        sys.exit(0)  # Success


if __name__ == '__main__':
    main()
