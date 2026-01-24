#!/usr/bin/env python3
"""
CPU Latency Verification Script - Async vs Sync Comparison

This script compares synchronous and asynchronous inference methods on CPU.
Uses smaller test cases for faster verification.

Usage:
    python verify_cpu_async.py
"""

import json
import sys
import time
from pathlib import Path
import numpy as np

import torch

# Configuration
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / 'exported_models'
RESULTS_DIR = SCRIPT_DIR / 'results'

FEATURE_DIM = 500
MODEL_NAMES = ['graphsage', 'gcn', 'gat']

# Test cases matching GPU verification
VERIFY_CASES = [
    {'nodes': 5000, 'edges': 50000},
    {'nodes': 10000, 'edges': 100000},
    {'nodes': 20000, 'edges': 200000},
    {'nodes': 50000, 'edges': 500000},
    {'nodes': 80000, 'edges': 800000},
    {'nodes': 100000, 'edges': 1000000},
]


def generate_input(num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """Generate dummy input for GNN models"""
    torch.manual_seed(42)
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return (x, edge_index)


def get_output_data(result):
    """Extract output data from OpenVINO result"""
    if isinstance(result, np.ndarray):
        return result
    output = result[0] if hasattr(result, '__getitem__') else result
    if isinstance(output, np.ndarray):
        return output
    if hasattr(output, 'data'):
        return output.data
    return np.array(output)


def measure_sync(compiled_model, inputs, num_warmup=10, num_iterations=50):
    """Synchronous inference: compiled_model(inputs)"""
    # Warmup
    for _ in range(num_warmup):
        result = compiled_model(inputs)
        _ = get_output_data(result)

    # Measure
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = compiled_model(inputs)
        output = get_output_data(result)
        _ = output.sum()
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }


def measure_async(compiled_model, inputs, num_warmup=10, num_iterations=50):
    """Asynchronous inference: start_async() + wait()"""
    import openvino.runtime as ov

    infer_request = compiled_model.create_infer_request()

    # Set inputs once
    for i in range(len(inputs)):
        infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))

    # Warmup
    for _ in range(num_warmup):
        infer_request.start_async()
        infer_request.wait()

    # Measure
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        infer_request.start_async()
        infer_request.wait()
        output = infer_request.get_output_tensor().data
        _ = output.sum()
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }


def measure_async_compute_only(compiled_model, inputs, num_warmup=10, num_iterations=50):
    """Async: Measure only start_async + wait (no output access)"""
    import openvino.runtime as ov

    infer_request = compiled_model.create_infer_request()

    # Set inputs once
    for i in range(len(inputs)):
        infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))

    # Warmup
    for _ in range(num_warmup):
        infer_request.start_async()
        infer_request.wait()

    # Measure
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        infer_request.start_async()
        infer_request.wait()
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }


def main():
    print("=" * 90)
    print("CPU Latency Verification - Sync vs Async Comparison")
    print("=" * 90)
    print()
    print("Comparing inference methods on CPU:")
    print("  Sync:           compiled_model(inputs) + data access")
    print("  Async:          start_async + wait + data access")
    print("  Async Compute:  start_async + wait only")
    print()
    print("Using smaller test cases for faster CPU verification")
    print()

    try:
        import openvino.runtime as ov
    except ImportError:
        print("ERROR: OpenVINO not found")
        sys.exit(1)

    core = ov.Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")
    print()

    results = []

    for model_name in MODEL_NAMES:
        print(f"\n{'='*90}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*90}")

        ir_path = MODELS_DIR / f"{model_name}_cpu.xml"

        if not ir_path.exists():
            print(f"  IR not found: {ir_path}")
            continue

        print(f"  Loading model: {ir_path.name}")
        try:
            model = core.read_model(str(ir_path))
            compiled_model = core.compile_model(model, 'CPU')
        except Exception as e:
            print(f"  Failed to compile: {e}")
            continue

        print()
        print(f"  {'Test Case':<25} {'Sync':<12} {'Async':<12} {'AsyncComp':<12} {'Diff':<12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for case in VERIFY_CASES:
            nodes, edges = case['nodes'], case['edges']
            test_name = f"{nodes//1000}k nodes, {edges//1000}k edges"

            try:
                dummy_input = generate_input(nodes, edges)
                inputs = [t.numpy() for t in dummy_input]

                r_sync = measure_sync(compiled_model, inputs)
                r_async = measure_async(compiled_model, inputs)
                r_async_compute = measure_async_compute_only(compiled_model, inputs)

                diff_pct = ((r_sync['mean'] - r_async['mean']) / r_async['mean']) * 100

                print(f"  {test_name:<25} {r_sync['mean']:>8.2f} ms  {r_async['mean']:>8.2f} ms  {r_async_compute['mean']:>8.2f} ms  {diff_pct:>+8.1f}%")

                results.append({
                    'model': model_name,
                    'nodes': nodes,
                    'edges': edges,
                    'sync_ms': r_sync['mean'],
                    'async_ms': r_async['mean'],
                    'async_compute_ms': r_async_compute['mean'],
                    'diff_percent': diff_pct
                })

            except Exception as e:
                error_msg = str(e)[:60]
                print(f"  {test_name:<25} FAILED: {error_msg}")
                results.append({
                    'model': model_name,
                    'nodes': nodes,
                    'edges': edges,
                    'error': str(e)
                })

    # Summary
    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)

    valid_results = [r for r in results if 'error' not in r]

    if valid_results:
        sync_times = [r['sync_ms'] for r in valid_results]
        async_times = [r['async_ms'] for r in valid_results]
        diffs = [r['diff_percent'] for r in valid_results]

        print(f"\nSync inference mean: {np.mean(sync_times):.2f} ms")
        print(f"Async inference mean: {np.mean(async_times):.2f} ms")
        print(f"Difference: {np.mean(diffs):+.1f}%")

        if abs(np.mean(diffs)) < 5:
            print("\n✓ Sync and Async produce similar results on CPU")
        else:
            print(f"\n~ Difference of {np.mean(diffs):.1f}% between methods")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / 'cpu_async_verification_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
