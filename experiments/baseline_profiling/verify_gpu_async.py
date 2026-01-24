#!/usr/bin/env python3
"""
GPU Latency Verification Script - Async vs Sync Comparison

This script compares synchronous and asynchronous inference methods:
1. Sync: compiled_model(inputs) - blocking call
2. Async: infer_request.start_async() + wait() - explicit async

Usage:
    python verify_gpu_async.py
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
TEST_CASES_FILE = SCRIPT_DIR / 'test_cases.json'

FEATURE_DIM = 500
MODEL_NAMES = ['graphsage', 'gcn', 'gat']

# Test subset (smaller set for quick verification)
VERIFY_CASES = [
    {'nodes': 5000, 'edges': 50000},
    {'nodes': 10000, 'edges': 100000},
    {'nodes': 20000, 'edges': 200000},
    {'nodes': 50000, 'edges': 500000},
    {'nodes': 50000, 'edges': 2500000},
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
    """Extract output data from OpenVINO result, handling different return types"""
    if isinstance(result, np.ndarray):
        return result
    output = result[0] if hasattr(result, '__getitem__') else result
    if isinstance(output, np.ndarray):
        return output
    if hasattr(output, 'data'):
        return output.data
    if hasattr(output, 'numpy'):
        return output.numpy()
    return np.array(output)


def measure_sync(compiled_model, inputs, num_warmup=5, num_iterations=20):
    """
    Synchronous inference: compiled_model(inputs)
    This is a blocking call that waits for inference to complete.
    """
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
        _ = output.sum()  # Force data access
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }


def measure_async(compiled_model, inputs, num_warmup=5, num_iterations=20):
    """
    Asynchronous inference: start_async() + wait()
    Explicit async API with manual synchronization.
    """
    import openvino.runtime as ov

    # Create infer request
    infer_request = compiled_model.create_infer_request()

    # Set inputs by index
    for i in range(len(inputs)):
        infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))

    # Warmup
    for _ in range(num_warmup):
        infer_request.start_async()
        infer_request.wait()
        output = infer_request.get_output_tensor().data
        _ = output.sum()

    # Measure
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        infer_request.start_async()
        infer_request.wait()
        output = infer_request.get_output_tensor().data
        _ = output.sum()  # Force data access
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }


def measure_async_only_compute(compiled_model, inputs, num_warmup=5, num_iterations=20):
    """
    Asynchronous inference: Measure only start_async() to wait() time
    Excludes input tensor setup time.
    """
    import openvino.runtime as ov

    # Create infer request
    infer_request = compiled_model.create_infer_request()

    # Set inputs once (reused for all iterations)
    for i in range(len(inputs)):
        infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))

    # Warmup
    for _ in range(num_warmup):
        infer_request.start_async()
        infer_request.wait()

    # Measure - only compute time (inputs already set)
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


def measure_async_with_output(compiled_model, inputs, num_warmup=5, num_iterations=20):
    """
    Asynchronous inference: Full timing including output access
    Measures: set_input + start_async + wait + get_output + data access
    """
    import openvino.runtime as ov

    # Create infer request
    infer_request = compiled_model.create_infer_request()

    # Warmup
    for _ in range(num_warmup):
        for i in range(len(inputs)):
            infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))
        infer_request.start_async()
        infer_request.wait()
        output = infer_request.get_output_tensor().data
        _ = output.sum()

    # Measure - full end-to-end
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()

        # Set inputs, run, get output
        for i in range(len(inputs)):
            infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))
        infer_request.start_async()
        infer_request.wait()
        output = infer_request.get_output_tensor().data
        _ = output.sum()  # Force complete data read

        latencies.append((time.perf_counter() - start) * 1000)

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }


def main():
    print("=" * 90)
    print("GPU Latency Verification - Sync vs Async Comparison")
    print("=" * 90)
    print()
    print("Comparing inference methods:")
    print("  Sync:           compiled_model(inputs) + data access")
    print("  Async Full:     set_input + start_async + wait + get_output + data access")
    print("  Async Compute:  start_async + wait only (no input/output in timing)")
    print()

    try:
        import openvino.runtime as ov
    except ImportError:
        print("ERROR: OpenVINO not found")
        sys.exit(1)

    core = ov.Core()

    # Check GPU availability
    devices = core.available_devices
    print(f"Available devices: {devices}")

    if 'GPU' not in devices:
        print("ERROR: GPU not available")
        sys.exit(1)

    print()

    results = []

    for model_name in MODEL_NAMES:
        print(f"\n{'='*90}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*90}")

        ir_path = MODELS_DIR / f"{model_name}_gpu.xml"

        if not ir_path.exists():
            print(f"  IR not found: {ir_path}")
            continue

        # Load and compile model
        print(f"  Loading model: {ir_path.name}")
        try:
            model = core.read_model(str(ir_path))
            compiled_model = core.compile_model(model, 'GPU')
        except Exception as e:
            print(f"  Failed to compile: {e}")
            continue

        print()
        print(f"  {'Test Case':<25} {'Sync':<12} {'AsyncFull':<12} {'AsyncComp':<12} {'Sync-Full':<12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for case in VERIFY_CASES:
            nodes, edges = case['nodes'], case['edges']
            test_name = f"{nodes//1000}k nodes, {edges//1000}k edges"

            try:
                # Generate input
                dummy_input = generate_input(nodes, edges)
                inputs = [t.numpy() for t in dummy_input]

                # Measure with different methods
                r_sync = measure_sync(compiled_model, inputs)
                r_async_full = measure_async_with_output(compiled_model, inputs)
                r_async_compute = measure_async_only_compute(compiled_model, inputs)

                diff_pct = ((r_sync['mean'] - r_async_full['mean']) / r_async_full['mean']) * 100

                print(f"  {test_name:<25} {r_sync['mean']:>8.2f} ms  {r_async_full['mean']:>8.2f} ms  {r_async_compute['mean']:>8.2f} ms  {diff_pct:>+8.1f}%")

                results.append({
                    'model': model_name,
                    'nodes': nodes,
                    'edges': edges,
                    'sync_ms': r_sync['mean'],
                    'sync_std': r_sync['std'],
                    'async_full_ms': r_async_full['mean'],
                    'async_full_std': r_async_full['std'],
                    'async_compute_ms': r_async_compute['mean'],
                    'async_compute_std': r_async_compute['std'],
                    'sync_vs_async_diff_percent': diff_pct
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
        async_full_times = [r['async_full_ms'] for r in valid_results]
        async_compute_times = [r['async_compute_ms'] for r in valid_results]
        diffs = [r['sync_vs_async_diff_percent'] for r in valid_results]

        print(f"\nSync inference (compiled_model + data access):")
        print(f"  Mean: {np.mean(sync_times):.2f} ms")
        print(f"  Range: {np.min(sync_times):.2f} - {np.max(sync_times):.2f} ms")

        print(f"\nAsync inference (full end-to-end with data access):")
        print(f"  Mean: {np.mean(async_full_times):.2f} ms")
        print(f"  Range: {np.min(async_full_times):.2f} - {np.max(async_full_times):.2f} ms")

        print(f"\nAsync inference (compute only, no I/O in timing):")
        print(f"  Mean: {np.mean(async_compute_times):.2f} ms")
        print(f"  Range: {np.min(async_compute_times):.2f} - {np.max(async_compute_times):.2f} ms")

        avg_diff = np.mean(diffs)
        print(f"\nSync vs Async Full difference: {avg_diff:+.1f}%")

        if abs(avg_diff) < 5:
            print("✓ Sync and Async produce similar results")
        elif avg_diff > 0:
            print(f"⚠ Sync is {avg_diff:.1f}% slower than Async")
        else:
            print(f"⚠ Sync is {-avg_diff:.1f}% faster than Async")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / 'gpu_async_verification_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
