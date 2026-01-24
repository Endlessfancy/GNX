#!/usr/bin/env python3
"""
GPU Latency Verification Script

This script verifies GPU latency measurements by comparing two methods:
1. Current method: Discard output (may not include GPU sync)
2. Verified method: Access output data (forces GPU sync and D2H transfer)

Usage:
    python verify_gpu_latency.py
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


def measure_gpu_method1(compiled_model, inputs, num_warmup=5, num_iterations=20):
    """
    Method 1: Current profiling method - discard output
    This may NOT include full GPU synchronization
    """
    # Warmup
    for _ in range(num_warmup):
        _ = compiled_model(inputs)

    # Measure
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = compiled_model(inputs)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }


def get_output_data(result):
    """Extract output data from OpenVINO result, handling different return types"""
    output = result[0]
    # Check if it's already a numpy array
    if isinstance(output, np.ndarray):
        return output
    # Otherwise try to get data or convert to numpy
    if hasattr(output, 'data'):
        return output.data
    if hasattr(output, 'numpy'):
        return output.numpy()
    return np.array(output)


def measure_gpu_method2(compiled_model, inputs, num_warmup=5, num_iterations=20):
    """
    Method 2: Verified method - access output data
    This FORCES GPU synchronization and Device-to-Host transfer
    """
    # Warmup
    for _ in range(num_warmup):
        result = compiled_model(inputs)
        _ = get_output_data(result)  # Force sync

    # Measure
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = compiled_model(inputs)
        _ = get_output_data(result)  # Force GPU sync by accessing output
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }


def measure_gpu_method3(compiled_model, inputs, num_warmup=5, num_iterations=20):
    """
    Method 3: Force data copy - ensures complete D2H transfer
    """
    # Warmup
    for _ in range(num_warmup):
        result = compiled_model(inputs)
        output = get_output_data(result)
        _ = output.sum()  # Force actual data access

    # Measure
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = compiled_model(inputs)
        output = get_output_data(result)
        _ = output.sum()  # Force complete data read
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }


def main():
    print("=" * 80)
    print("GPU Latency Verification Test")
    print("=" * 80)
    print()
    print("Comparing three measurement methods:")
    print("  Method 1: Discard output (current profiling method)")
    print("  Method 2: Access output.data (force sync)")
    print("  Method 3: Convert to numpy (force complete D2H transfer)")
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
        print(f"\n{'='*80}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*80}")

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
        print(f"  {'Test Case':<25} {'Method1':<12} {'Method2':<12} {'Method3':<12} {'Diff 1-3':<12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for case in VERIFY_CASES:
            nodes, edges = case['nodes'], case['edges']
            test_name = f"{nodes//1000}k nodes, {edges//1000}k edges"

            try:
                # Generate input
                dummy_input = generate_input(nodes, edges)
                inputs = [t.numpy() for t in dummy_input]

                # Measure with all three methods
                r1 = measure_gpu_method1(compiled_model, inputs)
                r2 = measure_gpu_method2(compiled_model, inputs)
                r3 = measure_gpu_method3(compiled_model, inputs)

                diff_pct = ((r3['mean'] - r1['mean']) / r1['mean']) * 100

                print(f"  {test_name:<25} {r1['mean']:>8.2f} ms  {r2['mean']:>8.2f} ms  {r3['mean']:>8.2f} ms  {diff_pct:>+8.1f}%")

                results.append({
                    'model': model_name,
                    'nodes': nodes,
                    'edges': edges,
                    'method1_discard_ms': r1['mean'],
                    'method2_access_ms': r2['mean'],
                    'method3_numpy_ms': r3['mean'],
                    'diff_percent': diff_pct
                })

            except Exception as e:
                error_msg = str(e)[:50]
                print(f"  {test_name:<25} FAILED: {error_msg}")
                results.append({
                    'model': model_name,
                    'nodes': nodes,
                    'edges': edges,
                    'error': str(e)
                })

    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    valid_results = [r for r in results if 'error' not in r]

    if valid_results:
        diffs = [r['diff_percent'] for r in valid_results]
        avg_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        min_diff = np.min(diffs)

        print(f"\nLatency difference (Method3 vs Method1):")
        print(f"  Average: {avg_diff:+.1f}%")
        print(f"  Min:     {min_diff:+.1f}%")
        print(f"  Max:     {max_diff:+.1f}%")

        if abs(avg_diff) < 5:
            print("\n✓ Results are consistent - previous profiling is likely accurate")
        elif avg_diff > 10:
            print(f"\n⚠ Method3 is {avg_diff:.1f}% slower - previous profiling may underestimate GPU latency")
        else:
            print(f"\n~ Small difference ({avg_diff:.1f}%) - results are approximately correct")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / 'gpu_verification_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
