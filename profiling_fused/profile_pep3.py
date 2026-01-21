#!/usr/bin/env python3
"""
PEP3 Profiling Script - Fused Block Testing

PEP3 Configuration:
    Block 0: CPU + GPU (30%:70% data parallel) → Fused Stages 1-4 (GATHER+MESSAGE+REDUCE)
    Block 1: NPU (100%) → Fused Stages 5-7 (NORMALIZE+TRANSFORM+ACTIVATE)

This script:
1. Exports FUSED models for the PEP3 configuration
2. Measures latency for each FUSED BLOCK on its assigned device(s)
3. Outputs CSV for pipeline analysis

Usage:
    python profile_pep3.py --export          # Export models only
    python profile_pep3.py --measure         # Measure latencies only
    python profile_pep3.py --all             # Full workflow
    python profile_pep3.py --analyze         # Generate summary from results
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

# Import fused block models
from models.Model_sage import FusedBlock0, FusedBlock1

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / 'exported_models'
RESULTS_DIR = SCRIPT_DIR / 'results'
TEST_CASES_FILE = SCRIPT_DIR / 'test_cases.json'

# Feature dimension (from test_cases.json)
FEATURE_DIM = 500

# PEP3 Configuration
PEP3 = {
    'name': 'PEP3',
    'block0': {
        'devices': ['CPU', 'GPU'],
        'description': 'Fused Stages 1-4 (GATHER+MESSAGE+REDUCE_SUM+REDUCE_COUNT)',
        'ratios': [0.3, 0.7]  # CPU 30%, GPU 70%
    },
    'block1': {
        'devices': ['NPU'],
        'description': 'Fused Stages 5-7 (NORMALIZE+TRANSFORM+ACTIVATE)',
        'ratios': [1.0]
    }
}

# ============================================================================
# Helper Functions
# ============================================================================

def load_config():
    """Load test configuration"""
    with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_block0_input(num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """Generate input for FusedBlock0 (stages 1-4)"""
    torch.manual_seed(42)
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return (x, edge_index)


def generate_block1_input(num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """Generate input for FusedBlock1 (stages 5-7)"""
    torch.manual_seed(42)
    sum_agg = torch.randn(num_nodes, feature_dim)
    count = torch.rand(num_nodes) * 10 + 1.0  # Random counts > 1
    x = torch.randn(num_nodes, feature_dim)
    return (sum_agg, count, x)


# ============================================================================
# Model Export Functions
# ============================================================================

def export_cpugpu_models():
    """Export CPU/GPU dynamic models for FusedBlock0"""
    print("=" * 70)
    print("Exporting CPU/GPU Dynamic Models (FusedBlock0: Stages 1-4)")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Create model
    model = FusedBlock0()
    model.eval()

    # Generate dummy input for export
    dummy_input = generate_block0_input(5000, 10000)

    # Dynamic axes for variable sizes
    dynamic_axes = {
        'x': {0: 'num_nodes'},
        'edge_index': {1: 'num_edges'}
    }

    # Export ONNX
    onnx_path = MODELS_DIR / "block0_fused_dynamic.onnx"

    print(f"Exporting FusedBlock0 to ONNX: {onnx_path.name}")

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['x', 'edge_index'],
                output_names=['sum_agg', 'count'],
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True
            )
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        return False

    # Convert to OpenVINO IR for CPU and GPU
    for device in ['CPU', 'GPU']:
        ir_path = MODELS_DIR / f"block0_fused_{device.lower()}.xml"
        print(f"  Converting to {device} IR: {ir_path.name}")
        success = convert_to_ir(onnx_path, ir_path)
        if not success:
            print(f"  WARNING: {device} IR conversion failed")

    print("CPU/GPU FusedBlock0 models exported (2 files)")
    return True


def export_npu_models(test_cases):
    """Export NPU static models for FusedBlock1"""
    print("\n" + "=" * 70)
    print("Exporting NPU Static Models (FusedBlock1: Stages 5-7)")
    print("=" * 70)
    print(f"Total: {len(test_cases)} static models (one per test case)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for idx, case in enumerate(test_cases):
        nodes, edges = case['nodes'], case['edges']
        print(f"[{idx+1}/{len(test_cases)}] NPU n{nodes}_e{edges}", end=' ')

        # Create model
        model = FusedBlock1(FEATURE_DIM, FEATURE_DIM)
        model.eval()

        # Generate dummy input
        dummy_input = generate_block1_input(nodes, edges)

        # Export ONNX (static shapes - no dynamic_axes)
        onnx_path = MODELS_DIR / f"block1_fused_npu_n{nodes}_e{edges}.onnx"

        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    input_names=['sum_agg', 'count', 'x'],
                    output_names=['activated'],
                    opset_version=17,
                    do_constant_folding=True
                )
        except Exception as e:
            print(f"ONNX failed: {e}")
            fail_count += 1
            continue

        # Convert to NPU IR
        npu_ir = MODELS_DIR / f"block1_fused_npu_n{nodes}_e{edges}.xml"
        success = convert_to_ir(onnx_path, npu_ir)

        if success:
            print("OK")
            success_count += 1
        else:
            print("IR failed")
            fail_count += 1

    print(f"\nNPU models exported: {success_count} success, {fail_count} failed")
    return success_count > 0


def convert_to_ir(onnx_path, ir_path):
    """Convert ONNX to OpenVINO IR"""
    try:
        from openvino.tools import mo
        from openvino import save_model

        ov_model = mo.convert_model(str(onnx_path))
        save_model(ov_model, str(ir_path))
        return True
    except Exception as e:
        print(f"IR conversion failed: {e}")
        return False


# ============================================================================
# Latency Measurement Functions
# ============================================================================

def measure_latency_openvino(ir_path, device, dummy_input, num_warmup=10, num_iterations=50):
    """Measure latency using OpenVINO"""
    try:
        import openvino.runtime as ov

        core = ov.Core()
        model = core.read_model(str(ir_path))
        compiled_model = core.compile_model(model, device)

        # Prepare inputs
        if isinstance(dummy_input, tuple):
            inputs = [t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
                     for t in dummy_input]
        else:
            inputs = [dummy_input.numpy() if isinstance(dummy_input, torch.Tensor)
                     else np.array(dummy_input)]

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
            'max': np.max(latencies),
            'failed': False
        }

    except Exception as e:
        return {
            'mean': -1, 'std': -1, 'min': -1, 'max': -1,
            'failed': True, 'error': str(e)
        }


def measure_block0(test_cases, config):
    """Measure Block 0: CPU and GPU for FusedBlock0 (stages 1-4)"""
    print("\n" + "=" * 70)
    print("Measuring Block 0: CPU + GPU (FusedBlock0: Stages 1-4)")
    print("=" * 70)

    num_warmup = config['config']['num_warmup']
    num_iterations = config['config']['num_iterations']
    results = {}

    for device in ['CPU', 'GPU']:
        print(f"\n--- {device} ---")
        ir_path = MODELS_DIR / f"block0_fused_{device.lower()}.xml"

        if not ir_path.exists():
            print(f"IR not found: {ir_path}")
            continue

        for case in test_cases:
            nodes, edges = case['nodes'], case['edges']
            print(f"  [{nodes}n, {edges}e]... ", end='', flush=True)

            dummy_input = generate_block0_input(nodes, edges)
            result = measure_latency_openvino(ir_path, device, dummy_input,
                                              num_warmup, num_iterations)

            key = f"{nodes},{edges},{device},block0"
            results[key] = result

            if result['failed']:
                print(f"FAILED: {result.get('error', '')[:50]}")
            else:
                print(f"{result['mean']:.2f}ms")

    return results


def measure_block1_npu(test_cases, config):
    """Measure Block 1: NPU for FusedBlock1 (stages 5-7) with incremental saving"""
    print("\n" + "=" * 70)
    print("Measuring Block 1: NPU (FusedBlock1: Stages 5-7)")
    print("=" * 70)
    print("Results will be saved incrementally.")

    num_warmup = config['config']['num_warmup']
    num_iterations = config['config']['num_iterations']

    # Load existing results if any (for resume capability)
    results_file = RESULTS_DIR / 'block1_npu.json'
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {results_file}")
        except:
            results = {}
    else:
        results = {}

    failed_count = 0
    success_count = 0

    for idx, case in enumerate(test_cases):
        nodes, edges = case['nodes'], case['edges']
        key = f"{nodes},{edges},NPU,block1"

        # Skip if already measured successfully
        if key in results and not results[key].get('failed', True):
            print(f"  [{nodes}n, {edges}e] (cached) {results[key]['mean']:.2f}ms")
            success_count += 1
            continue

        ir_path = MODELS_DIR / f"block1_fused_npu_n{nodes}_e{edges}.xml"

        if not ir_path.exists():
            print(f"  [{nodes}n, {edges}e] IR not found")
            results[key] = {'failed': True, 'error': 'IR not found'}
            failed_count += 1
            continue

        print(f"  [{nodes}n, {edges}e]... ", end='', flush=True)

        dummy_input = generate_block1_input(nodes, edges)
        result = measure_latency_openvino(ir_path, 'NPU', dummy_input,
                                          num_warmup, num_iterations)

        results[key] = result

        if result['failed']:
            print(f"FAILED: {result.get('error', '')[:50]}")
            failed_count += 1
        else:
            print(f"{result['mean']:.2f}ms")
            success_count += 1

        # Save results incrementally (every 5 tests)
        if (idx + 1) % 5 == 0:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

    # Final save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nNPU Summary: {success_count} succeeded, {failed_count} failed")
    return results


# ============================================================================
# Results Saving Functions
# ============================================================================

def save_results(results, filename):
    """Save results to JSON"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {filepath}")


def generate_csv(block0_results, block1_results, test_cases):
    """Generate CSV summary for pipeline analysis"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_lines = []
    header = "nodes,edges,ratio,scenario,block0_CPU_ms,block0_GPU_ms,block1_NPU_ms,block0_DP_ms,total_ms"
    csv_lines.append(header)

    for case in test_cases:
        nodes, edges = case['nodes'], case['edges']
        ratio = case.get('ratio', edges // nodes)
        scenario = case.get('scenario', '')

        # Block 0 results
        cpu_key = f"{nodes},{edges},CPU,block0"
        gpu_key = f"{nodes},{edges},GPU,block0"
        npu_key = f"{nodes},{edges},NPU,block1"

        cpu_ms = block0_results.get(cpu_key, {}).get('mean', -1) if not block0_results.get(cpu_key, {}).get('failed', True) else -1
        gpu_ms = block0_results.get(gpu_key, {}).get('mean', -1) if not block0_results.get(gpu_key, {}).get('failed', True) else -1
        npu_ms = block1_results.get(npu_key, {}).get('mean', -1) if not block1_results.get(npu_key, {}).get('failed', True) else -1

        # Data parallel estimation: max(CPU * 0.3, GPU * 0.7)
        if cpu_ms > 0 and gpu_ms > 0:
            block0_dp = max(cpu_ms * 0.3, gpu_ms * 0.7)
        elif cpu_ms > 0:
            block0_dp = cpu_ms
        elif gpu_ms > 0:
            block0_dp = gpu_ms
        else:
            block0_dp = -1

        # Total pipeline time
        if block0_dp > 0 and npu_ms > 0:
            total = block0_dp + npu_ms
        else:
            total = -1

        row = f"{nodes},{edges},{ratio},{scenario},{cpu_ms:.3f},{gpu_ms:.3f},{npu_ms:.3f},{block0_dp:.3f},{total:.3f}"
        csv_lines.append(row)

    csv_path = RESULTS_DIR / "pep3_latency.csv"
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines))

    print(f"\nCSV saved: {csv_path}")


def generate_summary(block0_results, block1_results, test_cases):
    """Generate human-readable summary"""
    print("\n" + "=" * 70)
    print("PEP3 Profiling Summary (Fused Blocks)")
    print("=" * 70)

    print("\nPEP3 Configuration:")
    print("  Block 0: CPU + GPU (30%:70% DP) → Fused Stages 1-4")
    print("  Block 1: NPU (100%) → Fused Stages 5-7")

    # Calculate statistics
    cpu_times = [v['mean'] for k, v in block0_results.items() if 'CPU' in k and not v.get('failed')]
    gpu_times = [v['mean'] for k, v in block0_results.items() if 'GPU' in k and not v.get('failed')]
    npu_times = [v['mean'] for k, v in block1_results.items() if not v.get('failed')]

    print("\n" + "-" * 70)
    print("Latency Statistics (ms)")
    print("-" * 70)

    if cpu_times:
        print(f"Block 0 CPU:  mean={np.mean(cpu_times):.2f}, min={np.min(cpu_times):.2f}, max={np.max(cpu_times):.2f}")
    if gpu_times:
        print(f"Block 0 GPU:  mean={np.mean(gpu_times):.2f}, min={np.min(gpu_times):.2f}, max={np.max(gpu_times):.2f}")
    if npu_times:
        print(f"Block 1 NPU:  mean={np.mean(npu_times):.2f}, min={np.min(npu_times):.2f}, max={np.max(npu_times):.2f}")

    # Pipeline estimation for sample sizes
    print("\n" + "-" * 70)
    print("Pipeline Time Estimation (sample sizes)")
    print("-" * 70)

    for case in test_cases[:10]:  # Show first 10
        nodes, edges = case['nodes'], case['edges']

        cpu_key = f"{nodes},{edges},CPU,block0"
        gpu_key = f"{nodes},{edges},GPU,block0"
        npu_key = f"{nodes},{edges},NPU,block1"

        cpu_ms = block0_results.get(cpu_key, {}).get('mean', 0)
        gpu_ms = block0_results.get(gpu_key, {}).get('mean', 0)
        npu_ms = block1_results.get(npu_key, {}).get('mean', 0)

        if cpu_ms > 0 and gpu_ms > 0:
            block0_dp = max(cpu_ms * 0.3, gpu_ms * 0.7)
            total = block0_dp + npu_ms if npu_ms > 0 else -1
            print(f"  {nodes:>6}n, {edges:>8}e: B0_DP={block0_dp:>6.1f}ms, B1_NPU={npu_ms:>6.1f}ms, Total={total:>6.1f}ms")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PEP3 Fused Block Profiling Script')
    parser.add_argument('--export', action='store_true', help='Export all models')
    parser.add_argument('--export-cpugpu', action='store_true', help='Export CPU/GPU models only')
    parser.add_argument('--export-npu', action='store_true', help='Export NPU models only')
    parser.add_argument('--measure', action='store_true', help='Measure all latencies')
    parser.add_argument('--measure-cpugpu', action='store_true', help='Measure CPU/GPU only')
    parser.add_argument('--measure-npu', action='store_true', help='Measure NPU only')
    parser.add_argument('--analyze', action='store_true', help='Generate summary from existing results')
    parser.add_argument('--all', action='store_true', help='Full workflow')

    args = parser.parse_args()

    # Load config
    config = load_config()
    test_cases = config['test_cases']

    print("=" * 70)
    print("PEP3 Profiling - Fused Block Testing")
    print("=" * 70)
    print(f"Test cases: {len(test_cases)}")
    print(f"Feature dim: {FEATURE_DIM}")
    print("\nPEP3 Configuration:")
    print("  Block 0: CPU + GPU (30%:70% DP) → Fused Stages 1-4")
    print("  Block 1: NPU (100%) → Fused Stages 5-7")
    print()

    # Export
    if args.export or args.export_cpugpu or args.all:
        export_cpugpu_models()

    if args.export or args.export_npu or args.all:
        export_npu_models(test_cases)

    # Measure
    block0_results = {}
    block1_results = {}

    if args.measure or args.measure_cpugpu or args.all:
        block0_results = measure_block0(test_cases, config)
        save_results(block0_results, 'block0_cpugpu.json')

    if args.measure or args.measure_npu or args.all:
        block1_results = measure_block1_npu(test_cases, config)
        # Already saved incrementally, but save final version
        save_results(block1_results, 'block1_npu.json')

    # Analyze
    if args.analyze or args.all:
        # Load results if not already measured
        if not block0_results:
            try:
                with open(RESULTS_DIR / 'block0_cpugpu.json', 'r') as f:
                    block0_results = json.load(f)
            except:
                print("No Block 0 results found")

        if not block1_results:
            try:
                with open(RESULTS_DIR / 'block1_npu.json', 'r') as f:
                    block1_results = json.load(f)
            except:
                print("No Block 1 results found")

        if block0_results or block1_results:
            generate_csv(block0_results, block1_results, test_cases)
            generate_summary(block0_results, block1_results, test_cases)

    if not any([args.export, args.export_cpugpu, args.export_npu,
                args.measure, args.measure_cpugpu, args.measure_npu,
                args.analyze, args.all]):
        parser.print_help()


if __name__ == '__main__':
    main()
