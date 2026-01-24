#!/usr/bin/env python3
"""
Baseline Profiling Script - Complete 1-Layer GNN Models

Tests complete 1-layer GNN models (GraphSAGE, GCN, GAT) on CPU/GPU/NPU.
- CPU/GPU: Dynamic shape models
- NPU: Static shape models (one per test case)

Usage:
    python profile_baseline.py --export          # Export all models
    python profile_baseline.py --export-cpugpu   # Export CPU/GPU models only
    python profile_baseline.py --export-npu      # Export NPU models only
    python profile_baseline.py --measure-cpu     # Measure CPU only
    python profile_baseline.py --measure-gpu     # Measure GPU only
    python profile_baseline.py --measure-cpugpu  # Measure CPU and GPU
    python profile_baseline.py --analyze         # Generate summary from results
    python profile_baseline.py --all             # Full workflow (export + measure + analyze)
"""

import argparse
import json
import sys
import time
import os
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

# Import models
from models import GraphSAGE1Layer, GCN1Layer, GAT1Layer, MODEL_REGISTRY

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / 'exported_models'
RESULTS_DIR = SCRIPT_DIR / 'results'
TEST_CASES_FILE = SCRIPT_DIR / 'test_cases.json'

# Feature dimensions
FEATURE_DIM = 500
OUT_DIM = 256

# Models to test
MODEL_NAMES = ['graphsage', 'gcn', 'gat']

# ============================================================================
# Helper Functions
# ============================================================================

def load_config():
    """Load test configuration"""
    with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_input(num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """Generate dummy input for GNN models"""
    torch.manual_seed(42)
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return (x, edge_index)


def get_model_class(model_name):
    """Get model class by name"""
    return MODEL_REGISTRY[model_name]


# ============================================================================
# Model Export Functions
# ============================================================================

def export_cpugpu_models():
    """Export CPU/GPU dynamic models for all GNN types"""
    print("=" * 70)
    print("Exporting CPU/GPU Dynamic Models (Complete 1-Layer GNNs)")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate dummy input for export
    dummy_input = generate_input(5000, 50000)

    # Dynamic axes for variable sizes
    dynamic_axes = {
        'x': {0: 'num_nodes'},
        'edge_index': {1: 'num_edges'},
        'output': {0: 'num_nodes'}
    }

    success_count = 0

    for model_name in MODEL_NAMES:
        print(f"\n--- {model_name.upper()} ---")

        # Create model
        ModelClass = get_model_class(model_name)
        model = ModelClass(FEATURE_DIM, OUT_DIM)
        model.eval()

        # Export ONNX
        onnx_path = MODELS_DIR / f"{model_name}_dynamic.onnx"
        print(f"  Exporting to ONNX: {onnx_path.name}")

        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    input_names=['x', 'edge_index'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                    opset_version=17,
                    do_constant_folding=True
                )
        except Exception as e:
            print(f"  ONNX export failed: {e}")
            continue

        # Convert to OpenVINO IR for CPU and GPU
        for device in ['CPU', 'GPU']:
            ir_path = MODELS_DIR / f"{model_name}_{device.lower()}.xml"
            print(f"  Converting to {device} IR: {ir_path.name}")
            success = convert_to_ir(onnx_path, ir_path)
            if success:
                success_count += 1
            else:
                print(f"  WARNING: {device} IR conversion failed")

    print(f"\nCPU/GPU models exported: {success_count} files")
    return success_count > 0


def export_npu_models(test_cases):
    """Export NPU static models for all GNN types and test cases"""
    print("\n" + "=" * 70)
    print("Exporting NPU Static Models (Complete 1-Layer GNNs)")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    total_models = len(MODEL_NAMES) * len(test_cases)
    print(f"Total models to export: {total_models} ({len(MODEL_NAMES)} models x {len(test_cases)} test cases)")

    success_count = 0
    fail_count = 0
    idx = 0

    for model_name in MODEL_NAMES:
        print(f"\n--- {model_name.upper()} ---")

        for case in test_cases:
            idx += 1
            nodes, edges = case['nodes'], case['edges']
            print(f"[{idx}/{total_models}] {model_name} n{nodes}_e{edges}", end=' ')

            # Create model
            ModelClass = get_model_class(model_name)
            model = ModelClass(FEATURE_DIM, OUT_DIM)
            model.eval()

            # Generate dummy input (static shape)
            dummy_input = generate_input(nodes, edges)

            # Export ONNX (static - no dynamic_axes)
            onnx_path = MODELS_DIR / f"{model_name}_npu_n{nodes}_e{edges}.onnx"

            try:
                with torch.no_grad():
                    torch.onnx.export(
                        model,
                        dummy_input,
                        str(onnx_path),
                        input_names=['x', 'edge_index'],
                        output_names=['output'],
                        opset_version=17,
                        do_constant_folding=True
                    )
            except Exception as e:
                print(f"ONNX failed: {e}")
                fail_count += 1
                continue

            # Convert to NPU IR
            ir_path = MODELS_DIR / f"{model_name}_npu_n{nodes}_e{edges}.xml"
            success = convert_to_ir(onnx_path, ir_path)

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
    """Measure latency using OpenVINO async API for more accurate timing"""
    try:
        import openvino.runtime as ov

        core = ov.Core()
        model = core.read_model(str(ir_path))
        compiled_model = core.compile_model(model, device)

        # Prepare inputs
        inputs = [t.numpy() if isinstance(t, torch.Tensor) else np.array(t) for t in dummy_input]

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


def measure_cpugpu(test_cases, config, devices=['CPU', 'GPU']):
    """Measure CPU and/or GPU latencies for all models"""
    print("\n" + "=" * 70)
    print(f"Measuring {'/'.join(devices)} Latencies (Complete 1-Layer GNNs)")
    print("=" * 70)

    num_warmup = config['config']['num_warmup']
    num_iterations = config['config']['num_iterations']
    results = {}

    for model_name in MODEL_NAMES:
        print(f"\n--- {model_name.upper()} ---")

        for device in devices:
            print(f"\n  [{device}]")
            ir_path = MODELS_DIR / f"{model_name}_{device.lower()}.xml"

            if not ir_path.exists():
                print(f"  IR not found: {ir_path}")
                continue

            for case in test_cases:
                nodes, edges = case['nodes'], case['edges']
                print(f"    [{nodes}n, {edges}e]... ", end='', flush=True)

                dummy_input = generate_input(nodes, edges)
                result = measure_latency_openvino(ir_path, device, dummy_input,
                                                  num_warmup, num_iterations)

                key = f"{model_name},{nodes},{edges},{device}"
                results[key] = result

                if result['failed']:
                    print(f"FAILED: {result.get('error', '')[:50]}")
                else:
                    print(f"{result['mean']:.2f}ms")

    return results


def measure_cpu(test_cases, config):
    """Measure CPU only"""
    return measure_cpugpu(test_cases, config, devices=['CPU'])


def measure_gpu(test_cases, config):
    """Measure GPU only"""
    return measure_cpugpu(test_cases, config, devices=['GPU'])


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


def generate_csv(results, test_cases):
    """Generate CSV summary"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_lines = []
    header = "model,nodes,edges,ratio,scenario,CPU_ms,GPU_ms,NPU_ms"
    csv_lines.append(header)

    for model_name in MODEL_NAMES:
        for case in test_cases:
            nodes, edges = case['nodes'], case['edges']
            ratio = case.get('ratio', edges // nodes)
            scenario = case.get('scenario', '')

            cpu_key = f"{model_name},{nodes},{edges},CPU"
            gpu_key = f"{model_name},{nodes},{edges},GPU"
            npu_key = f"{model_name},{nodes},{edges},NPU"

            cpu_ms = results.get(cpu_key, {}).get('mean', -1) if not results.get(cpu_key, {}).get('failed', True) else -1
            gpu_ms = results.get(gpu_key, {}).get('mean', -1) if not results.get(gpu_key, {}).get('failed', True) else -1
            npu_ms = results.get(npu_key, {}).get('mean', -1) if not results.get(npu_key, {}).get('failed', True) else -1

            row = f"{model_name},{nodes},{edges},{ratio},{scenario},{cpu_ms:.3f},{gpu_ms:.3f},{npu_ms:.3f}"
            csv_lines.append(row)

    csv_path = RESULTS_DIR / "baseline_latency.csv"
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines))

    print(f"\nCSV saved: {csv_path}")


def generate_summary(results, test_cases):
    """Generate human-readable summary"""
    print("\n" + "=" * 70)
    print("Baseline Profiling Summary (Complete 1-Layer GNNs)")
    print("=" * 70)

    print("\nModels tested: GraphSAGE, GCN, GAT (1 layer each)")

    for model_name in MODEL_NAMES:
        print(f"\n--- {model_name.upper()} ---")

        cpu_times = [v['mean'] for k, v in results.items()
                     if k.startswith(f"{model_name},") and k.endswith(',CPU') and not v.get('failed')]
        gpu_times = [v['mean'] for k, v in results.items()
                     if k.startswith(f"{model_name},") and k.endswith(',GPU') and not v.get('failed')]
        npu_times = [v['mean'] for k, v in results.items()
                     if k.startswith(f"{model_name},") and k.endswith(',NPU') and not v.get('failed')]

        if cpu_times:
            print(f"  CPU: mean={np.mean(cpu_times):.2f}ms, min={np.min(cpu_times):.2f}ms, max={np.max(cpu_times):.2f}ms ({len(cpu_times)} tests)")
        if gpu_times:
            print(f"  GPU: mean={np.mean(gpu_times):.2f}ms, min={np.min(gpu_times):.2f}ms, max={np.max(gpu_times):.2f}ms ({len(gpu_times)} tests)")
        if npu_times:
            print(f"  NPU: mean={np.mean(npu_times):.2f}ms, min={np.min(npu_times):.2f}ms, max={np.max(npu_times):.2f}ms ({len(npu_times)} tests)")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Baseline Profiling - Complete 1-Layer GNNs')
    parser.add_argument('--export', action='store_true', help='Export all models')
    parser.add_argument('--export-cpugpu', action='store_true', help='Export CPU/GPU models only')
    parser.add_argument('--export-npu', action='store_true', help='Export NPU models only')
    parser.add_argument('--measure-cpu', action='store_true', help='Measure CPU only')
    parser.add_argument('--measure-gpu', action='store_true', help='Measure GPU only')
    parser.add_argument('--measure-cpugpu', action='store_true', help='Measure CPU and GPU')
    parser.add_argument('--analyze', action='store_true', help='Generate summary from existing results')
    parser.add_argument('--all', action='store_true', help='Full workflow')

    args = parser.parse_args()

    # Load config
    config = load_config()
    test_cases = config['test_cases']

    print("=" * 70)
    print("Baseline Profiling - Complete 1-Layer GNN Models")
    print("=" * 70)
    print(f"Models: {', '.join(MODEL_NAMES)}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Feature dim: {FEATURE_DIM}, Output dim: {OUT_DIM}")
    print()

    # Export
    if args.export or args.export_cpugpu or args.all:
        export_cpugpu_models()

    if args.export or args.export_npu or args.all:
        export_npu_models(test_cases)

    # Measure
    results = {}

    if args.measure_cpu:
        cpu_results = measure_cpu(test_cases, config)
        results.update(cpu_results)
        save_results(cpu_results, 'cpu_results.json')

    if args.measure_gpu:
        gpu_results = measure_gpu(test_cases, config)
        results.update(gpu_results)
        save_results(gpu_results, 'gpu_results.json')

    if args.measure_cpugpu or args.all:
        cpugpu_results = measure_cpugpu(test_cases, config)
        results.update(cpugpu_results)
        save_results(cpugpu_results, 'cpugpu_results.json')

    # Analyze
    if args.analyze or args.all:
        # Load all available results
        for filename in ['cpugpu_results.json', 'cpu_results.json', 'gpu_results.json', 'npu_results.json']:
            filepath = RESULTS_DIR / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        loaded = json.load(f)
                        results.update(loaded)
                except:
                    pass

        if results:
            generate_csv(results, test_cases)
            generate_summary(results, test_cases)
        else:
            print("No results found to analyze")

    if not any([args.export, args.export_cpugpu, args.export_npu,
                args.measure_cpu, args.measure_gpu, args.measure_cpugpu,
                args.analyze, args.all]):
        parser.print_help()


if __name__ == '__main__':
    main()
