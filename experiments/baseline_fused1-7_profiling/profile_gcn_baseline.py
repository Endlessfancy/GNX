#!/usr/bin/env python3
"""
GCN Fused Baseline Profiling Script

Tests FusedGCN (all 6 stages combined) on CPU/GPU as a fair baseline
for comparing against the multi-device pipeline.

Usage:
    python profile_gcn_baseline.py --export          # Export models
    python profile_gcn_baseline.py --measure-cpu     # Measure CPU only
    python profile_gcn_baseline.py --measure-gpu     # Measure GPU only
    python profile_gcn_baseline.py --measure-cpugpu  # Measure CPU and GPU
    python profile_gcn_baseline.py --all             # Full workflow
"""

import argparse
import json
import sys
import time
import multiprocessing
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

from models import FusedGCN

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / 'exported_models'
RESULTS_DIR = SCRIPT_DIR / 'results'
TEST_CASES_FILE = SCRIPT_DIR / 'test_cases.json'

# Platform identifier (set via --platform argument)
PLATFORM = ""

# Feature dimensions
FEATURE_DIM = 500
OUT_DIM = 256


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


# ============================================================================
# Model Export Functions
# ============================================================================

def export_models():
    """Export CPU/GPU dynamic models"""
    print("=" * 70)
    print("Exporting FusedGCN Models (6 Stages Combined)")
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

    # Create model
    model = FusedGCN(FEATURE_DIM, OUT_DIM)
    model.eval()

    # Export ONNX
    onnx_path = MODELS_DIR / "fused_gcn.onnx"
    print(f"\nExporting to ONNX: {onnx_path.name}")

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
        print("  ONNX export: OK")
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        return False

    # Convert to OpenVINO IR for CPU and GPU
    success_count = 0
    for device in ['CPU', 'GPU']:
        ir_path = MODELS_DIR / f"fused_gcn_{device.lower()}.xml"
        print(f"Converting to {device} IR: {ir_path.name}")
        success = convert_to_ir(onnx_path, ir_path)
        if success:
            print(f"  {device} IR: OK")
            success_count += 1
        else:
            print(f"  {device} IR: FAILED")

    print(f"\nModels exported: {success_count} files")
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
        print(f"  IR conversion failed: {e}")
        return False


# ============================================================================
# Latency Measurement Functions
# ============================================================================

def remove_outliers_iqr(data, k=1.5):
    """Remove outliers using IQR method"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]


def measure_latency_openvino(ir_path, device, dummy_input, num_warmup=10, num_iterations=50):
    """Measure latency using OpenVINO async API"""
    try:
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(ir_path))
        compiled_model = core.compile_model(model, device)

        # Prepare inputs
        inputs = [t.numpy() if isinstance(t, torch.Tensor) else np.array(t) for t in dummy_input]

        # Create infer request
        infer_request = compiled_model.create_infer_request()

        # Warmup
        for _ in range(num_warmup):
            for i in range(len(inputs)):
                infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))
            infer_request.start_async()
            infer_request.wait()

        # Measure
        latencies = []
        for _ in range(num_iterations):
            for i in range(len(inputs)):
                infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))

            start = time.perf_counter()
            infer_request.start_async()
            infer_request.wait()
            latencies.append((time.perf_counter() - start) * 1000)

        # Remove outliers
        latencies_clean = remove_outliers_iqr(latencies)
        if len(latencies_clean) < len(latencies) * 0.5:
            latencies_clean = latencies

        return {
            'mean': np.mean(latencies_clean),
            'median': np.median(latencies_clean),
            'std': np.std(latencies_clean),
            'min': np.min(latencies_clean),
            'max': np.max(latencies_clean),
            'raw_mean': np.mean(latencies),
            'outliers_removed': len(latencies) - len(latencies_clean),
            'failed': False
        }

    except Exception as e:
        return {
            'mean': -1, 'median': -1, 'std': -1, 'min': -1, 'max': -1,
            'failed': True, 'error': str(e)
        }


def _subprocess_measure_worker(ir_path, device, nodes, edges, feature_dim, num_warmup, num_iterations, result_queue):
    """子进程中运行测量，结果通过queue返回。子进程崩溃不影响主进程。"""
    try:
        dummy_input = generate_input(nodes, edges, feature_dim)
        result = measure_latency_openvino(ir_path, device, dummy_input, num_warmup, num_iterations)
        result_queue.put(result)
    except Exception as e:
        result_queue.put({
            'mean': -1, 'median': -1, 'std': -1, 'min': -1, 'max': -1,
            'failed': True, 'error': f'subprocess exception: {str(e)}'
        })


def measure_in_subprocess(ir_path, device, nodes, edges, feature_dim, num_warmup, num_iterations, timeout=300):
    """在子进程中运行测量，防止GPU OOM导致主进程崩溃"""
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_subprocess_measure_worker,
        args=(ir_path, device, nodes, edges, feature_dim, num_warmup, num_iterations, result_queue)
    )
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join(5)
        return {
            'mean': -1, 'median': -1, 'std': -1, 'min': -1, 'max': -1,
            'failed': True, 'error': f'subprocess timeout ({timeout}s)'
        }

    if p.exitcode != 0:
        return {
            'mean': -1, 'median': -1, 'std': -1, 'min': -1, 'max': -1,
            'failed': True, 'error': f'subprocess crashed (exit code {p.exitcode})'
        }

    if not result_queue.empty():
        return result_queue.get()

    return {
        'mean': -1, 'median': -1, 'std': -1, 'min': -1, 'max': -1,
        'failed': True, 'error': 'subprocess returned no result'
    }


def save_checkpoint_incremental(results, checkpoint_name):
    """增量保存checkpoint（每测完一个点就保存）"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = RESULTS_DIR / f'{checkpoint_name}.json'
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def load_checkpoint(checkpoint_name):
    """加载checkpoint（断点续跑）"""
    checkpoint_file = RESULTS_DIR / f'{checkpoint_name}.json'
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def measure_cpugpu(test_cases, config, devices=['CPU', 'GPU']):
    """Measure CPU and/or GPU latencies (with incremental checkpoint, OOM skip, subprocess isolation)"""
    print("\n" + "=" * 70)
    print(f"Measuring {'/'.join(devices)} Latencies (FusedGCN)")
    print("=" * 70)

    num_warmup = config['config']['num_warmup']
    num_iterations = config['config']['num_iterations']
    feature_dim = config['config'].get('feature_dim', FEATURE_DIM)

    checkpoint_name = f"fused_gcn_{'+'.join(devices).lower()}"

    # 断点续测
    results = load_checkpoint(checkpoint_name)
    if results is None:
        results = {}
        print(f"Starting fresh measurement...")
    else:
        print(f"Resuming from checkpoint: {len(results)} existing entries")

    total = len(devices) * len(test_cases)
    count = 0

    for device in devices:
        print(f"\n[{device}]")
        ir_path = MODELS_DIR / f"fused_gcn_{device.lower()}.xml"

        if not ir_path.exists():
            print(f"  IR not found: {ir_path}")
            count += len(test_cases)
            continue

        oom_skip_nodes = set()

        for case in test_cases:
            count += 1
            nodes, edges = case['nodes'], case['edges']
            key = f"fused_gcn,{nodes},{edges},{device}"

            if key in results:
                print(f"  [{count}/{total}] {nodes}n {edges}e... SKIP (cached)")
                continue

            if nodes in oom_skip_nodes:
                print(f"  [{count}/{total}] {nodes}n {edges}e... SKIP (OOM at smaller edge)")
                results[key] = {
                    'mean': -1, 'median': -1, 'std': -1, 'min': -1, 'max': -1,
                    'failed': True, 'error': f'skipped: OOM detected at smaller edge count for {nodes}n'
                }
                save_checkpoint_incremental(results, checkpoint_name)
                continue

            print(f"  [{count}/{total}] {nodes}n {edges}e... ", end='', flush=True)

            if device == 'GPU':
                result = measure_in_subprocess(
                    ir_path, device, nodes, edges, feature_dim,
                    num_warmup, num_iterations
                )
            else:
                try:
                    dummy_input = generate_input(nodes, edges, feature_dim)
                    result = measure_latency_openvino(ir_path, device, dummy_input,
                                                      num_warmup, num_iterations)
                except Exception as e:
                    result = {
                        'mean': -1, 'median': -1, 'std': -1, 'min': -1, 'max': -1,
                        'failed': True, 'error': f'outer catch: {str(e)}'
                    }

            results[key] = result

            if result.get('failed', False):
                print(f"FAILED: {result.get('error', '')[:60]}")
                oom_skip_nodes.add(nodes)
            else:
                print(f"{result['mean']:.2f}ms")

            save_checkpoint_incremental(results, checkpoint_name)

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

def generate_summary(results):
    """Generate summary markdown"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "# Fused GCN Baseline Results",
        "",
        "Single-device baseline using all 6 stages fused together.",
        "",
        "## Results Summary",
        "",
        "| Nodes | Edges | CPU (ms) | GPU (ms) | GPU Speedup |",
        "|-------|-------|----------|----------|-------------|",
    ]

    # Group results by test case
    test_cases = {}
    for key, result in results.items():
        if result.get('failed'):
            continue
        parts = key.split(',')
        nodes, edges, device = int(parts[1]), int(parts[2]), parts[3]
        case_key = (nodes, edges)
        if case_key not in test_cases:
            test_cases[case_key] = {}
        test_cases[case_key][device] = result['mean']

    # Generate table rows
    for (nodes, edges), devices in sorted(test_cases.items()):
        cpu_ms = devices.get('CPU', -1)
        gpu_ms = devices.get('GPU', -1)
        speedup = cpu_ms / gpu_ms if cpu_ms > 0 and gpu_ms > 0 else 0

        cpu_str = f"{cpu_ms:.2f}" if cpu_ms > 0 else "N/A"
        gpu_str = f"{gpu_ms:.2f}" if gpu_ms > 0 else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"

        summary_lines.append(f"| {nodes} | {edges} | {cpu_str} | {gpu_str} | {speedup_str} |")

    summary_path = RESULTS_DIR / "gcn_summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nSummary saved to: {summary_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    global PLATFORM, RESULTS_DIR

    parser = argparse.ArgumentParser(description="GCN Fused Baseline Profiling")
    parser.add_argument("--export", action="store_true", help="Export models")
    parser.add_argument("--measure-cpu", action="store_true", help="Measure CPU only")
    parser.add_argument("--measure-gpu", action="store_true", help="Measure GPU only")
    parser.add_argument("--measure-cpugpu", action="store_true", help="Measure CPU and GPU")
    parser.add_argument("--analyze", action="store_true", help="Generate summary")
    parser.add_argument("--all", action="store_true", help="Full workflow")
    parser.add_argument("--platform", type=str, default="", help="Platform identifier (e.g., 185H, 265V)")
    args = parser.parse_args()

    # Set platform
    if args.platform:
        PLATFORM = args.platform
        RESULTS_DIR = SCRIPT_DIR / 'results' / PLATFORM / 'gcn'
        print(f"Platform: {PLATFORM}")
        print(f"Results will be saved to: {RESULTS_DIR}")

    # Load config
    config = load_config()
    test_cases = config['test_cases']

    if args.all:
        args.export = True
        args.measure_cpugpu = True
        args.analyze = True

    all_results = {}

    # Export models
    if args.export:
        export_models()

    # Measure latencies (results saved incrementally via checkpoint)
    if args.measure_cpu:
        results = measure_cpu(test_cases, config)
        all_results.update(results)

    if args.measure_gpu:
        results = measure_gpu(test_cases, config)
        all_results.update(results)

    if args.measure_cpugpu:
        results = measure_cpugpu(test_cases, config)
        all_results.update(results)

    # Generate summary
    if args.analyze:
        if not all_results:
            for name in ['fused_gcn_cpu+gpu', 'fused_gcn_cpu', 'fused_gcn_gpu']:
                data = load_checkpoint(name)
                if data:
                    all_results.update(data)
        if all_results:
            generate_summary(all_results)

    if not any([args.export, args.measure_cpu, args.measure_gpu, args.measure_cpugpu, args.analyze]):
        parser.print_help()


if __name__ == '__main__':
    main()
