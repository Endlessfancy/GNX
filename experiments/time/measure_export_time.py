#!/usr/bin/env python3
"""
Model Export Time Measurement Script

Measures the time to export models:
- CPU/GPU: FusedBlock0 (stages 1-4)
- NPU: FusedBlock1 (stages 5-7)

Usage:
    python measure_export_time.py --all
    python measure_export_time.py --cpugpu
    python measure_export_time.py --npu
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

# Add profiling_fused to path
SCRIPT_DIR = Path(__file__).parent
PROFILING_FUSED_DIR = SCRIPT_DIR.parent.parent / 'profiling_fused'
sys.path.insert(0, str(PROFILING_FUSED_DIR))

import torch
import torch.nn as nn

from models.Model_sage import FusedBlock0, FusedBlock1

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = SCRIPT_DIR / 'results'
MODELS_DIR = SCRIPT_DIR / 'exported_models'

# Feature dimensions
FEATURE_DIM = 500
OUT_DIM = 256

# Test sizes for NPU (each size needs a static model)
NPU_NODE_SIZES = [5000, 10000, 20000, 50000, 80000, 100000]


# ============================================================================
# Helper Functions
# ============================================================================

def generate_block0_input(num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """Generate input for FusedBlock0 (stages 1-4)"""
    torch.manual_seed(42)
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return (x, edge_index)


def generate_block1_input(num_nodes, feature_dim=FEATURE_DIM):
    """Generate input for FusedBlock1 (stages 5-7)"""
    torch.manual_seed(42)
    sum_agg = torch.randn(num_nodes, feature_dim)
    count = torch.randint(1, 100, (num_nodes,)).float()
    x = torch.randn(num_nodes, feature_dim)
    return (sum_agg, count, x)


# ============================================================================
# Export Functions with Timing
# ============================================================================

def measure_cpugpu_export():
    """Measure CPU/GPU model export time (FusedBlock0: stages 1-4)"""
    print("=" * 70)
    print("Measuring CPU/GPU Export Time (FusedBlock0: Stages 1-4)")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        'model': 'FusedBlock0',
        'stages': '1-4',
        'description': 'GATHER + MESSAGE + REDUCE_SUM + REDUCE_COUNT'
    }

    # Create model
    model = FusedBlock0()
    model.eval()

    # Generate dummy input
    dummy_input = generate_block0_input(5000, 50000)

    # Dynamic axes
    dynamic_axes = {
        'x': {0: 'num_nodes'},
        'edge_index': {1: 'num_edges'},
        'sum_agg': {0: 'num_nodes'},
        'count': {0: 'num_nodes'}
    }

    # ===== Measure ONNX Export =====
    onnx_path = MODELS_DIR / "block0_fused.onnx"
    print(f"\n[1] ONNX Export: {onnx_path.name}")

    start_time = time.perf_counter()
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
        onnx_time = time.perf_counter() - start_time
        print(f"    Time: {onnx_time:.3f}s")
        results['onnx_export_time_s'] = onnx_time
    except Exception as e:
        print(f"    FAILED: {e}")
        results['onnx_export_time_s'] = -1
        return results

    # ===== Measure OpenVINO IR Conversion (CPU) =====
    print(f"\n[2] OpenVINO CPU IR Conversion")
    cpu_ir_path = MODELS_DIR / "block0_fused_cpu.xml"

    start_time = time.perf_counter()
    try:
        from openvino.tools import mo
        from openvino import save_model

        ov_model = mo.convert_model(str(onnx_path))
        save_model(ov_model, str(cpu_ir_path))
        cpu_ir_time = time.perf_counter() - start_time
        print(f"    Time: {cpu_ir_time:.3f}s")
        results['cpu_ir_conversion_time_s'] = cpu_ir_time
    except Exception as e:
        print(f"    FAILED: {e}")
        results['cpu_ir_conversion_time_s'] = -1

    # ===== Measure OpenVINO IR Conversion (GPU) =====
    print(f"\n[3] OpenVINO GPU IR Conversion")
    gpu_ir_path = MODELS_DIR / "block0_fused_gpu.xml"

    start_time = time.perf_counter()
    try:
        from openvino.tools import mo
        from openvino import save_model

        ov_model = mo.convert_model(str(onnx_path))
        save_model(ov_model, str(gpu_ir_path))
        gpu_ir_time = time.perf_counter() - start_time
        print(f"    Time: {gpu_ir_time:.3f}s")
        results['gpu_ir_conversion_time_s'] = gpu_ir_time
    except Exception as e:
        print(f"    FAILED: {e}")
        results['gpu_ir_conversion_time_s'] = -1

    # Total time
    total_time = results.get('onnx_export_time_s', 0) + \
                 results.get('cpu_ir_conversion_time_s', 0) + \
                 results.get('gpu_ir_conversion_time_s', 0)
    results['total_cpugpu_time_s'] = total_time

    print(f"\n[Summary] CPU/GPU Export")
    print(f"    ONNX:   {results.get('onnx_export_time_s', -1):.3f}s")
    print(f"    CPU IR: {results.get('cpu_ir_conversion_time_s', -1):.3f}s")
    print(f"    GPU IR: {results.get('gpu_ir_conversion_time_s', -1):.3f}s")
    print(f"    Total:  {total_time:.3f}s")

    return results


def measure_npu_export():
    """Measure NPU model export time (FusedBlock1: stages 5-7)"""
    print("\n" + "=" * 70)
    print("Measuring NPU Export Time (FusedBlock1: Stages 5-7)")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        'model': 'FusedBlock1',
        'stages': '5-7',
        'description': 'NORMALIZE + TRANSFORM + ACTIVATE',
        'node_sizes': NPU_NODE_SIZES,
        'per_size_times': {}
    }

    total_onnx_time = 0
    total_ir_time = 0

    for nodes in NPU_NODE_SIZES:
        print(f"\n[NPU] Node size: {nodes}")

        # Create model
        model = FusedBlock1(FEATURE_DIM, OUT_DIM)
        model.eval()

        # Generate dummy input
        dummy_input = generate_block1_input(nodes)

        # ONNX Export
        onnx_path = MODELS_DIR / f"block1_fused_npu_n{nodes}.onnx"

        start_time = time.perf_counter()
        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    input_names=['sum_agg', 'count', 'x'],
                    output_names=['output'],
                    opset_version=17,
                    do_constant_folding=True
                )
            onnx_time = time.perf_counter() - start_time
            print(f"    ONNX:   {onnx_time:.3f}s")
        except Exception as e:
            print(f"    ONNX FAILED: {e}")
            onnx_time = -1

        # OpenVINO IR Conversion (NPU)
        ir_path = MODELS_DIR / f"block1_fused_npu_n{nodes}.xml"

        start_time = time.perf_counter()
        try:
            from openvino.tools import mo
            from openvino import save_model

            ov_model = mo.convert_model(str(onnx_path))
            save_model(ov_model, str(ir_path), compress_to_fp16=False)
            ir_time = time.perf_counter() - start_time
            print(f"    IR:     {ir_time:.3f}s")
        except Exception as e:
            print(f"    IR FAILED: {e}")
            ir_time = -1

        results['per_size_times'][str(nodes)] = {
            'onnx_time_s': onnx_time,
            'ir_time_s': ir_time,
            'total_s': onnx_time + ir_time if onnx_time > 0 and ir_time > 0 else -1
        }

        if onnx_time > 0:
            total_onnx_time += onnx_time
        if ir_time > 0:
            total_ir_time += ir_time

    results['total_onnx_time_s'] = total_onnx_time
    results['total_ir_time_s'] = total_ir_time
    results['total_npu_time_s'] = total_onnx_time + total_ir_time
    results['num_models'] = len(NPU_NODE_SIZES)

    print(f"\n[Summary] NPU Export ({len(NPU_NODE_SIZES)} models)")
    print(f"    Total ONNX: {total_onnx_time:.3f}s")
    print(f"    Total IR:   {total_ir_time:.3f}s")
    print(f"    Total:      {total_onnx_time + total_ir_time:.3f}s")
    print(f"    Avg/model:  {(total_onnx_time + total_ir_time) / len(NPU_NODE_SIZES):.3f}s")

    return results


def save_results(results, filename):
    """Save results to JSON"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Measure Model Export Time")
    parser.add_argument("--cpugpu", action="store_true", help="Measure CPU/GPU export only")
    parser.add_argument("--npu", action="store_true", help="Measure NPU export only")
    parser.add_argument("--all", action="store_true", help="Measure all exports")
    args = parser.parse_args()

    if args.all:
        args.cpugpu = True
        args.npu = True

    if not any([args.cpugpu, args.npu]):
        parser.print_help()
        return

    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'feature_dim': FEATURE_DIM,
        'out_dim': OUT_DIM
    }

    if args.cpugpu:
        cpugpu_results = measure_cpugpu_export()
        all_results['cpugpu'] = cpugpu_results
        save_results(cpugpu_results, 'export_time_cpugpu.json')

    if args.npu:
        npu_results = measure_npu_export()
        all_results['npu'] = npu_results
        save_results(npu_results, 'export_time_npu.json')

    # Save combined results
    if args.cpugpu and args.npu:
        total_time = all_results.get('cpugpu', {}).get('total_cpugpu_time_s', 0) + \
                     all_results.get('npu', {}).get('total_npu_time_s', 0)
        all_results['total_export_time_s'] = total_time

        print("\n" + "=" * 70)
        print("TOTAL EXPORT TIME SUMMARY")
        print("=" * 70)
        print(f"CPU/GPU (1 model):  {all_results.get('cpugpu', {}).get('total_cpugpu_time_s', 0):.3f}s")
        print(f"NPU ({len(NPU_NODE_SIZES)} models):    {all_results.get('npu', {}).get('total_npu_time_s', 0):.3f}s")
        print(f"TOTAL:              {total_time:.3f}s")

        save_results(all_results, 'export_time_all.json')


if __name__ == '__main__':
    main()
