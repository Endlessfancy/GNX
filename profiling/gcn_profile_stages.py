"""
GNX GCN Stage Profiling Script

Automated profiling workflow for GCN (Graph Convolutional Network) 6-stage decomposition:
1. Export models: CPU/GPU dynamic + NPU static
2. Measure latency: Test on different input sizes and PUs
3. Bandwidth estimation: Separate compute and transfer time via regression
4. Generate results: lookup_table.json + bandwidth_table.json + report.txt

GCN 6-Stage Decomposition:
    Stage 1: COMPUTE_NORM - Calculate edge normalization weights [SCATTER - NPU skip]
    Stage 2: GATHER       - Collect source node features x_j
    Stage 3: MESSAGE      - msg = norm_ij * x_j (weighted message)
    Stage 4: REDUCE_SUM   - agg = sum_j msg_ij [SCATTER - NPU skip]
    Stage 5: TRANSFORM    - out = W @ agg + b (linear transform)
    Stage 6: ACTIVATE     - output = ReLU(out)

NPU Skip: Stage 1, 4 (scatter operations)

Usage:
    python gcn_profile_stages.py --all              # Run full workflow
    python gcn_profile_stages.py --export           # Export all models
    python gcn_profile_stages.py --measure          # Measure CPU/GPU latencies
    python gcn_profile_stages.py --analyze          # Analyze and generate report
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
import numpy as np

# Add current directory to path for importing models
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

# Import GCN stage models
from models.Model_gcn import (
    GCNStage1_ComputeNorm,
    GCNStage2_Gather,
    GCNStage3_Message,
    GCNStage4_ReduceSum,
    GCNStage5_Transform,
    GCNStage6_Activate
)

# ============================================================================
# Configuration
# ============================================================================

PROFILING_DIR = Path(__file__).parent
MODELS_DIR = PROFILING_DIR / 'gcn_exported_models'
RESULTS_DIR = PROFILING_DIR / 'gcn_results'
TEST_CASES_FILE = PROFILING_DIR / 'test_cases.json'

# GCN has 6 stages, NPU skips stages 1 and 4 (scatter operations)
NUM_STAGES = 6
NPU_SKIP_STAGES = [1, 4]

# ============================================================================
# Helper Functions
# ============================================================================

def load_config():
    """Load test configuration"""
    with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def get_stage_model(stage_id, num_nodes=None, feature_dim=500):
    """
    Get GCN stage model

    Args:
        stage_id: Stage number (1-6)
        num_nodes: Static node count (for NPU export of scatter stages)
        feature_dim: Feature dimension (default 500)
    """
    models = {
        1: GCNStage1_ComputeNorm(num_nodes_static=num_nodes),
        2: GCNStage2_Gather(),
        3: GCNStage3_Message(),
        4: GCNStage4_ReduceSum(num_nodes_static=num_nodes),
        5: GCNStage5_Transform(feature_dim, feature_dim),  # in=500, out=500
        6: GCNStage6_Activate()
    }
    return models[stage_id]

def generate_dummy_input(stage_id, num_nodes, num_edges, feature_dim=500):
    """Generate dummy input for each GCN stage"""
    torch.manual_seed(42)

    if stage_id == 1:
        # Stage 1: COMPUTE_NORM - edge_index[2, E], num_nodes -> norm[E]
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        return (edge_index, torch.tensor(num_nodes))

    elif stage_id == 2:
        # Stage 2: GATHER - x[N, F], edge_index[2, E] -> x_j[E, F]
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        return (x, edge_index)

    elif stage_id == 3:
        # Stage 3: MESSAGE - x_j[E, F], norm[E] -> msg[E, F]
        x_j = torch.randn(num_edges, feature_dim)
        norm = torch.rand(num_edges) + 0.1  # normalization weights
        return (x_j, norm)

    elif stage_id == 4:
        # Stage 4: REDUCE_SUM - msg[E, F], edge_index[2, E], num_nodes -> agg[N, F]
        msg = torch.randn(num_edges, feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        return (msg, edge_index, torch.tensor(num_nodes))

    elif stage_id == 5:
        # Stage 5: TRANSFORM - agg[N, F] -> out[N, F']
        agg = torch.randn(num_nodes, feature_dim)
        return (agg,)

    elif stage_id == 6:
        # Stage 6: ACTIVATE - out[N, F'] -> h[N, F']
        out = torch.randn(num_nodes, feature_dim)
        return (out,)

    else:
        raise ValueError(f"Invalid stage_id: {stage_id}")

def estimate_data_size(stage_id, num_nodes, num_edges, feature_dim=500):
    """Estimate input + output data size (bytes)"""
    bytes_per_float = 4
    bytes_per_int = 8

    if stage_id == 1:
        # Input: edge_index[2, E], Output: norm[E]
        input_size = 2 * num_edges * bytes_per_int
        output_size = num_edges * bytes_per_float

    elif stage_id == 2:
        # Input: x[N, F] + edge_index[2, E], Output: x_j[E, F]
        input_size = num_nodes * feature_dim * bytes_per_float + 2 * num_edges * bytes_per_int
        output_size = num_edges * feature_dim * bytes_per_float

    elif stage_id == 3:
        # Input: x_j[E, F] + norm[E], Output: msg[E, F]
        input_size = num_edges * feature_dim * bytes_per_float + num_edges * bytes_per_float
        output_size = num_edges * feature_dim * bytes_per_float

    elif stage_id == 4:
        # Input: msg[E, F] + edge_index[2, E], Output: agg[N, F]
        input_size = num_edges * feature_dim * bytes_per_float + 2 * num_edges * bytes_per_int
        output_size = num_nodes * feature_dim * bytes_per_float

    elif stage_id == 5:
        # Input: agg[N, F], Output: out[N, F']
        input_size = num_nodes * feature_dim * bytes_per_float
        output_size = num_nodes * feature_dim * bytes_per_float

    elif stage_id == 6:
        # Input: out[N, F'], Output: h[N, F']
        input_size = num_nodes * feature_dim * bytes_per_float
        output_size = num_nodes * feature_dim * bytes_per_float

    else:
        raise ValueError(f"Invalid stage_id: {stage_id}")

    return input_size + output_size

# ============================================================================
# Model Export Functions
# ============================================================================

def get_dynamic_axes_for_stage(stage_id):
    """
    Get dynamic_axes and input_names for each GCN stage

    Returns:
        tuple: (dynamic_axes_dict, input_names_list)
    """
    if stage_id == 1:
        # Stage 1: edge_index[2, E], num_nodes
        dynamic_axes = {
            'edge_index': {1: 'num_edges'}
        }
        input_names = ['edge_index', 'num_nodes']

    elif stage_id == 2:
        # Stage 2: x[N, F], edge_index[2, E]
        dynamic_axes = {
            'x': {0: 'num_nodes'},
            'edge_index': {1: 'num_edges'}
        }
        input_names = ['x', 'edge_index']

    elif stage_id == 3:
        # Stage 3: x_j[E, F], norm[E]
        dynamic_axes = {
            'x_j': {0: 'num_edges'},
            'norm': {0: 'num_edges'}
        }
        input_names = ['x_j', 'norm']

    elif stage_id == 4:
        # Stage 4: msg[E, F], edge_index[2, E], num_nodes
        dynamic_axes = {
            'msg': {0: 'num_edges'},
            'edge_index': {1: 'num_edges'}
        }
        input_names = ['msg', 'edge_index', 'num_nodes']

    elif stage_id == 5:
        # Stage 5: agg[N, F]
        dynamic_axes = {
            'agg': {0: 'num_nodes'}
        }
        input_names = ['agg']

    elif stage_id == 6:
        # Stage 6: out[N, F']
        dynamic_axes = {
            'out': {0: 'num_nodes'}
        }
        input_names = ['out']

    else:
        raise ValueError(f"Invalid stage_id: {stage_id}")

    return dynamic_axes, input_names

def export_dynamic_models():
    """Export CPU/GPU dynamic models"""
    print("=" * 70)
    print("=== Exporting GCN Dynamic Models (CPU/GPU) ===")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for stage_id in range(1, NUM_STAGES + 1):
        print(f"\nStage {stage_id}:")
        model = get_stage_model(stage_id)
        model.eval()

        # Use medium size for dummy input
        dummy_input = generate_dummy_input(stage_id, 5000, 5000)

        # Get dynamic_axes config
        dynamic_axes, input_names = get_dynamic_axes_for_stage(stage_id)

        # Export ONNX (dynamic shape)
        onnx_path = MODELS_DIR / f"stage{stage_id}_dynamic.onnx"
        print(f"  Exporting ONNX: {onnx_path.name}")

        try:
            with torch.no_grad():
                torch.onnx.export(
                    model, dummy_input, str(onnx_path),
                    input_names=input_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=17,
                    do_constant_folding=True
                )
        except Exception as e:
            print(f"    Warning: ONNX export failed: {e}")
            continue

        # Convert to CPU IR
        cpu_ir = MODELS_DIR / f"stage{stage_id}_cpu.xml"
        print(f"  Converting to CPU IR: {cpu_ir.name}")
        convert_to_ir(onnx_path, cpu_ir, 'CPU')

        # Convert to GPU IR
        gpu_ir = MODELS_DIR / f"stage{stage_id}_gpu.xml"
        print(f"  Converting to GPU IR: {gpu_ir.name}")
        convert_to_ir(onnx_path, gpu_ir, 'GPU')

        print(f"  Done: Stage {stage_id} dynamic models exported")

    print(f"\nDone: All GCN dynamic models exported ({NUM_STAGES * 2} IR files)")

def export_npu_static_models(test_cases):
    """Export NPU static models (one per size)

    Note: NPU does not support Stage 1/4 (scatter operations), these are skipped
    """
    print("\n" + "=" * 70)
    print("=== Exporting GCN NPU Static Models ===")
    print("=" * 70)

    npu_stages = [s for s in range(1, NUM_STAGES + 1) if s not in NPU_SKIP_STAGES]
    print(f"Total: {len(npu_stages)} stages x {len(test_cases)} sizes = {len(npu_stages) * len(test_cases)} models")
    print(f"Skipping stages: {NPU_SKIP_STAGES} (scatter operations)")
    print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(npu_stages) * len(test_cases)
    count = 0

    for stage_id in range(1, NUM_STAGES + 1):
        # Skip scatter stages for NPU
        if stage_id in NPU_SKIP_STAGES:
            continue

        for case in test_cases:
            count += 1
            nodes, edges = case['nodes'], case['edges']
            print(f"[{count}/{total}] Stage {stage_id} - NPU n{nodes}_e{edges}", end=' ')

            # For scatter stages (if ever enabled), pass num_nodes
            if stage_id in [1, 4]:
                model = get_stage_model(stage_id, num_nodes=nodes)
            else:
                model = get_stage_model(stage_id)
            model.eval()

            # Generate exact size dummy input
            dummy_input = generate_dummy_input(stage_id, nodes, edges)

            # Export static ONNX
            onnx_path = MODELS_DIR / f"stage{stage_id}_npu_n{nodes}_e{edges}.onnx"

            try:
                with torch.no_grad():
                    torch.onnx.export(
                        model, dummy_input, str(onnx_path),
                        opset_version=17,
                        do_constant_folding=True
                    )
            except Exception as e:
                print(f"Warning: Failed: {e}")
                continue

            # Convert to NPU IR (static shape)
            npu_ir = MODELS_DIR / f"stage{stage_id}_npu_n{nodes}_e{edges}.xml"
            success = convert_to_ir(onnx_path, npu_ir, 'NPU', static_shape=(nodes, edges))

            if success:
                print("Done")
            else:
                print("Warning")

    print(f"\nDone: All GCN NPU static models exported ({total} files)")

def convert_to_ir(onnx_path, ir_path, device, static_shape=None):
    """Convert ONNX to OpenVINO IR"""
    try:
        from openvino.tools import mo
        from openvino import save_model

        if static_shape:
            # Static shape (NPU)
            nodes, edges = static_shape
            ov_model = mo.convert_model(str(onnx_path))
        else:
            # Dynamic shape (CPU/GPU)
            ov_model = mo.convert_model(str(onnx_path))

        save_model(ov_model, str(ir_path))
        return True

    except ImportError:
        print(f"    Warning: OpenVINO not available, skipping IR conversion")
        return False
    except Exception as e:
        print(f"    Warning: IR conversion failed: {e}")
        return False

# ============================================================================
# Latency Measurement Functions
# ============================================================================

def measure_latency_pytorch(model, dummy_input, num_warmup=10, num_iterations=50):
    """Measure latency using PyTorch (fallback)"""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(*dummy_input)

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(*dummy_input)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }

def measure_latency_openvino(ir_path, pu, dummy_input, num_warmup=10, num_iterations=50):
    """Measure latency using OpenVINO"""
    try:
        import openvino.runtime as ov

        core = ov.Core()
        model = core.read_model(str(ir_path))
        compiled_model = core.compile_model(model, pu)

        # Prepare inputs (convert to numpy)
        inputs = [t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
                  for t in dummy_input]

        # Warmup
        for _ in range(num_warmup):
            _ = compiled_model(inputs)

        # Measure
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = compiled_model(inputs)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies)
        }

    except Exception as e:
        error_msg = str(e)
        print(f"    Warning: OpenVINO measurement failed: {error_msg}")
        return {
            'mean': -1,
            'std': -1,
            'min': -1,
            'max': -1,
            'failed': True,
            'error': error_msg
        }

def measure_all_latencies(test_cases, config, pu_list=None):
    """
    Measure latencies for specified PUs

    Args:
        test_cases: Test case list
        config: Configuration info
        pu_list: PU list to measure, e.g., ['CPU', 'GPU'] or ['NPU'], None means all
    """
    if pu_list is None:
        pu_list = ['CPU', 'GPU', 'NPU']

    pu_name = '+'.join(pu_list)
    print("\n" + "=" * 70)
    print(f"=== Measuring GCN Latencies ({pu_name}) ===")
    print("=" * 70)

    num_warmup = config['config']['num_warmup']
    num_iterations = config['config']['num_iterations']
    feature_dim = config['config']['feature_dim']

    results = {}

    # Calculate total measurements (NPU skips stages 1/4)
    total = 0
    for pu in pu_list:
        if pu == 'NPU':
            total += (NUM_STAGES - len(NPU_SKIP_STAGES)) * len(test_cases)
        else:
            total += NUM_STAGES * len(test_cases)
    count = 0

    # Measure CPU/GPU (dynamic models)
    if 'CPU' in pu_list or 'GPU' in pu_list:
        dynamic_pus = [pu for pu in ['CPU', 'GPU'] if pu in pu_list]
        for stage_id in range(1, NUM_STAGES + 1):
            for pu in dynamic_pus:
                ir_path = MODELS_DIR / f"stage{stage_id}_{pu.lower()}.xml"

                if not ir_path.exists():
                    print(f"Warning: Skipping {pu} Stage {stage_id}: IR not found")
                    continue

                for case in test_cases:
                    count += 1
                    nodes, edges = case['nodes'], case['edges']
                    print(f"[{count}/{total}] Stage {stage_id} on {pu} - {nodes}n {edges}e... ", end='', flush=True)

                    dummy_input = generate_dummy_input(stage_id, nodes, edges, feature_dim)
                    result = measure_latency_openvino(ir_path, pu, dummy_input, num_warmup, num_iterations)

                    key = (nodes, edges, pu, stage_id)
                    results[key] = result
                    print(f"{result['mean']:.2f}ms +/-{result['std']:.2f}")

    # Measure NPU (static models, one per size)
    if 'NPU' in pu_list:
        for stage_id in range(1, NUM_STAGES + 1):
            # Skip scatter stages for NPU
            if stage_id in NPU_SKIP_STAGES:
                continue
            for case in test_cases:
                count += 1
                nodes, edges = case['nodes'], case['edges']
                print(f"[{count}/{total}] Stage {stage_id} on NPU - {nodes}n {edges}e... ", end='', flush=True)

                ir_path = MODELS_DIR / f"stage{stage_id}_npu_n{nodes}_e{edges}.xml"

                if not ir_path.exists():
                    print("Warning: IR not found")
                    continue

                dummy_input = generate_dummy_input(stage_id, nodes, edges, feature_dim)
                result = measure_latency_openvino(ir_path, 'NPU', dummy_input, num_warmup, num_iterations)

                key = (nodes, edges, 'NPU', stage_id)
                results[key] = result
                print(f"{result['mean']:.2f}ms +/-{result['std']:.2f}")

    print(f"\nDone: Measured {len(results)} configurations for {pu_name}")
    return results

# ============================================================================
# Bandwidth Estimation Functions
# ============================================================================

def estimate_bandwidth_and_compute_time(raw_results, feature_dim=500):
    """Estimate bandwidth via linear regression and separate compute time"""
    print("\n" + "=" * 70)
    print("=== Estimating GCN Bandwidth and Compute Time ===")
    print("=" * 70)

    bandwidth_table = {}
    compute_table = {}

    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("Warning: sklearn not available, skipping bandwidth regression")
        print("  Saving raw latencies as compute time (no separation)")
        for key, result in raw_results.items():
            compute_table[f"{key[0]},{key[1]},{key[2]},{key[3]}"] = {
                'compute_time_ms': result['mean'],
                'std_ms': result['std']
            }
        return {}, compute_table

    for stage_id in range(1, NUM_STAGES + 1):
        for pu in ['CPU', 'GPU', 'NPU']:
            # Collect all data points for this (stage, pu)
            data_points = []
            for key, result in raw_results.items():
                nodes, edges, measured_pu, measured_stage = key
                if measured_pu == pu and measured_stage == stage_id:
                    data_size = estimate_data_size(stage_id, nodes, edges, feature_dim)
                    latency = result['mean']
                    data_points.append((data_size, latency))

            if len(data_points) < 3:
                print(f"  Warning: Insufficient data for {pu} Stage {stage_id}, skipping regression")
                continue

            # Linear regression: latency = a * data_size + b
            X = np.array([p[0] for p in data_points]).reshape(-1, 1)
            y = np.array([p[1] for p in data_points])

            reg = LinearRegression().fit(X, y)

            # Bandwidth (MB/s)
            bandwidth_coef = reg.coef_[0]
            if bandwidth_coef > 0:
                bandwidth = (1000 / bandwidth_coef) / (1024 * 1024)  # bytes/ms -> MB/s
            else:
                bandwidth = float('inf')

            compute_time_base = reg.intercept_

            bandwidth_table[f"{pu}_stage{stage_id}"] = bandwidth
            print(f"  {pu} Stage {stage_id}: {bandwidth:.2f} MB/s (R2={reg.score(X, y):.3f})")

            # Subtract transfer time for each test point
            for key, result in raw_results.items():
                nodes, edges, measured_pu, measured_stage = key
                if measured_pu == pu and measured_stage == stage_id:
                    data_size = estimate_data_size(stage_id, nodes, edges, feature_dim)
                    if bandwidth != float('inf'):
                        transfer_time = (data_size / (1024 * 1024)) / bandwidth * 1000  # ms
                    else:
                        transfer_time = 0

                    compute_time = max(result['mean'] - transfer_time, 0)

                    key_str = f"{nodes},{edges},{pu},{stage_id}"
                    compute_table[key_str] = {
                        'compute_time_ms': compute_time,
                        'transfer_time_ms': transfer_time,
                        'total_time_ms': result['mean'],
                        'std_ms': result['std']
                    }

    print(f"Done: Bandwidth estimation completed")
    return bandwidth_table, compute_table

# ============================================================================
# Checkpoint and Result Saving Functions
# ============================================================================

def save_checkpoint(raw_results, stage_name):
    """Save intermediate checkpoint (prevent data loss)"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = RESULTS_DIR / f'checkpoint_{stage_name}.json'

    # Convert tuple keys to strings for JSON
    raw_serializable = {f"{k[0]},{k[1]},{k[2]},{k[3]}": v for k, v in raw_results.items()}

    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(raw_serializable, f, indent=2)

    print(f"Done: Checkpoint saved: {checkpoint_file.name} ({len(raw_results)} entries)")
    return checkpoint_file

def load_checkpoint(stage_name):
    """Load checkpoint (resume from breakpoint)"""
    checkpoint_file = RESULTS_DIR / f'checkpoint_{stage_name}.json'

    if not checkpoint_file.exists():
        return None

    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        raw_serializable = json.load(f)

    # Convert string keys back to tuples
    raw_results = {}
    for key_str, value in raw_serializable.items():
        nodes, edges, pu, stage = key_str.split(',')
        raw_results[(int(nodes), int(edges), pu, int(stage))] = value

    print(f"Done: Loaded checkpoint: {checkpoint_file.name} ({len(raw_results)} entries)")
    return raw_results

def save_results(raw_results, lookup_table, bandwidth_table):
    """Save all results"""
    print("\n" + "=" * 70)
    print("=== Saving GCN Results ===")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Raw measurements
    raw_file = RESULTS_DIR / 'raw_measurements.json'
    raw_serializable = {f"{k[0]},{k[1]},{k[2]},{k[3]}": v for k, v in raw_results.items()}
    with open(raw_file, 'w', encoding='utf-8') as f:
        json.dump(raw_serializable, f, indent=2)
    print(f"  Done: Saved: {raw_file}")

    # 2. Lookup table (compute time only)
    lookup_file = RESULTS_DIR / 'lookup_table.json'
    with open(lookup_file, 'w', encoding='utf-8') as f:
        json.dump(lookup_table, f, indent=2)
    print(f"  Done: Saved: {lookup_file}")

    # 3. Bandwidth table
    bandwidth_file = RESULTS_DIR / 'bandwidth_table.json'
    with open(bandwidth_file, 'w', encoding='utf-8') as f:
        json.dump(bandwidth_table, f, indent=2)
    print(f"  Done: Saved: {bandwidth_file}")

def generate_report(lookup_table, bandwidth_table, test_cases):
    """Generate readable statistics report"""
    report = []
    report.append("=" * 80)
    report.append("GNX GCN Stage Profiling Report")
    report.append("=" * 80)
    report.append("")

    # Configuration info
    report.append("Test Configuration:")
    report.append(f"  - Test cases: {len(test_cases)}")
    report.append(f"  - Node sizes: {sorted(set(c['nodes'] for c in test_cases))}")
    report.append(f"  - Edge sizes: {sorted(set(c['edges'] for c in test_cases))}")
    report.append(f"  - Total measurements: {len(lookup_table)}")
    report.append(f"  - NPU skip stages: {NPU_SKIP_STAGES} (scatter operations)")
    report.append("")

    # GCN stage descriptions
    report.append("GCN 6-Stage Decomposition:")
    report.append("  Stage 1: COMPUTE_NORM - Calculate edge normalization [SCATTER]")
    report.append("  Stage 2: GATHER       - Collect source node features")
    report.append("  Stage 3: MESSAGE      - Apply normalization to messages")
    report.append("  Stage 4: REDUCE_SUM   - Aggregate messages [SCATTER]")
    report.append("  Stage 5: TRANSFORM    - Linear transformation")
    report.append("  Stage 6: ACTIVATE     - ReLU activation")
    report.append("")

    # Average compute time per stage
    report.append("Average Compute Time by Stage (ms):")
    report.append("-" * 80)
    report.append(f"{'Stage':<10} {'CPU':<15} {'GPU':<15} {'NPU':<15}")
    report.append("-" * 80)

    for stage_id in range(1, NUM_STAGES + 1):
        avgs = {}
        for pu in ['CPU', 'GPU', 'NPU']:
            times = [v['compute_time_ms'] for k, v in lookup_table.items()
                     if k.endswith(f",{pu},{stage_id}")]
            avgs[pu] = np.mean(times) if times else 0

        npu_val = f"{avgs['NPU']:.2f}" if stage_id not in NPU_SKIP_STAGES else "N/A (scatter)"
        report.append(f"Stage {stage_id:<3} {avgs['CPU']:<15.2f} {avgs['GPU']:<15.2f} {npu_val:<15}")

    # Bandwidth statistics
    report.append("")
    report.append("Estimated Bandwidth by Stage (MB/s):")
    report.append("-" * 80)

    for stage_id in range(1, NUM_STAGES + 1):
        bws = {}
        for pu in ['CPU', 'GPU', 'NPU']:
            key = f"{pu}_stage{stage_id}"
            bws[pu] = bandwidth_table.get(key, 0)

        if any(bws.values()):
            npu_val = f"{bws['NPU']:.1f}" if stage_id not in NPU_SKIP_STAGES else "N/A"
            report.append(f"Stage {stage_id}: CPU={bws['CPU']:.1f} | GPU={bws['GPU']:.1f} | NPU={npu_val}")

    # Speedup analysis
    report.append("")
    report.append("Speedup Analysis (GPU vs CPU, NPU vs CPU):")
    report.append("-" * 80)

    for stage_id in range(1, NUM_STAGES + 1):
        cpu_times = [v['compute_time_ms'] for k, v in lookup_table.items()
                     if k.endswith(f",CPU,{stage_id}")]
        gpu_times = [v['compute_time_ms'] for k, v in lookup_table.items()
                     if k.endswith(f",GPU,{stage_id}")]
        npu_times = [v['compute_time_ms'] for k, v in lookup_table.items()
                     if k.endswith(f",NPU,{stage_id}")]

        line = f"Stage {stage_id}:"
        if cpu_times and gpu_times:
            gpu_speedup = np.mean(cpu_times) / np.mean(gpu_times)
            line += f" GPU speedup = {gpu_speedup:.2f}x"
        if cpu_times and npu_times:
            npu_speedup = np.mean(cpu_times) / np.mean(npu_times)
            line += f", NPU speedup = {npu_speedup:.2f}x"
        if cpu_times and (gpu_times or npu_times):
            report.append(line)

    report.append("")
    report.append("=" * 80)

    # Save report
    report_file = RESULTS_DIR / 'profiling_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print("\n" + '\n'.join(report))
    print(f"\nDone: Report saved: {report_file}")

# ============================================================================
# Main Function
# ============================================================================

def merge_npu_checkpoints():
    """Merge all NPU checkpoint files from individual (nodes, stage) tests"""
    print("\n" + "=" * 70)
    print("=== Merging GCN NPU Checkpoints ===")
    print("=" * 70)

    merged = {}
    checkpoint_files = list(RESULTS_DIR.glob("npu_stage*_n*.json"))

    if not checkpoint_files:
        print("Warning: No NPU checkpoint files found")
        return None

    for ckpt_file in sorted(checkpoint_files):
        print(f"  Loading: {ckpt_file.name}")
        with open(ckpt_file, 'r') as f:
            data = json.load(f)
            # Convert string keys to tuple keys for consistency
            for key, value in data.items():
                parts = key.split(',')
                nodes, edges, pu, stage = int(parts[0]), int(parts[1]), parts[2], int(parts[3])
                # Only include successful measurements
                if not value.get('failed', False):
                    merged[(nodes, edges, pu, stage)] = value
                else:
                    print(f"    Skipping failed: {key}")

    print(f"\nDone: Merged {len(merged)} successful NPU measurements from {len(checkpoint_files)} files")

    # Save merged checkpoint
    save_checkpoint(merged, 'npu')
    return merged

def main():
    parser = argparse.ArgumentParser(description='GNX GCN Stage Profiling Script')
    parser.add_argument('--all', action='store_true', help='Export all + measure CPU/GPU')
    parser.add_argument('--export', action='store_true', help='Export all models (CPU/GPU dynamic + NPU static)')
    parser.add_argument('--export-cpugpu', action='store_true', help='Export CPU/GPU dynamic models only')
    parser.add_argument('--export-npu', action='store_true', help='Export NPU static models only')
    parser.add_argument('--measure', action='store_true', help='Measure CPU/GPU latencies only')
    parser.add_argument('--measure-cpugpu', action='store_true', help='Measure CPU/GPU latencies only (alias)')
    parser.add_argument('--merge-npu', action='store_true', help='Merge NPU results from profile_npu.py outputs')
    parser.add_argument('--analyze', action='store_true', help='Analyze and generate results only')

    args = parser.parse_args()

    # Handle aliases
    if args.measure_cpugpu:
        args.measure = True

    # Load configuration
    config = load_config()
    test_cases = config['test_cases']

    print("=" * 70)
    print("GNX GCN Stage Profiling - Incremental Pipeline")
    print("=" * 70)
    print(f"Test cases: {len(test_cases)}")
    print(f"Node sizes: {sorted(set(c['nodes'] for c in test_cases))}")
    print(f"GCN Stages: {NUM_STAGES} (NPU skips: {NPU_SKIP_STAGES})")
    print()
    print("Workflow:")
    print("  1. python gcn_profile_stages.py --export          # Export all models")
    print("  2. python gcn_profile_stages.py --measure         # Measure CPU/GPU")
    print("  3. run_profiling.bat                              # Run NPU tests (isolated)")
    print("  4. python gcn_profile_stages.py --merge-npu       # Merge NPU results")
    print("  5. python gcn_profile_stages.py --analyze         # Generate final report")
    print()

    # ========================================================================
    # Handle --export-cpugpu (CPU/GPU dynamic models only)
    # ========================================================================
    if args.export_cpugpu:
        print("\n" + "=" * 70)
        print("Exporting GCN CPU/GPU Dynamic Models Only")
        print("=" * 70)
        export_dynamic_models()
        print("\nDone: CPU/GPU dynamic models exported!")
        return

    # ========================================================================
    # Handle --export-npu (NPU static models only)
    # ========================================================================
    if args.export_npu:
        print("\n" + "=" * 70)
        print("Exporting GCN NPU Static Models Only")
        print("=" * 70)
        export_npu_static_models(test_cases)
        print("\nDone: NPU static models exported!")
        return

    # ========================================================================
    # Handle --merge-npu (Merge NPU checkpoint files)
    # ========================================================================
    if args.merge_npu:
        npu_results = merge_npu_checkpoints()
        if npu_results:
            print(f"\nDone: NPU results merged into checkpoint_npu.json")
            print(f"  Total successful measurements: {len(npu_results)}")
        return

    # ========================================================================
    # Handle --export (Export all models)
    # ========================================================================
    if args.export or args.all:
        print("\n" + "=" * 70)
        print("PHASE 1: Exporting All GCN Models")
        print("=" * 70)

        print("\n[1/2] Exporting CPU/GPU dynamic models...")
        export_dynamic_models()

        print("\n[2/2] Exporting NPU static models...")
        export_npu_static_models(test_cases)

        print("\nDone: All models exported!")

        if args.export and not args.all:
            return

    # ========================================================================
    # Handle --measure (Measure CPU/GPU only)
    # ========================================================================
    if args.measure or args.all:
        print("\n" + "=" * 70)
        print("PHASE 2: Measuring GCN CPU/GPU Latencies")
        print("=" * 70)
        print("Note: NPU measurements should be done via profile_npu.py for isolation")

        cpugpu_results = measure_all_latencies(test_cases, config, pu_list=['CPU', 'GPU'])
        save_checkpoint(cpugpu_results, 'cpugpu')

        print("\nDone: CPU/GPU measurements completed and saved!")
        print(f"  Total: {len(cpugpu_results)} entries")
        print("\nNext steps:")
        print("  1. Run NPU tests: run_profiling.bat (or manually via profile_npu.py)")
        print("  2. Merge NPU results: python gcn_profile_stages.py --merge-npu")
        print("  3. Generate report: python gcn_profile_stages.py --analyze")

        if args.measure and not args.all:
            return

    # ========================================================================
    # Handle --analyze (Generate final results)
    # ========================================================================
    if args.analyze or args.all:
        print("\n" + "=" * 70)
        print("PHASE 3: Analyzing GCN Results")
        print("=" * 70)

        # Load all checkpoints
        cpugpu_data = load_checkpoint('cpugpu')
        npu_data = load_checkpoint('npu')

        if cpugpu_data is None:
            print("ERROR: CPU/GPU checkpoint not found.")
            print("  Run: python gcn_profile_stages.py --measure")
            return

        # Merge results
        all_results = cpugpu_data.copy()
        if npu_data is not None:
            all_results.update(npu_data)
            print(f"Done: Merged: {len(cpugpu_data)} CPU/GPU + {len(npu_data)} NPU = {len(all_results)} total")
        else:
            print(f"WARNING: NPU data not found, using CPU/GPU only ({len(all_results)} entries)")
            print("  Run NPU tests: run_profiling.bat")
            print("  Then merge: python gcn_profile_stages.py --merge-npu")

        # Estimate bandwidth and compute time
        bandwidth_table, lookup_table = estimate_bandwidth_and_compute_time(
            all_results, config['config']['feature_dim']
        )

        # Save final results
        save_results(all_results, lookup_table, bandwidth_table)
        generate_report(lookup_table, bandwidth_table, test_cases)

    # Final summary (only if something was done)
    if args.all or args.export or args.measure or args.analyze or args.merge_npu:
        print("\n" + "=" * 70)
        print("GCN Profiling Session Complete")
        print("=" * 70)
        print(f"\nResults in: {RESULTS_DIR}")
        print(f"  - lookup_table.json      (compute times)")
        print(f"  - bandwidth_table.json   (bandwidth estimates)")
        print(f"  - profiling_report.txt   (summary report)")
        print(f"\nCheckpoints:")
        print(f"  - checkpoint_cpugpu.json (CPU/GPU data)")
        print(f"  - checkpoint_npu.json    (NPU data)")
    else:
        # No flags provided, show help
        parser.print_help()

if __name__ == '__main__':
    main()
