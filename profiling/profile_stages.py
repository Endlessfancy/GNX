"""
GNX Stage Profiling Script (精简版)

自动化profiling流程：
1. 导出模型：CPU/GPU动态模型 + NPU静态模型
2. 测量延迟：在不同input size和PU上测试
3. 带宽估计：回归分离计算和传输时间
4. 生成结果：lookup_table.json + bandwidth_table.json + report.txt

Usage:
    python profile_stages.py --all              # 运行全部流程
    python profile_stages.py --export           # 只导出模型
    python profile_stages.py --measure          # 只测量延迟
    python profile_stages.py --analyze          # 只分析数据
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
import numpy as np

# Add current directory to path for importing models
# Expected structure: models/ is inside profiling/ directory
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
# from torch_geometric.data import Data  # Not used, commented out

# Import stage models
from models.Model_sage import (
    SAGEStage1_Gather,
    SAGEStage2_Message,
    SAGEStage3_ReduceSum,
    SAGEStage4_ReduceCount,
    SAGEStage5_Normalize,
    SAGEStage6_Transform,
    SAGEStage7_Activate
)

# ============================================================================
# Configuration
# ============================================================================

PROFILING_DIR = Path(__file__).parent
MODELS_DIR = PROFILING_DIR / 'exported_models'
RESULTS_DIR = PROFILING_DIR / 'results'
TEST_CASES_FILE = PROFILING_DIR / 'test_cases.json'

# ============================================================================
# Helper Functions
# ============================================================================

def load_config():
    """加载测试配置"""
    with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def get_stage_model(stage_id, num_nodes=None):
    """
    获取stage模型

    Args:
        stage_id: stage编号 (1-7)
        num_nodes: 静态节点数（仅用于Stage 3 NPU导出）
    """
    models = {
        1: SAGEStage1_Gather(),
        2: SAGEStage2_Message(),
        3: SAGEStage3_ReduceSum(num_nodes_static=num_nodes),  # Pass num_nodes for NPU
        4: SAGEStage4_ReduceCount(),
        5: SAGEStage5_Normalize(),
        6: SAGEStage6_Transform(500, 500),  # in_dim=500, out_dim=500
        7: SAGEStage7_Activate()
    }
    return models[stage_id]

def generate_dummy_input(stage_id, num_nodes, num_edges, feature_dim=500):
    """为每个stage生成dummy输入"""
    torch.manual_seed(42)

    if stage_id == 1:
        # Stage 1: (x, edge_index)
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        return (x, edge_index)

    elif stage_id == 2:
        # Stage 2: x_j [num_edges, feature_dim]
        return torch.randn(num_edges, feature_dim)

    elif stage_id == 3:
        # Stage 3: (messages, edge_index, num_nodes)
        messages = torch.randn(num_edges, feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        return (messages, edge_index, torch.tensor(num_nodes))

    elif stage_id == 4:
        # Stage 4: (edge_index, num_nodes, num_edges)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        return (edge_index, torch.tensor(num_nodes), torch.tensor(num_edges))

    elif stage_id == 5:
        # Stage 5: (sum_agg, count)
        sum_agg = torch.randn(num_nodes, feature_dim)
        count = torch.rand(num_nodes) + 1.0  # avoid zeros
        return (sum_agg, count)

    elif stage_id == 6:
        # Stage 6: (mean_agg, x)
        mean_agg = torch.randn(num_nodes, feature_dim)
        x = torch.randn(num_nodes, feature_dim)
        return (mean_agg, x)

    elif stage_id == 7:
        # Stage 7: out
        return torch.randn(num_nodes, feature_dim)

    else:
        raise ValueError(f"Invalid stage_id: {stage_id}")

def estimate_data_size(stage_id, num_nodes, num_edges, feature_dim=500):
    """估计输入+输出的数据大小（bytes）"""
    bytes_per_float = 4
    bytes_per_int = 8

    if stage_id == 1:
        # Input: x[N,F] + edge_index[2,E], Output: x_j[E,F]
        input_size = num_nodes * feature_dim * bytes_per_float + 2 * num_edges * bytes_per_int
        output_size = num_edges * feature_dim * bytes_per_float

    elif stage_id == 2:
        # Input: x_j[E,F], Output: messages[E,F]
        input_size = num_edges * feature_dim * bytes_per_float
        output_size = num_edges * feature_dim * bytes_per_float

    elif stage_id == 3:
        # Input: messages[E,F] + edge_index[2,E], Output: sum_agg[N,F]
        input_size = num_edges * feature_dim * bytes_per_float + 2 * num_edges * bytes_per_int
        output_size = num_nodes * feature_dim * bytes_per_float

    elif stage_id == 4:
        # Input: edge_index[2,E], Output: count[N]
        input_size = 2 * num_edges * bytes_per_int
        output_size = num_nodes * bytes_per_float

    elif stage_id == 5:
        # Input: sum_agg[N,F] + count[N], Output: mean_agg[N,F]
        input_size = num_nodes * feature_dim * bytes_per_float + num_nodes * bytes_per_float
        output_size = num_nodes * feature_dim * bytes_per_float

    elif stage_id == 6:
        # Input: mean_agg[N,F] + x[N,F], Output: out[N,F]
        input_size = 2 * num_nodes * feature_dim * bytes_per_float
        output_size = num_nodes * feature_dim * bytes_per_float

    elif stage_id == 7:
        # Input: out[N,F], Output: activated[N,F]
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
    获取每个stage的dynamic_axes和input_names配置

    Returns:
        tuple: (dynamic_axes_dict, input_names_list)
    """
    if stage_id == 1:
        # Stage 1: (x[N,F], edge_index[2,E])
        dynamic_axes = {
            'x': {0: 'num_nodes'},
            'edge_index': {1: 'num_edges'}
        }
        input_names = ['x', 'edge_index']

    elif stage_id == 2:
        # Stage 2: x_j[E,F]
        dynamic_axes = {
            'x_j': {0: 'num_edges'}
        }
        input_names = ['x_j']

    elif stage_id == 3:
        # Stage 3: (messages[E,F], edge_index[2,E], num_nodes)
        dynamic_axes = {
            'messages': {0: 'num_edges'},
            'edge_index': {1: 'num_edges'}
        }
        input_names = ['messages', 'edge_index', 'num_nodes']

    elif stage_id == 4:
        # Stage 4: (edge_index[2,E], num_nodes, num_edges)
        dynamic_axes = {
            'edge_index': {1: 'num_edges'}
        }
        input_names = ['edge_index', 'num_nodes', 'num_edges']

    elif stage_id == 5:
        # Stage 5: (sum_agg[N,F], count[N])
        dynamic_axes = {
            'sum_agg': {0: 'num_nodes'},
            'count': {0: 'num_nodes'}
        }
        input_names = ['sum_agg', 'count']

    elif stage_id == 6:
        # Stage 6: (mean_agg[N,F], x[N,F])
        dynamic_axes = {
            'mean_agg': {0: 'num_nodes'},
            'x': {0: 'num_nodes'}
        }
        input_names = ['mean_agg', 'x']

    elif stage_id == 7:
        # Stage 7: out[N,F]
        dynamic_axes = {
            'out': {0: 'num_nodes'}
        }
        input_names = ['out']

    else:
        raise ValueError(f"Invalid stage_id: {stage_id}")

    return dynamic_axes, input_names

def export_dynamic_models():
    """导出CPU/GPU动态模型"""
    print("=" * 70)
    print("=== Exporting Dynamic Models (CPU/GPU) ===")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for stage_id in range(1, 8):
        if stage_id == 2:
                continue
        print(f"\nStage {stage_id}:")
        model = get_stage_model(stage_id)
        model.eval()

        # 使用中等大小作为dummy input
        dummy_input = generate_dummy_input(stage_id, 5000, 5000)

        # 获取dynamic_axes配置
        dynamic_axes, input_names = get_dynamic_axes_for_stage(stage_id)

        # 导出ONNX（动态shape）
        onnx_path = MODELS_DIR / f"stage{stage_id}_dynamic.onnx"
        print(f"  Exporting ONNX: {onnx_path.name}")

        try:
            with torch.no_grad():
                if isinstance(dummy_input, tuple):
                    torch.onnx.export(
                        model, dummy_input, str(onnx_path),
                        input_names=input_names,
                        dynamic_axes=dynamic_axes,
                        opset_version=17,
                        do_constant_folding=True
                    )
                else:
                    torch.onnx.export(
                        model, (dummy_input,), str(onnx_path),
                        input_names=input_names,
                        dynamic_axes=dynamic_axes,
                        opset_version=17,
                        do_constant_folding=True
                    )
        except Exception as e:
            print(f"    ⚠ ONNX export failed: {e}")
            continue

        # 转换为CPU IR
        cpu_ir = MODELS_DIR / f"stage{stage_id}_cpu.xml"
        print(f"  Converting to CPU IR: {cpu_ir.name}")
        convert_to_ir(onnx_path, cpu_ir, 'CPU')

        # 转换为GPU IR
        gpu_ir = MODELS_DIR / f"stage{stage_id}_gpu.xml"
        print(f"  Converting to GPU IR: {gpu_ir.name}")
        convert_to_ir(onnx_path, gpu_ir, 'GPU')

        print(f"  ✓ Stage {stage_id} dynamic models exported")

    print("\n✓ All dynamic models exported (12 files)")

def export_npu_static_models(test_cases):
    """导出NPU静态模型（每个size一个）

    Note: NPU不支持Stage 3/4 (scatter_add操作)，且脚本跳过Stage 2
    """
    print("\n" + "=" * 70)
    print("=== Exporting NPU Static Models ===")
    print("=" * 70)
    print(f"Total: 4 stages × {len(test_cases)} sizes = {4 * len(test_cases)} models (skipping Stage 2/3/4)")
    print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    total = 4 * len(test_cases)
    count = 0

    for stage_id in range(1, 8):
        if stage_id == 2:
            continue
        # NPU不支持Stage 3/4 (scatter_add操作不支持)，且脚本跳过Stage 2
        if stage_id in [3, 4]:
            continue
        for case in test_cases:
            count += 1
            nodes, edges = case['nodes'], case['edges']
            print(f"[{count}/{total}] Stage {stage_id} - NPU n{nodes}_e{edges}", end=' ')

            # For Stage 3, pass num_nodes to avoid dynamic shape issues on NPU
            if stage_id == 3:
                model = get_stage_model(stage_id, num_nodes=nodes)
            else:
                model = get_stage_model(stage_id)
            model.eval()

            # 生成精确size的dummy input
            dummy_input = generate_dummy_input(stage_id, nodes, edges)

            # 导出静态ONNX
            onnx_path = MODELS_DIR / f"stage{stage_id}_npu_n{nodes}_e{edges}.onnx"

            try:
                with torch.no_grad():
                    if isinstance(dummy_input, tuple):
                        torch.onnx.export(
                            model, dummy_input, str(onnx_path),
                            opset_version=17,
                            do_constant_folding=True
                        )
                    else:
                        torch.onnx.export(
                            model, (dummy_input,), str(onnx_path),
                            opset_version=17,
                            do_constant_folding=True
                        )
            except Exception as e:
                print(f"⚠ Failed: {e}")
                continue

            # 转换为NPU IR（静态shape）
            npu_ir = MODELS_DIR / f"stage{stage_id}_npu_n{nodes}_e{edges}.xml"
            success = convert_to_ir(onnx_path, npu_ir, 'NPU', static_shape=(nodes, edges))

            if success:
                print("✓")
            else:
                print("⚠")

    print(f"\n✓ All NPU static models exported ({total} files)")

def convert_to_ir(onnx_path, ir_path, device, static_shape=None):
    """转换ONNX到OpenVINO IR"""
    try:
        from openvino.tools import mo
        from openvino import save_model

        if static_shape:
            # 静态shape（NPU）
            nodes, edges = static_shape
            # Note: 这里简化处理，实际可能需要更精确的shape定义
            ov_model = mo.convert_model(str(onnx_path))
        else:
            # 动态shape（CPU/GPU）
            ov_model = mo.convert_model(str(onnx_path))

        save_model(ov_model, str(ir_path))
        return True

    except ImportError:
        print(f"    ⚠ OpenVINO not available, skipping IR conversion")
        return False
    except Exception as e:
        print(f"    ⚠ IR conversion failed: {e}")
        return False

# ============================================================================
# Latency Measurement Functions
# ============================================================================

def measure_latency_pytorch(model, dummy_input, num_warmup=10, num_iterations=50):
    """使用PyTorch测量延迟（fallback）"""
    model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            if isinstance(dummy_input, tuple):
                _ = model(*dummy_input)
            else:
                _ = model(dummy_input)

    # 测量
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            if isinstance(dummy_input, tuple):
                _ = model(*dummy_input)
            else:
                _ = model(dummy_input)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }

def measure_latency_openvino(ir_path, pu, dummy_input, num_warmup=10, num_iterations=50):
    """使用OpenVINO异步API测量延迟（包含完整的CPU↔GPU数据传输时间）"""
    try:
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(ir_path))
        compiled_model = core.compile_model(model, pu)

        # 准备输入（转换为numpy）
        if isinstance(dummy_input, tuple):
            inputs = [t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
                     for t in dummy_input]
        else:
            inputs = [dummy_input.numpy() if isinstance(dummy_input, torch.Tensor)
                     else np.array(dummy_input)]

        # 创建推理请求
        infer_request = compiled_model.create_infer_request()

        # 预热（每次都重新设置tensor，模拟真实场景）
        for _ in range(num_warmup):
            for i in range(len(inputs)):
                infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))
            infer_request.start_async()
            infer_request.wait()

        # 测量（每次都重新设置tensor，确保包含CPU→GPU传输时间）
        latencies = []
        for _ in range(num_iterations):
            # 重新设置输入tensor，触发数据传输
            for i in range(len(inputs)):
                infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))

            start = time.perf_counter()
            infer_request.start_async()
            infer_request.wait()
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies)
        }

    except Exception as e:
        error_msg = str(e)
        print(f"    ⚠ OpenVINO measurement failed: {error_msg}")
        # Return failed result instead of fallback
        return {
            'mean': -1,
            'std': -1,
            'min': -1,
            'max': -1,
            'failed': True,
            'error': error_msg
        }

def save_checkpoint_incremental(results, checkpoint_name):
    """增量保存checkpoint（每测完一个点就保存）"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = RESULTS_DIR / f'checkpoint_{checkpoint_name}.json'

    # Convert tuple keys to strings for JSON
    raw_serializable = {f"{k[0]},{k[1]},{k[2]},{k[3]}": v for k, v in results.items()}

    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(raw_serializable, f, indent=2)

    return checkpoint_file

def measure_all_latencies(test_cases, config, pu_list=None, checkpoint_name=None):
    """
    测量指定PU的延迟（支持断点续测和增量保存）

    Args:
        test_cases: 测试用例列表
        config: 配置信息
        pu_list: 要测量的PU列表，如['CPU', 'GPU']或['NPU']，None表示全部
        checkpoint_name: checkpoint名称，用于增量保存和断点续测
    """
    if pu_list is None:
        pu_list = ['CPU', 'GPU', 'NPU']

    if checkpoint_name is None:
        checkpoint_name = '+'.join(pu_list).lower()

    pu_name = '+'.join(pu_list)
    print("\n" + "=" * 70)
    print(f"=== Measuring Latencies ({pu_name}) ===")
    print("=" * 70)

    num_warmup = config['config']['num_warmup']
    num_iterations = config['config']['num_iterations']
    feature_dim = config['config']['feature_dim']

    # 尝试加载已有checkpoint（断点续测）
    results = load_checkpoint(checkpoint_name)
    if results is None:
        results = {}
        print(f"Starting fresh measurement...")
    else:
        print(f"Resuming from checkpoint: {len(results)} existing entries")

    # 计算总测量次数 (CPU/GPU跳过Stage 2, NPU跳过Stage 2/3/4)
    total = 0
    for pu in pu_list:
        if pu == 'NPU':
            total += 4 * len(test_cases)  # NPU: 4 stages (skip 2/3/4)
        else:
            total += 6 * len(test_cases)  # CPU/GPU: 6 stages (skip 2)
    count = 0

    # 测量CPU/GPU（动态模型）
    if 'CPU' in pu_list or 'GPU' in pu_list:
        dynamic_pus = [pu for pu in ['CPU', 'GPU'] if pu in pu_list]
        for stage_id in range(1, 8):
            if stage_id == 2:
                continue
            for pu in dynamic_pus:
                ir_path = MODELS_DIR / f"stage{stage_id}_{pu.lower()}.xml"

                if not ir_path.exists():
                    print(f"⚠ Skipping {pu} Stage {stage_id}: IR not found")
                    continue

                for case in test_cases:
                    count += 1
                    nodes, edges = case['nodes'], case['edges']
                    key = (nodes, edges, pu, stage_id)

                    # 跳过已测量的（断点续测）
                    if key in results:
                        print(f"[{count}/{total}] Stage {stage_id} on {pu} - {nodes}n {edges}e... SKIP (cached)")
                        continue

                    print(f"[{count}/{total}] Stage {stage_id} on {pu} - {nodes}n {edges}e... ", end='', flush=True)

                    dummy_input = generate_dummy_input(stage_id, nodes, edges, feature_dim)
                    result = measure_latency_openvino(ir_path, pu, dummy_input, num_warmup, num_iterations)

                    results[key] = result
                    print(f"{result['mean']:.2f}ms ±{result['std']:.2f}")

                    # 增量保存checkpoint
                    save_checkpoint_incremental(results, checkpoint_name)

    # 测量NPU（静态模型，每个size一个模型，跳过Stage 2/3/4）
    if 'NPU' in pu_list:
        for stage_id in range(1, 8):
            # NPU不支持Stage 3/4 (scatter_add操作不支持)，且脚本跳过Stage 2
            if stage_id == 2:
                continue
            if stage_id in [3, 4]:
                continue
            for case in test_cases:
                count += 1
                nodes, edges = case['nodes'], case['edges']
                key = (nodes, edges, 'NPU', stage_id)

                # 跳过已测量的（断点续测）
                if key in results:
                    print(f"[{count}/{total}] Stage {stage_id} on NPU - {nodes}n {edges}e... SKIP (cached)")
                    continue

                print(f"[{count}/{total}] Stage {stage_id} on NPU - {nodes}n {edges}e... ", end='', flush=True)

                ir_path = MODELS_DIR / f"stage{stage_id}_npu_n{nodes}_e{edges}.xml"

                if not ir_path.exists():
                    print("IR not found")
                    continue

                dummy_input = generate_dummy_input(stage_id, nodes, edges, feature_dim)
                result = measure_latency_openvino(ir_path, 'NPU', dummy_input, num_warmup, num_iterations)

                results[key] = result
                print(f"{result['mean']:.2f}ms ±{result['std']:.2f}")

                # 增量保存checkpoint
                save_checkpoint_incremental(results, checkpoint_name)

    print(f"\n✓ Measured {len(results)} configurations for {pu_name}")
    return results

# ============================================================================
# Bandwidth Estimation Functions
# ============================================================================

def estimate_bandwidth_and_compute_time(raw_results, feature_dim=500):
    """通过线性回归估计带宽，并分离计算时间"""
    print("\n" + "=" * 70)
    print("=== Estimating Bandwidth and Compute Time ===")
    print("=" * 70)

    bandwidth_table = {}
    compute_table = {}

    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("⚠ sklearn not available, skipping bandwidth regression")
        print("  Saving raw latencies as compute time (no separation)")
        for key, result in raw_results.items():
            compute_table[f"{key[0]},{key[1]},{key[2]},{key[3]}"] = {
                'compute_time_ms': result['mean'],
                'std_ms': result['std']
            }
        return {}, compute_table

    for stage_id in range(1, 8):
        if stage_id == 2:
            continue
        for pu in ['CPU', 'GPU', 'NPU']:
            # 收集该(stage, pu)的所有数据点
            data_points = []
            for key, result in raw_results.items():
                nodes, edges, measured_pu, measured_stage = key
                if measured_pu == pu and measured_stage == stage_id:
                    data_size = estimate_data_size(stage_id, nodes, edges, feature_dim)
                    latency = result['mean']
                    data_points.append((data_size, latency))

            if len(data_points) < 3:
                print(f"  ⚠ Insufficient data for {pu} Stage {stage_id}, skipping regression")
                continue

            # 线性回归: latency = a * data_size + b
            X = np.array([p[0] for p in data_points]).reshape(-1, 1)
            y = np.array([p[1] for p in data_points])

            reg = LinearRegression().fit(X, y)

            # 带宽（MB/s）
            bandwidth_coef = reg.coef_[0]
            if bandwidth_coef > 0:
                bandwidth = (1000 / bandwidth_coef) / (1024 * 1024)  # bytes/ms -> MB/s
            else:
                bandwidth = float('inf')

            compute_time_base = reg.intercept_

            bandwidth_table[f"{pu}_stage{stage_id}"] = bandwidth
            print(f"  {pu} Stage {stage_id}: {bandwidth:.2f} MB/s (R²={reg.score(X, y):.3f})")

            # 为每个测试点减去传输时间
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

    print(f"✓ Bandwidth estimation completed")
    return bandwidth_table, compute_table

# ============================================================================
# Checkpoint and Result Saving Functions
# ============================================================================

def save_checkpoint(raw_results, stage_name):
    """保存中间checkpoint（防止数据丢失）"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = RESULTS_DIR / f'checkpoint_{stage_name}.json'

    # Convert tuple keys to strings for JSON
    raw_serializable = {f"{k[0]},{k[1]},{k[2]},{k[3]}": v for k, v in raw_results.items()}

    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(raw_serializable, f, indent=2)

    print(f"✓ Checkpoint saved: {checkpoint_file.name} ({len(raw_results)} entries)")
    return checkpoint_file

def load_checkpoint(stage_name):
    """加载checkpoint（断点续跑）"""
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

    print(f"✓ Loaded checkpoint: {checkpoint_file.name} ({len(raw_results)} entries)")
    return raw_results

def save_results(raw_results, lookup_table, bandwidth_table):
    """保存所有结果"""
    print("\n" + "=" * 70)
    print("=== Saving Results ===")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Raw measurements
    raw_file = RESULTS_DIR / 'raw_measurements.json'
    raw_serializable = {f"{k[0]},{k[1]},{k[2]},{k[3]}": v for k, v in raw_results.items()}
    with open(raw_file, 'w', encoding='utf-8') as f:
        json.dump(raw_serializable, f, indent=2)
    print(f"  ✓ Saved: {raw_file}")

    # 2. Lookup table (compute time only)
    lookup_file = RESULTS_DIR / 'lookup_table.json'
    with open(lookup_file, 'w', encoding='utf-8') as f:
        json.dump(lookup_table, f, indent=2)
    print(f"  ✓ Saved: {lookup_file}")

    # 3. Bandwidth table
    bandwidth_file = RESULTS_DIR / 'bandwidth_table.json'
    with open(bandwidth_file, 'w', encoding='utf-8') as f:
        json.dump(bandwidth_table, f, indent=2)
    print(f"  ✓ Saved: {bandwidth_file}")

def generate_report(lookup_table, bandwidth_table, test_cases):
    """生成可读的统计报告"""
    report = []
    report.append("=" * 80)
    report.append("GNX Stage Profiling Report (精简版)")
    report.append("=" * 80)
    report.append("")

    # 配置信息
    report.append("Test Configuration:")
    report.append(f"  - Test cases: {len(test_cases)}")
    report.append(f"  - Node sizes: {sorted(set(c['nodes'] for c in test_cases))}")
    report.append(f"  - Edge sizes: {sorted(set(c['edges'] for c in test_cases))}")
    report.append(f"  - Total measurements: {len(lookup_table)}")
    report.append("")

    # 每个stage的平均计算时间
    report.append("Average Compute Time by Stage (ms):")
    report.append("-" * 80)
    report.append(f"{'Stage':<10} {'CPU':<15} {'GPU':<15} {'NPU':<15}")
    report.append("-" * 80)

    for stage_id in range(1, 8):
        if stage_id == 2:
                continue
        avgs = {}
        for pu in ['CPU', 'GPU', 'NPU']:
            times = [v['compute_time_ms'] for k, v in lookup_table.items()
                    if k.endswith(f",{pu},{stage_id}")]
            avgs[pu] = np.mean(times) if times else 0

        report.append(f"Stage {stage_id:<3} {avgs['CPU']:<15.2f} {avgs['GPU']:<15.2f} {avgs['NPU']:<15.2f}")

    # 带宽统计
    report.append("")
    report.append("Estimated Bandwidth by Stage (MB/s):")
    report.append("-" * 80)

    for stage_id in range(1, 8):
        if stage_id == 2:
                continue
        bws = {}
        for pu in ['CPU', 'GPU', 'NPU']:
            key = f"{pu}_stage{stage_id}"
            bws[pu] = bandwidth_table.get(key, 0)

        if any(bws.values()):
            report.append(f"Stage {stage_id}: CPU={bws['CPU']:.1f} | GPU={bws['GPU']:.1f} | NPU={bws['NPU']:.1f}")

    # 加速比分析
    report.append("")
    report.append("Speedup Analysis (GPU vs CPU, NPU vs CPU):")
    report.append("-" * 80)

    for stage_id in range(1, 8):
        if stage_id == 2:
                continue
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

    # 保存报告
    report_file = RESULTS_DIR / 'profiling_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print("\n" + '\n'.join(report))
    print(f"\n✓ Report saved: {report_file}")

# ============================================================================
# Main Function
# ============================================================================

def merge_npu_checkpoints():
    """Merge all NPU checkpoint files from individual (nodes, stage) tests"""
    print("\n" + "=" * 70)
    print("=== Merging NPU Checkpoints ===")
    print("=" * 70)

    merged = {}
    checkpoint_files = list(RESULTS_DIR.glob("npu_stage*_n*.json"))

    if not checkpoint_files:
        print("⚠ No NPU checkpoint files found")
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

    print(f"\n✓ Merged {len(merged)} successful NPU measurements from {len(checkpoint_files)} files")

    # Save merged checkpoint
    save_checkpoint(merged, 'npu')
    return merged

def main():
    parser = argparse.ArgumentParser(description='GNX Stage Profiling Script')
    parser.add_argument('--all', action='store_true', help='Export all + measure CPU/GPU (NPU via profile_npu.py)')
    parser.add_argument('--export', action='store_true', help='Export all models (CPU/GPU dynamic + NPU static)')
    parser.add_argument('--export-cpugpu', action='store_true', help='Export CPU/GPU dynamic models only')
    parser.add_argument('--export-npu', action='store_true', help='Export NPU static models only')
    parser.add_argument('--measure', action='store_true', help='Measure CPU/GPU latencies only')
    parser.add_argument('--measure-cpugpu', action='store_true', help='Measure CPU/GPU latencies only (alias)')
    parser.add_argument('--measure-gpu', action='store_true', help='Measure GPU latencies only')
    parser.add_argument('--measure-cpu', action='store_true', help='Measure CPU latencies only')
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
    print("GNX Stage Profiling - Incremental Pipeline")
    print("=" * 70)
    print(f"Test cases: {len(test_cases)}")
    print(f"Node sizes: {sorted(set(c['nodes'] for c in test_cases))}")
    print()
    print("Workflow:")
    print("  1. python profile_stages.py --export          # Export all models")
    print("  2. python profile_stages.py --measure         # Measure CPU/GPU")
    print("  3. run_profiling.bat                          # Run NPU tests (isolated)")
    print("  4. python profile_stages.py --merge-npu       # Merge NPU results")
    print("  5. python profile_stages.py --analyze         # Generate final report")
    print()

    # ========================================================================
    # Handle --export-cpugpu (CPU/GPU dynamic models only)
    # ========================================================================
    if args.export_cpugpu:
        print("\n" + "=" * 70)
        print("Exporting CPU/GPU Dynamic Models Only")
        print("=" * 70)
        export_dynamic_models()
        print("\n✓ CPU/GPU dynamic models exported!")
        return

    # ========================================================================
    # Handle --export-npu (NPU static models only)
    # ========================================================================
    if args.export_npu:
        print("\n" + "=" * 70)
        print("Exporting NPU Static Models Only")
        print("=" * 70)
        export_npu_static_models(test_cases)
        print("\n✓ NPU static models exported!")
        return

    # ========================================================================
    # Handle --merge-npu (Merge NPU checkpoint files)
    # ========================================================================
    if args.merge_npu:
        npu_results = merge_npu_checkpoints()
        if npu_results:
            print(f"\n✓ NPU results merged into checkpoint_npu.json")
            print(f"  Total successful measurements: {len(npu_results)}")
        return

    # ========================================================================
    # Handle --export (Export all models)
    # ========================================================================
    if args.export or args.all:
        print("\n" + "=" * 70)
        print("PHASE 1: Exporting All Models")
        print("=" * 70)

        print("\n[1/2] Exporting CPU/GPU dynamic models...")
        export_dynamic_models()

        print("\n[2/2] Exporting NPU static models...")
        export_npu_static_models(test_cases)

        print("\n✓ All models exported!")

        if args.export and not args.all:
            return

    # ========================================================================
    # Handle --measure (Measure CPU/GPU only - NPU via profile_npu.py)
    # ========================================================================
    if args.measure or args.all:
        print("\n" + "=" * 70)
        print("PHASE 2: Measuring CPU/GPU Latencies")
        print("=" * 70)
        print("Note: NPU measurements should be done via profile_npu.py for isolation")

        cpugpu_results = measure_all_latencies(test_cases, config, pu_list=['CPU', 'GPU'])
        save_checkpoint(cpugpu_results, 'cpugpu')

        print("\n✓ CPU/GPU measurements completed and saved!")
        print(f"  Total: {len(cpugpu_results)} entries")
        print("\nNext steps:")
        print("  1. Run NPU tests: run_profiling.bat (or manually via profile_npu.py)")
        print("  2. Merge NPU results: python profile_stages.py --merge-npu")
        print("  3. Generate report: python profile_stages.py --analyze")

        if args.measure and not args.all:
            return

    # ========================================================================
    # Handle --measure-gpu (Measure GPU only)
    # ========================================================================
    if args.measure_gpu:
        print("\n" + "=" * 70)
        print("Measuring GPU Latencies Only")
        print("=" * 70)

        gpu_results = measure_all_latencies(test_cases, config, pu_list=['GPU'])
        save_checkpoint(gpu_results, 'gpu')

        print("\n✓ GPU measurements completed and saved!")
        print(f"  Total: {len(gpu_results)} entries")
        print("  Checkpoint: results/checkpoint_gpu.json")
        return

    # ========================================================================
    # Handle --measure-cpu (Measure CPU only)
    # ========================================================================
    if args.measure_cpu:
        print("\n" + "=" * 70)
        print("Measuring CPU Latencies Only")
        print("=" * 70)

        cpu_results = measure_all_latencies(test_cases, config, pu_list=['CPU'])
        save_checkpoint(cpu_results, 'cpu')

        print("\n✓ CPU measurements completed and saved!")
        print(f"  Total: {len(cpu_results)} entries")
        print("  Checkpoint: results/checkpoint_cpu.json")
        return

    # ========================================================================
    # Handle --analyze (Generate final results)
    # ========================================================================
    if args.analyze or args.all:
        print("\n" + "=" * 70)
        print("PHASE 3: Analyzing Results")
        print("=" * 70)

        # Load all checkpoints
        cpugpu_data = load_checkpoint('cpugpu')
        npu_data = load_checkpoint('npu')

        if cpugpu_data is None:
            print("ERROR: CPU/GPU checkpoint not found.")
            print("  Run: python profile_stages.py --measure")
            return

        # Merge results
        all_results = cpugpu_data.copy()
        if npu_data is not None:
            all_results.update(npu_data)
            print(f"✓ Merged: {len(cpugpu_data)} CPU/GPU + {len(npu_data)} NPU = {len(all_results)} total")
        else:
            print(f"WARNING: NPU data not found, using CPU/GPU only ({len(all_results)} entries)")
            print("  Run NPU tests: run_profiling.bat")
            print("  Then merge: python profile_stages.py --merge-npu")

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
        print("Profiling Session Complete")
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
