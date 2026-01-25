#!/usr/bin/env python3
"""
测试 OpenVINO GPU 数据传输时间

验证：
1. set_input_tensor 每次循环设置 vs 只设置一次的差异
2. 输入传输+计算 vs 输出传回的时间分离
3. Stage 6、Fused 1-4、Fused 1-7 的时间对比

Usage:
    python test_transfer_time.py
"""

import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'profiling_fused'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments' / 'baseline_fused1-7_profiling'))

import torch

# Paths
SCRIPT_DIR = Path(__file__).parent
PROFILING_DIR = SCRIPT_DIR
PROFILING_FUSED_DIR = SCRIPT_DIR.parent / "profiling_fused"
BASELINE_FUSED_DIR = SCRIPT_DIR.parent / "experiments" / "baseline_fused1-7_profiling"

# Test configurations
TEST_CASES = [
    {"nodes": 5000, "edges": 50000, "name": "5k nodes, 50k edges"},
    {"nodes": 20000, "edges": 200000, "name": "20k nodes, 200k edges"},
]

FEATURE_DIM = 500
OUT_DIM = 256
NUM_WARMUP = 10
NUM_ITERATIONS = 30


# ============================================================================
# Input Generation
# ============================================================================

def generate_stage3_input(num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """Generate input for Stage 3 (REDUCE_SUM / scatter_add): messages, edge_index, num_nodes

    Note: Stage 3 uses scatter_add which is a GPU bottleneck operation.
    This makes transfer time more visible relative to compute time.
    """
    torch.manual_seed(42)
    messages = torch.randn(num_edges, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return (messages, edge_index, torch.tensor(num_nodes))


def generate_stage6_input(num_nodes, feature_dim=FEATURE_DIM):
    """Generate input for Stage 6 (TRANSFORM): mean_agg, x"""
    torch.manual_seed(42)
    mean_agg = torch.randn(num_nodes, feature_dim)
    x = torch.randn(num_nodes, feature_dim)
    return (mean_agg, x)


def generate_block0_input(num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """Generate input for FusedBlock0 (stages 1-4): x, edge_index"""
    torch.manual_seed(42)
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return (x, edge_index)


def generate_block07_input(num_nodes, num_edges, feature_dim=FEATURE_DIM):
    """Generate input for FusedBlock0_7 (stages 1-7): x, edge_index"""
    torch.manual_seed(42)
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return (x, edge_index)


# ============================================================================
# Model Export (if needed)
# ============================================================================

def export_stage3_model():
    """Export Stage 3 model for GPU (scatter_add - GPU bottleneck)"""
    from models.Model_sage import Stage3_ReduceSum

    MODELS_DIR = PROFILING_DIR / 'exported_models'
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = Stage3_ReduceSum()
    model.eval()

    dummy_input = generate_stage3_input(5000, 50000)
    onnx_path = MODELS_DIR / "stage3_gpu.onnx"

    print(f"Exporting Stage 3 (scatter_add) to {onnx_path}...")
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=['messages', 'edge_index', 'num_nodes'],
            output_names=['output'],
            dynamic_axes={
                'messages': {0: 'num_edges'},
                'edge_index': {1: 'num_edges'},
                'output': {0: 'num_nodes'}
            },
            opset_version=17
        )

    ir_path = MODELS_DIR / "stage3_gpu.xml"
    from openvino.tools import mo
    from openvino import save_model
    ov_model = mo.convert_model(str(onnx_path))
    save_model(ov_model, str(ir_path))
    print(f"  Exported to {ir_path}")
    return ir_path


def export_stage6_model():
    """Export Stage 6 model for GPU"""
    from models.Model_sage import Stage6_Transform

    MODELS_DIR = PROFILING_DIR / 'exported_models'
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = Stage6_Transform(FEATURE_DIM, OUT_DIM)
    model.eval()

    dummy_input = generate_stage6_input(5000)
    onnx_path = MODELS_DIR / "stage6_gpu.onnx"

    print(f"Exporting Stage 6 to {onnx_path}...")
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=['mean_agg', 'x'],
            output_names=['output'],
            dynamic_axes={
                'mean_agg': {0: 'num_nodes'},
                'x': {0: 'num_nodes'},
                'output': {0: 'num_nodes'}
            },
            opset_version=17
        )

    # Convert to IR
    ir_path = MODELS_DIR / "stage6_gpu.xml"
    from openvino.tools import mo
    from openvino import save_model
    ov_model = mo.convert_model(str(onnx_path))
    save_model(ov_model, str(ir_path))
    print(f"  Exported to {ir_path}")
    return ir_path


def export_block0_model():
    """Export FusedBlock0 (stages 1-4) for GPU"""
    from models.Model_sage import FusedBlock0

    MODELS_DIR = PROFILING_FUSED_DIR / 'exported_models'
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = FusedBlock0(FEATURE_DIM)
    model.eval()

    dummy_input = generate_block0_input(5000, 50000)
    onnx_path = MODELS_DIR / "block0_fused_gpu.onnx"

    print(f"Exporting FusedBlock0 to {onnx_path}...")
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=['x', 'edge_index'],
            output_names=['sum_agg', 'count'],
            dynamic_axes={
                'x': {0: 'num_nodes'},
                'edge_index': {1: 'num_edges'},
                'sum_agg': {0: 'num_nodes'},
                'count': {0: 'num_nodes'}
            },
            opset_version=17
        )

    ir_path = MODELS_DIR / "block0_fused_gpu.xml"
    from openvino.tools import mo
    from openvino import save_model
    ov_model = mo.convert_model(str(onnx_path))
    save_model(ov_model, str(ir_path))
    print(f"  Exported to {ir_path}")
    return ir_path


def export_block07_model():
    """Export FusedBlock0_7 (stages 1-7) for GPU"""
    from models import FusedBlock0_7

    MODELS_DIR = BASELINE_FUSED_DIR / 'exported_models'
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = FusedBlock0_7(FEATURE_DIM, OUT_DIM)
    model.eval()

    dummy_input = generate_block07_input(5000, 50000)
    onnx_path = MODELS_DIR / "fused_block0_7_gpu.onnx"

    print(f"Exporting FusedBlock0_7 to {onnx_path}...")
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=['x', 'edge_index'],
            output_names=['output'],
            dynamic_axes={
                'x': {0: 'num_nodes'},
                'edge_index': {1: 'num_edges'},
                'output': {0: 'num_nodes'}
            },
            opset_version=17
        )

    ir_path = MODELS_DIR / "fused_block0_7_gpu.xml"
    from openvino.tools import mo
    from openvino import save_model
    ov_model = mo.convert_model(str(onnx_path))
    save_model(ov_model, str(ir_path))
    print(f"  Exported to {ir_path}")
    return ir_path


# ============================================================================
# Measurement Functions
# ============================================================================

def measure_detailed(ir_path, device, dummy_input, mode='set_every_iter',
                     num_warmup=NUM_WARMUP, num_iterations=NUM_ITERATIONS):
    """
    详细测量传输时间

    Args:
        mode:
            'set_every_iter' = 同一份数据，每次循环都 set_input_tensor
            'set_once'       = 同一份数据，只 set 一次
            'new_data'       = 每次生成新随机数据 + set_tensor (强制传输)

    Returns:
        dict:
        - compute_time: start_async + wait (输入传输 + 计算)
        - output_time: get_output_tensor (输出传回)
        - total_time: 完整端到端
    """
    try:
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(ir_path))
        compiled_model = core.compile_model(model, device)

        # Prepare inputs as numpy
        if isinstance(dummy_input, tuple):
            inputs = [t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
                     for t in dummy_input]
        else:
            inputs = [dummy_input.numpy()]

        # 保存 shape 和 dtype 用于生成新数据
        input_shapes = [inp.shape for inp in inputs]
        input_dtypes = [inp.dtype for inp in inputs]

        infer_request = compiled_model.create_infer_request()

        # Warmup (always set tensor)
        for _ in range(num_warmup):
            for i in range(len(inputs)):
                infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))
            infer_request.start_async()
            infer_request.wait()
            _ = infer_request.get_output_tensor(0).data

        # 如果 mode='set_once'，在测量前只设置一次
        if mode == 'set_once':
            for i in range(len(inputs)):
                infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))

        # Measure
        compute_times = []
        output_times = []
        total_times = []

        for iter_idx in range(num_iterations):
            if mode == 'set_every_iter':
                # 同一份数据，每次都 set
                for i in range(len(inputs)):
                    infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))
            elif mode == 'new_data':
                # 每次生成新随机数据（相同 shape），强制传输
                for i in range(len(inputs)):
                    if len(input_shapes[i]) == 0:
                        # 标量输入（如 num_nodes），保持不变
                        infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))
                    elif input_dtypes[i] == np.int64:
                        # edge_index 等整数类型
                        high = max(input_shapes[i]) if input_shapes[i] else 1
                        new_data = np.random.randint(0, high, input_shapes[i], dtype=input_dtypes[i])
                        infer_request.set_input_tensor(i, ov.Tensor(new_data))
                    else:
                        # float 类型
                        new_data = np.random.randn(*input_shapes[i]).astype(input_dtypes[i])
                        infer_request.set_input_tensor(i, ov.Tensor(new_data))
            # mode == 'set_once': 不做任何事

            t_start = time.perf_counter()
            infer_request.start_async()
            infer_request.wait()
            t_compute = time.perf_counter()

            # 获取输出（触发 Device→CPU 传输）
            output = infer_request.get_output_tensor(0).data
            t_output = time.perf_counter()

            compute_times.append((t_compute - t_start) * 1000)
            output_times.append((t_output - t_compute) * 1000)
            total_times.append((t_output - t_start) * 1000)

        return {
            'compute_ms': np.mean(compute_times),
            'compute_std': np.std(compute_times),
            'output_ms': np.mean(output_times),
            'output_std': np.std(output_times),
            'total_ms': np.mean(total_times),
            'total_std': np.std(total_times),
            'failed': False
        }

    except Exception as e:
        return {'failed': True, 'error': str(e)}


def print_result(name, result_set, result_cache, result_new=None):
    """打印对比结果（三种模式）"""
    if result_set['failed']:
        print(f"  {name}: FAILED - {result_set.get('error', '')[:60]}")
        return

    print(f"\n  {name}:")
    print(f"    [A] 同数据+每次set:  输入+计算: {result_set['compute_ms']:7.2f}ms, "
          f"输出: {result_set['output_ms']:6.3f}ms, "
          f"总计: {result_set['total_ms']:7.2f}ms")
    print(f"    [B] 同数据+只set1次: 输入+计算: {result_cache['compute_ms']:7.2f}ms, "
          f"输出: {result_cache['output_ms']:6.3f}ms, "
          f"总计: {result_cache['total_ms']:7.2f}ms")

    if result_new and not result_new['failed']:
        print(f"    [C] 新数据+每次set:  输入+计算: {result_new['compute_ms']:7.2f}ms, "
              f"输出: {result_new['output_ms']:6.3f}ms, "
              f"总计: {result_new['total_ms']:7.2f}ms")

    # 分析
    diff_ab = result_set['compute_ms'] - result_cache['compute_ms']
    pct_ab = diff_ab / result_set['compute_ms'] * 100 if result_set['compute_ms'] > 0 else 0

    print(f"\n    分析:")
    print(f"    A vs B (同数据): {diff_ab:+.2f}ms ({pct_ab:+.1f}%)")

    if result_new and not result_new['failed']:
        diff_ca = result_new['compute_ms'] - result_set['compute_ms']
        pct_ca = diff_ca / result_set['compute_ms'] * 100 if result_set['compute_ms'] > 0 else 0
        print(f"    C vs A (新数据vs同数据): {diff_ca:+.2f}ms ({pct_ca:+.1f}%)")

        # 结论
        if abs(pct_ab) < 5 and pct_ca > 10:
            print(f"    → 结论: 相同数据被优化跳过传输，新数据触发真实传输")
            print(f"    → 传输时间估算: ~{diff_ca:.2f}ms")
        elif abs(pct_ab) < 5 and abs(pct_ca) < 5:
            print(f"    → 结论: 传输时间很小，或被计算完全掩盖")
        else:
            print(f"    → 结论: 需要进一步分析")


# ============================================================================
# Main Test
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', action='store_true', help='Export models first')
    args = parser.parse_args()

    print("=" * 70)
    print("OpenVINO GPU 数据传输时间测试")
    print("=" * 70)
    print(f"Warmup: {NUM_WARMUP}, Iterations: {NUM_ITERATIONS}")
    print(f"Feature dim: {FEATURE_DIM}, Output dim: {OUT_DIM}")

    # Export models if needed
    if args.export:
        print("\n--- Exporting Models ---")
        try:
            export_stage3_model()
        except Exception as e:
            print(f"  Stage 3 export failed: {e}")
        try:
            export_stage6_model()
        except Exception as e:
            print(f"  Stage 6 export failed: {e}")
        try:
            export_block0_model()
        except Exception as e:
            print(f"  Block 0 export failed: {e}")
        try:
            export_block07_model()
        except Exception as e:
            print(f"  Block 0-7 export failed: {e}")

    # Model paths
    stage3_path = PROFILING_DIR / "exported_models" / "stage3_gpu.xml"
    stage6_path = PROFILING_DIR / "exported_models" / "stage6_gpu.xml"
    block0_path = PROFILING_FUSED_DIR / "exported_models" / "block0_fused_gpu.xml"
    block07_path = BASELINE_FUSED_DIR / "exported_models" / "fused_block0_7_gpu.xml"

    for test_case in TEST_CASES:
        nodes, edges = test_case['nodes'], test_case['edges']

        print(f"\n{'='*70}")
        print(f"测试: {test_case['name']}")
        print(f"{'='*70}")

        # Test Stage 3 (scatter_add - GPU bottleneck)
        if stage3_path.exists():
            dummy_input = generate_stage3_input(nodes, edges)
            result_set = measure_detailed(stage3_path, 'GPU', dummy_input, mode='set_every_iter')
            result_cache = measure_detailed(stage3_path, 'GPU', dummy_input, mode='set_once')
            result_new = measure_detailed(stage3_path, 'GPU', dummy_input, mode='new_data')
            print_result("Stage 3 (REDUCE_SUM/scatter_add) - GPU瓶颈", result_set, result_cache, result_new)
        else:
            print(f"\n  Stage 3: IR not found at {stage3_path}")
            print(f"           Run with --export to create")

        # Test Stage 6 (matmul - GPU擅长)
        if stage6_path.exists():
            dummy_input = generate_stage6_input(nodes)
            result_set = measure_detailed(stage6_path, 'GPU', dummy_input, mode='set_every_iter')
            result_cache = measure_detailed(stage6_path, 'GPU', dummy_input, mode='set_once')
            result_new = measure_detailed(stage6_path, 'GPU', dummy_input, mode='new_data')
            print_result("Stage 6 (TRANSFORM/matmul) - GPU擅长", result_set, result_cache, result_new)
        else:
            print(f"\n  Stage 6: IR not found at {stage6_path}")
            print(f"           Run with --export to create")

        # Test Fused Block 0 (stages 1-4)
        if block0_path.exists():
            dummy_input = generate_block0_input(nodes, edges)
            result_set = measure_detailed(block0_path, 'GPU', dummy_input, mode='set_every_iter')
            result_cache = measure_detailed(block0_path, 'GPU', dummy_input, mode='set_once')
            result_new = measure_detailed(block0_path, 'GPU', dummy_input, mode='new_data')
            print_result("Fused 1-4 (Block0)", result_set, result_cache, result_new)
        else:
            print(f"\n  Fused 1-4: IR not found at {block0_path}")
            print(f"             Run with --export to create")

        # Test Fused Block 0-7 (stages 1-7)
        if block07_path.exists():
            dummy_input = generate_block07_input(nodes, edges)
            result_set = measure_detailed(block07_path, 'GPU', dummy_input, mode='set_every_iter')
            result_cache = measure_detailed(block07_path, 'GPU', dummy_input, mode='set_once')
            result_new = measure_detailed(block07_path, 'GPU', dummy_input, mode='new_data')
            print_result("Fused 1-7 (Block0_7)", result_set, result_cache, result_new)
        else:
            print(f"\n  Fused 1-7: IR not found at {block07_path}")
            print(f"             Run with --export to create")

    # Summary
    print(f"\n{'='*70}")
    print("解释")
    print(f"{'='*70}")
    print("""
测试三种模式:
  [A] 同数据 + 每次 set_tensor  - 相同 numpy 数组，每次创建新 ov.Tensor
  [B] 同数据 + 只 set 一次     - 相同 numpy 数组，warmup 后不再 set
  [C] 新数据 + 每次 set_tensor  - 每次生成新随机数据（相同 shape）

分析逻辑:
  - A ≈ B, C > A  → OpenVINO 检测到相同数据，跳过传输；新数据触发真实传输
  - A ≈ B ≈ C     → 传输时间很小，或被 GPU 计算完全掩盖
  - A > B, C > A  → 每次 set 都有开销，新数据开销更大

时间组成:
  - compute_ms = start_async() + wait() = CPU→GPU传输 + GPU计算
  - output_ms  = get_output_tensor().data = GPU→CPU传输

Pipeline 实际场景:
  - 每个 stage 的输入都是新数据
  - 应该参考 [C] 新数据模式的结果
""")


if __name__ == '__main__':
    main()
