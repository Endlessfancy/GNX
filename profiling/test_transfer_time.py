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

def measure_detailed(ir_path, device, dummy_input, set_every_iter=True,
                     num_warmup=NUM_WARMUP, num_iterations=NUM_ITERATIONS):
    """
    详细测量传输时间

    Args:
        set_every_iter: True = 每次循环都 set_input_tensor (包含传输)
                        False = 只 set 一次 (可能用 GPU cache)

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

        infer_request = compiled_model.create_infer_request()

        # Warmup (always set tensor)
        for _ in range(num_warmup):
            for i in range(len(inputs)):
                infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))
            infer_request.start_async()
            infer_request.wait()
            _ = infer_request.get_output_tensor(0).data

        # 如果 set_every_iter=False，在测量前只设置一次
        if not set_every_iter:
            for i in range(len(inputs)):
                infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))

        # Measure
        compute_times = []
        output_times = []
        total_times = []

        for _ in range(num_iterations):
            # 根据参数决定是否每次都 set tensor
            if set_every_iter:
                for i in range(len(inputs)):
                    infer_request.set_input_tensor(i, ov.Tensor(inputs[i]))

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


def print_result(name, result_set, result_cache):
    """打印对比结果"""
    if result_set['failed']:
        print(f"  {name}: FAILED - {result_set.get('error', '')[:60]}")
        return

    print(f"\n  {name}:")
    print(f"    [每次set tensor] 输入+计算: {result_set['compute_ms']:7.2f}ms, "
          f"输出传回: {result_set['output_ms']:6.3f}ms, "
          f"总计: {result_set['total_ms']:7.2f}ms")
    print(f"    [只set一次]     输入+计算: {result_cache['compute_ms']:7.2f}ms, "
          f"输出传回: {result_cache['output_ms']:6.3f}ms, "
          f"总计: {result_cache['total_ms']:7.2f}ms")

    diff = result_set['compute_ms'] - result_cache['compute_ms']
    pct = diff / result_set['compute_ms'] * 100 if result_set['compute_ms'] > 0 else 0
    print(f"    → compute 差异: {diff:+.2f}ms ({pct:+.1f}%)")

    if diff > 1.0:
        print(f"    → 结论: 每次 set 确实触发了 CPU→GPU 数据传输")
    else:
        print(f"    → 结论: 差异小，可能 GPU 有缓存机制")


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
    stage6_path = PROFILING_DIR / "exported_models" / "stage6_gpu.xml"
    block0_path = PROFILING_FUSED_DIR / "exported_models" / "block0_fused_gpu.xml"
    block07_path = BASELINE_FUSED_DIR / "exported_models" / "fused_block0_7_gpu.xml"

    for test_case in TEST_CASES:
        nodes, edges = test_case['nodes'], test_case['edges']

        print(f"\n{'='*70}")
        print(f"测试: {test_case['name']}")
        print(f"{'='*70}")

        # Test Stage 6
        if stage6_path.exists():
            dummy_input = generate_stage6_input(nodes)
            result_set = measure_detailed(stage6_path, 'GPU', dummy_input, set_every_iter=True)
            result_cache = measure_detailed(stage6_path, 'GPU', dummy_input, set_every_iter=False)
            print_result("Stage 6 (TRANSFORM)", result_set, result_cache)
        else:
            print(f"\n  Stage 6: IR not found at {stage6_path}")
            print(f"           Run with --export to create")

        # Test Fused Block 0 (stages 1-4)
        if block0_path.exists():
            dummy_input = generate_block0_input(nodes, edges)
            result_set = measure_detailed(block0_path, 'GPU', dummy_input, set_every_iter=True)
            result_cache = measure_detailed(block0_path, 'GPU', dummy_input, set_every_iter=False)
            print_result("Fused 1-4 (Block0)", result_set, result_cache)
        else:
            print(f"\n  Fused 1-4: IR not found at {block0_path}")
            print(f"             Run with --export to create")

        # Test Fused Block 0-7 (stages 1-7)
        if block07_path.exists():
            dummy_input = generate_block07_input(nodes, edges)
            result_set = measure_detailed(block07_path, 'GPU', dummy_input, set_every_iter=True)
            result_cache = measure_detailed(block07_path, 'GPU', dummy_input, set_every_iter=False)
            print_result("Fused 1-7 (Block0_7)", result_set, result_cache)
        else:
            print(f"\n  Fused 1-7: IR not found at {block07_path}")
            print(f"             Run with --export to create")

    # Summary
    print(f"\n{'='*70}")
    print("解释")
    print(f"{'='*70}")
    print("""
1. compute_ms = start_async() + wait()
   - 包含: CPU→GPU 输入传输 + GPU 计算
   - 不包含: GPU→CPU 输出传输

2. output_ms = get_output_tensor().data
   - GPU→CPU 输出传输时间

3. 每次set vs 只set一次:
   - 差异大 → 每次 set 确实触发了数据传输
   - 差异小 → GPU 可能缓存了输入数据

4. Pipeline 实际场景:
   - 每个 stage 的输入都是新数据
   - 应该用 "每次set" 的结果作为参考
""")


if __name__ == '__main__':
    main()
