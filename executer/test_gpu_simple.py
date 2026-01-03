"""
Simple GPU Test Script
单独测试 OpenVINO GPU 推理，排除 executor 复杂逻辑的影响
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Test different data sizes
TEST_SIZES = [
    (100, 500),      # 小数据
    (1000, 5000),    # 中等数据
    (5000, 25000),   # 较大数据
    (10000, 50000),  # 大数据
    (50000, 200000), # 很大数据 (接近 Flickr 子图大小)
]

NUM_FEATURES = 500
OUTPUT_DIM = 256


class SimpleGatherModel(nn.Module):
    """简单的 Gather 模型 (类似 Stage 1-2)"""
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        # Stage 1: Gather
        x_j = x[edge_index[0]]
        # Stage 2: Identity
        return x_j


class SimpleScatterModel(nn.Module):
    """简单的 Scatter 模型 (类似 Stage 3)"""
    def __init__(self, num_nodes_static):
        super().__init__()
        self.num_nodes = num_nodes_static

    def forward(self, messages, edge_index):
        # Stage 3: ReduceSum using index_add_
        out = torch.zeros(self.num_nodes, messages.size(1),
                         dtype=messages.dtype, device=messages.device)
        target_nodes = edge_index[1]
        out.index_add_(0, target_nodes, messages)
        return out


class SimpleLinearModel(nn.Module):
    """简单的 Linear 模型 (类似 Stage 6)"""
    def __init__(self, in_dim=500, out_dim=256):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lin(x)


def export_and_test_model(model, model_name, dummy_inputs, input_names,
                          num_nodes, num_edges, test_gpu=True):
    """导出模型并测试 GPU 推理"""

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"  Data size: {num_nodes} nodes, {num_edges} edges")
    print(f"{'='*60}")

    # 1. Export to ONNX
    onnx_path = f"test_models/{model_name}.onnx"
    os.makedirs("test_models", exist_ok=True)

    model.eval()

    try:
        torch.onnx.export(
            model,
            dummy_inputs,
            onnx_path,
            input_names=input_names,
            output_names=['output'],
            opset_version=18,
            do_constant_folding=True,
            verbose=False
        )
        print(f"  ✓ ONNX exported: {os.path.getsize(onnx_path) / 1024:.1f} KB")
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
        return False

    # 2. Convert to OpenVINO IR and test
    try:
        import openvino as ov
        from openvino.runtime import Core

        core = Core()
        print(f"  Available devices: {core.available_devices}")

        # Convert ONNX to IR
        ov_model = ov.convert_model(onnx_path)
        ir_path = f"test_models/{model_name}.xml"
        ov.save_model(ov_model, ir_path, compress_to_fp16=False)
        print(f"  ✓ IR converted")

        # Read model
        model_ir = core.read_model(ir_path)

        # Test on CPU first
        print(f"\n  Testing CPU inference...")
        compiled_cpu = core.compile_model(model_ir, "CPU")
        infer_cpu = compiled_cpu.create_infer_request()

        # Prepare numpy inputs
        numpy_inputs = {}
        for i, name in enumerate(input_names):
            tensor = dummy_inputs[i] if isinstance(dummy_inputs, tuple) else dummy_inputs
            if isinstance(tensor, torch.Tensor):
                numpy_inputs[name] = tensor.numpy()
            else:
                numpy_inputs[name] = tensor

        # CPU inference
        for name, arr in numpy_inputs.items():
            infer_cpu.set_tensor(name, ov.Tensor(arr))
        infer_cpu.infer()
        cpu_output = infer_cpu.get_output_tensor().data
        print(f"  ✓ CPU inference OK, output shape: {cpu_output.shape}")

        # Test on GPU
        if test_gpu and 'GPU' in core.available_devices:
            print(f"\n  Testing GPU inference...")
            try:
                compiled_gpu = core.compile_model(model_ir, "GPU")
                print(f"  ✓ GPU compilation OK")

                infer_gpu = compiled_gpu.create_infer_request()
                for name, arr in numpy_inputs.items():
                    infer_gpu.set_tensor(name, ov.Tensor(arr))

                infer_gpu.infer()
                gpu_output = infer_gpu.get_output_tensor().data
                print(f"  ✓ GPU inference OK, output shape: {gpu_output.shape}")

                # Compare outputs
                diff = np.abs(cpu_output - gpu_output).max()
                print(f"  ✓ CPU vs GPU max diff: {diff:.6f}")

            except Exception as e:
                print(f"  ✗ GPU failed: {e}")
                return False
        else:
            print(f"  ⚠ GPU not available, skipping GPU test")

        return True

    except Exception as e:
        print(f"  ✗ OpenVINO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gather_model(num_nodes, num_edges):
    """测试 Gather 模型 (Stage 1-2)"""
    model = SimpleGatherModel()

    dummy_x = torch.randn(num_nodes, NUM_FEATURES, dtype=torch.float32)
    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.int64)

    return export_and_test_model(
        model,
        f"gather_{num_nodes}_{num_edges}",
        (dummy_x, dummy_edge_index),
        ['x', 'edge_index'],
        num_nodes, num_edges
    )


def test_scatter_model(num_nodes, num_edges):
    """测试 Scatter 模型 (Stage 3)"""
    model = SimpleScatterModel(num_nodes)

    # messages 来自 Stage 2 输出
    dummy_messages = torch.randn(num_edges, NUM_FEATURES, dtype=torch.float32)
    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.int64)

    return export_and_test_model(
        model,
        f"scatter_{num_nodes}_{num_edges}",
        (dummy_messages, dummy_edge_index),
        ['messages', 'edge_index'],
        num_nodes, num_edges
    )


def test_linear_model(num_nodes):
    """测试 Linear 模型 (Stage 6)"""
    model = SimpleLinearModel(NUM_FEATURES, OUTPUT_DIM)

    dummy_x = torch.randn(num_nodes, NUM_FEATURES, dtype=torch.float32)

    return export_and_test_model(
        model,
        f"linear_{num_nodes}",
        (dummy_x,),
        ['x'],
        num_nodes, 0
    )


def main():
    print("="*70)
    print("Simple GPU Test")
    print("="*70)

    results = []

    # Test 1: Linear model (最简单，应该不会失败)
    print("\n" + "="*70)
    print("TEST 1: Linear Model (Stage 6 only)")
    print("="*70)

    for num_nodes, num_edges in TEST_SIZES:
        success = test_linear_model(num_nodes)
        results.append(("Linear", num_nodes, success))
        if not success:
            print(f"  ⚠ Linear model failed at {num_nodes} nodes")
            break

    # Test 2: Gather model (Stage 1-2)
    print("\n" + "="*70)
    print("TEST 2: Gather Model (Stage 1-2)")
    print("="*70)

    for num_nodes, num_edges in TEST_SIZES:
        success = test_gather_model(num_nodes, num_edges)
        results.append(("Gather", num_nodes, success))
        if not success:
            print(f"  ⚠ Gather model failed at {num_nodes} nodes, {num_edges} edges")
            break

    # Test 3: Scatter model (Stage 3)
    print("\n" + "="*70)
    print("TEST 3: Scatter Model (Stage 3)")
    print("="*70)

    for num_nodes, num_edges in TEST_SIZES:
        success = test_scatter_model(num_nodes, num_edges)
        results.append(("Scatter", num_nodes, success))
        if not success:
            print(f"  ⚠ Scatter model failed at {num_nodes} nodes, {num_edges} edges")
            break

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for model_type, size, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {model_type:10} @ {size:6} nodes: {status}")

    failed = [r for r in results if not r[2]]
    if failed:
        print(f"\n⚠ {len(failed)} test(s) failed")
        print("  First failure:", failed[0])
    else:
        print(f"\n✓ All tests passed!")


if __name__ == "__main__":
    main()
