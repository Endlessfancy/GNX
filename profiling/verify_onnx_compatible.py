"""
验证 ONNX 兼容版本的 GAT 和 GCN stages
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

# 原始版本（用于对比）
from models.Model_gat import (
    GATStage4_AttentionSoftmax as GATStage4_Original,
    GATStage6_ReduceSum as GATStage6_Original,
)
from models.Model_gcn import (
    GCNStage1_ComputeNorm as GCNStage1_Original,
    GCNStage4_ReduceSum as GCNStage4_Original,
)

# ONNX 兼容版本
from models.Model_gat_onnx import (
    GATStage4_AttentionSoftmax_ONNX,
    GATStage6_ReduceSum_ONNX,
    GATLayerONNX as GATLayer_ONNX,
)
from models.Model_gcn_onnx import (
    GCNStage1_ComputeNorm_ONNX,
    GCNStage4_ReduceSum_ONNX,
    GCNLayerONNX as GCNLayer_ONNX,
)

NUM_NODES = 100
NUM_EDGES = 500
FEATURE_DIM = 64


def test_gat_stage4():
    """测试 GAT Stage 4 (ATTENTION_SOFTMAX) ONNX 兼容版本"""
    print("\n" + "=" * 60)
    print("Testing GAT Stage 4: ATTENTION_SOFTMAX (ONNX Compatible)")
    print("=" * 60)

    torch.manual_seed(42)
    e = torch.randn(NUM_EDGES)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))

    # 原始版本
    model_orig = GATStage4_Original(num_nodes_static=NUM_NODES)
    model_orig.eval()
    with torch.no_grad():
        alpha_orig = model_orig(e, edge_index, NUM_NODES)

    # ONNX 兼容版本
    model_onnx = GATStage4_AttentionSoftmax_ONNX(NUM_NODES, NUM_EDGES)
    model_onnx.eval()
    with torch.no_grad():
        alpha_onnx = model_onnx(e, edge_index)

    # 比较结果
    diff = torch.abs(alpha_orig - alpha_onnx).max().item()
    print(f"  Original output shape: {alpha_orig.shape}")
    print(f"  ONNX version output shape: {alpha_onnx.shape}")
    print(f"  Max difference: {diff:.6f}")

    # 验证 softmax 属性：每个目标节点的 alpha 之和应该 = 1
    target_nodes = edge_index[1]
    for node in range(min(5, NUM_NODES)):
        mask = target_nodes == node
        if mask.sum() > 0:
            sum_alpha = alpha_onnx[mask].sum().item()
            print(f"  Node {node}: sum(alpha) = {sum_alpha:.4f} (should be ~1.0)")

    # 尝试导出 ONNX
    print("\n  Testing ONNX export...")
    try:
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model_onnx, (e, edge_index), onnx_path, opset_version=17)
        print(f"  ONNX export: SUCCESS")

        # 运行 ONNX
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        inputs = {sess.get_inputs()[i].name: [e.numpy(), edge_index.numpy()][i]
                  for i in range(len(sess.get_inputs()))}
        alpha_ort = sess.run(None, inputs)[0]
        ort_diff = np.abs(alpha_onnx.numpy() - alpha_ort).max()
        print(f"  ONNX Runtime execution: SUCCESS")
        print(f"  PyTorch vs ONNX Runtime diff: {ort_diff:.6f}")
        os.remove(onnx_path)
        return "PASS" if diff < 1e-4 and ort_diff < 1e-4 else "FAIL"
    except Exception as ex:
        print(f"  ONNX export/run FAILED: {ex}")
        return "FAIL"


def test_gat_stage6():
    """测试 GAT Stage 6 (REDUCE_SUM) ONNX 兼容版本"""
    print("\n" + "=" * 60)
    print("Testing GAT Stage 6: REDUCE_SUM (ONNX Compatible)")
    print("=" * 60)

    torch.manual_seed(42)
    msg = torch.randn(NUM_EDGES, FEATURE_DIM)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))

    # 原始版本
    model_orig = GATStage6_Original(num_nodes_static=NUM_NODES)
    model_orig.eval()
    with torch.no_grad():
        h_orig = model_orig(msg, edge_index, NUM_NODES)

    # ONNX 兼容版本
    model_onnx = GATStage6_ReduceSum_ONNX(NUM_NODES, NUM_EDGES)
    model_onnx.eval()
    with torch.no_grad():
        h_onnx = model_onnx(msg, edge_index)

    # 比较结果
    diff = torch.abs(h_orig - h_onnx).max().item()
    print(f"  Original output shape: {h_orig.shape}")
    print(f"  ONNX version output shape: {h_onnx.shape}")
    print(f"  Max difference: {diff:.6f}")

    # 尝试导出 ONNX
    print("\n  Testing ONNX export...")
    try:
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model_onnx, (msg, edge_index), onnx_path, opset_version=17)
        print(f"  ONNX export: SUCCESS")

        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        inputs = {sess.get_inputs()[i].name: [msg.numpy(), edge_index.numpy()][i]
                  for i in range(len(sess.get_inputs()))}
        h_ort = sess.run(None, inputs)[0]
        ort_diff = np.abs(h_onnx.numpy() - h_ort).max()
        print(f"  ONNX Runtime execution: SUCCESS")
        print(f"  PyTorch vs ONNX Runtime diff: {ort_diff:.6f}")
        os.remove(onnx_path)
        return "PASS" if diff < 1e-4 and ort_diff < 1e-4 else "FAIL"
    except Exception as ex:
        print(f"  ONNX export/run FAILED: {ex}")
        return "FAIL"


def test_gcn_stage1():
    """测试 GCN Stage 1 (COMPUTE_NORM) ONNX 兼容版本"""
    print("\n" + "=" * 60)
    print("Testing GCN Stage 1: COMPUTE_NORM (ONNX Compatible)")
    print("=" * 60)

    torch.manual_seed(42)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))

    # 原始版本
    model_orig = GCNStage1_Original(num_nodes_static=NUM_NODES)
    model_orig.eval()
    with torch.no_grad():
        norm_orig = model_orig(edge_index, NUM_NODES)

    # ONNX 兼容版本
    model_onnx = GCNStage1_ComputeNorm_ONNX(NUM_NODES)
    model_onnx.eval()
    with torch.no_grad():
        norm_onnx = model_onnx(edge_index)

    # 比较结果
    diff = torch.abs(norm_orig - norm_onnx).max().item()
    print(f"  Original output shape: {norm_orig.shape}")
    print(f"  ONNX version output shape: {norm_onnx.shape}")
    print(f"  Max difference: {diff:.6f}")

    # 尝试导出 ONNX
    print("\n  Testing ONNX export...")
    try:
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model_onnx, (edge_index,), onnx_path, opset_version=17)
        print(f"  ONNX export: SUCCESS")

        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        norm_ort = sess.run(None, {'edge_index': edge_index.numpy()})[0]
        ort_diff = np.abs(norm_onnx.numpy() - norm_ort).max()
        print(f"  ONNX Runtime execution: SUCCESS")
        print(f"  PyTorch vs ONNX Runtime diff: {ort_diff:.6f}")
        os.remove(onnx_path)
        return "PASS" if diff < 1e-4 and ort_diff < 1e-4 else "FAIL"
    except Exception as ex:
        print(f"  ONNX export/run FAILED: {ex}")
        return "FAIL"


def test_gcn_stage4():
    """测试 GCN Stage 4 (REDUCE_SUM) ONNX 兼容版本"""
    print("\n" + "=" * 60)
    print("Testing GCN Stage 4: REDUCE_SUM (ONNX Compatible)")
    print("=" * 60)

    torch.manual_seed(42)
    msg = torch.randn(NUM_EDGES, FEATURE_DIM)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))

    # 原始版本
    model_orig = GCNStage4_Original(num_nodes_static=NUM_NODES)
    model_orig.eval()
    with torch.no_grad():
        agg_orig = model_orig(msg, edge_index, NUM_NODES)

    # ONNX 兼容版本
    model_onnx = GCNStage4_ReduceSum_ONNX(NUM_NODES)
    model_onnx.eval()
    with torch.no_grad():
        agg_onnx = model_onnx(msg, edge_index)

    # 比较结果
    diff = torch.abs(agg_orig - agg_onnx).max().item()
    print(f"  Original output shape: {agg_orig.shape}")
    print(f"  ONNX version output shape: {agg_onnx.shape}")
    print(f"  Max difference: {diff:.6f}")

    # 尝试导出 ONNX
    print("\n  Testing ONNX export...")
    try:
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model_onnx, (msg, edge_index), onnx_path, opset_version=17)
        print(f"  ONNX export: SUCCESS")

        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        inputs = {sess.get_inputs()[i].name: [msg.numpy(), edge_index.numpy()][i]
                  for i in range(len(sess.get_inputs()))}
        agg_ort = sess.run(None, inputs)[0]
        ort_diff = np.abs(agg_onnx.numpy() - agg_ort).max()
        print(f"  ONNX Runtime execution: SUCCESS")
        print(f"  PyTorch vs ONNX Runtime diff: {ort_diff:.6f}")
        os.remove(onnx_path)
        return "PASS" if diff < 1e-4 and ort_diff < 1e-4 else "FAIL"
    except Exception as ex:
        print(f"  ONNX export/run FAILED: {ex}")
        return "FAIL"


def test_full_gat_layer():
    """测试完整的 ONNX 兼容 GAT Layer"""
    print("\n" + "=" * 60)
    print("Testing Full GAT Layer (ONNX Compatible)")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(NUM_NODES, FEATURE_DIM)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))

    model = GATLayer_ONNX(FEATURE_DIM, FEATURE_DIM, NUM_NODES, NUM_EDGES)
    model.eval()

    with torch.no_grad():
        out = model(x, edge_index)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")

    # 尝试导出 ONNX
    print("\n  Testing ONNX export...")
    try:
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (x, edge_index), onnx_path, opset_version=17)
        print(f"  ONNX export: SUCCESS")

        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        inputs = {sess.get_inputs()[i].name: [x.numpy(), edge_index.numpy()][i]
                  for i in range(len(sess.get_inputs()))}
        out_ort = sess.run(None, inputs)[0]
        ort_diff = np.abs(out.numpy() - out_ort).max()
        print(f"  ONNX Runtime execution: SUCCESS")
        print(f"  PyTorch vs ONNX Runtime diff: {ort_diff:.6f}")
        os.remove(onnx_path)
        return "PASS" if ort_diff < 1e-3 else "FAIL"
    except Exception as ex:
        print(f"  ONNX export/run FAILED: {ex}")
        return "FAIL"


def test_full_gcn_layer():
    """测试完整的 ONNX 兼容 GCN Layer"""
    print("\n" + "=" * 60)
    print("Testing Full GCN Layer (ONNX Compatible)")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(NUM_NODES, FEATURE_DIM)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))

    model = GCNLayer_ONNX(FEATURE_DIM, FEATURE_DIM, NUM_NODES)
    model.eval()

    with torch.no_grad():
        out = model(x, edge_index)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")

    # 尝试导出 ONNX
    print("\n  Testing ONNX export...")
    try:
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (x, edge_index), onnx_path, opset_version=17)
        print(f"  ONNX export: SUCCESS")

        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        inputs = {sess.get_inputs()[i].name: [x.numpy(), edge_index.numpy()][i]
                  for i in range(len(sess.get_inputs()))}
        out_ort = sess.run(None, inputs)[0]
        ort_diff = np.abs(out.numpy() - out_ort).max()
        print(f"  ONNX Runtime execution: SUCCESS")
        print(f"  PyTorch vs ONNX Runtime diff: {ort_diff:.6f}")
        os.remove(onnx_path)
        return "PASS" if ort_diff < 1e-3 else "FAIL"
    except Exception as ex:
        print(f"  ONNX export/run FAILED: {ex}")
        return "FAIL"


def main():
    print("=" * 60)
    print("ONNX Compatible Stage Verification")
    print("=" * 60)
    print(f"Parameters: nodes={NUM_NODES}, edges={NUM_EDGES}, features={FEATURE_DIM}")

    results = {}

    results['GAT_Stage4_ONNX'] = test_gat_stage4()
    results['GAT_Stage6_ONNX'] = test_gat_stage6()
    results['GCN_Stage1_ONNX'] = test_gcn_stage1()
    results['GCN_Stage4_ONNX'] = test_gcn_stage4()
    results['GAT_Full_Layer'] = test_full_gat_layer()
    results['GCN_Full_Layer'] = test_full_gcn_layer()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "✅ PASS" if result == "PASS" else "❌ FAIL"
        print(f"  {name}: {status}")

    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    print(f"\nOverall: {passed}/{total} passed")


if __name__ == '__main__':
    main()
