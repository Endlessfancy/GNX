"""
Verification script for GAT and GCN stage ONNX export and execution
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np

# Import GAT stage models
from models.Model_gat import (
    GATStage1_Linear,
    GATStage2_GatherBoth,
    GATStage3_AttentionScore,
    GATStage4_AttentionSoftmax,
    GATStage5_MessageWeighted,
    GATStage6_ReduceSum,
    GATStage7_Activate
)

# Import GCN stage models
from models.Model_gcn import (
    GCNStage1_ComputeNorm,
    GCNStage2_Gather,
    GCNStage3_Message,
    GCNStage4_ReduceSum,
    GCNStage5_Transform,
    GCNStage6_Activate
)

# Test parameters
NUM_NODES = 100
NUM_EDGES = 500
FEATURE_DIM = 64

def test_gat_stages():
    """Test all GAT stages"""
    print("=" * 60)
    print("Testing GAT Stages (7 stages)")
    print("=" * 60)

    results = {}
    torch.manual_seed(42)

    # Stage 1: LINEAR
    print("\n[GAT Stage 1] LINEAR: x[N,F] -> Wx[N,F']")
    try:
        model = GATStage1_Linear(FEATURE_DIM, FEATURE_DIM)
        model.eval()
        x = torch.randn(NUM_NODES, FEATURE_DIM)

        # Test PyTorch forward
        with torch.no_grad():
            out_pt = model(x)
        print(f"  PyTorch output shape: {out_pt.shape}")

        # Export to ONNX
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (x,), onnx_path, opset_version=17,
                         input_names=['x'], dynamic_axes={'x': {0: 'num_nodes'}})
        print(f"  ONNX export: OK")

        # Run ONNX
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        out_onnx = sess.run(None, {'x': x.numpy()})[0]
        print(f"  ONNX output shape: {out_onnx.shape}")

        # Compare
        diff = np.abs(out_pt.numpy() - out_onnx).max()
        print(f"  Max diff: {diff:.6f}")
        results['GAT_Stage1'] = 'PASS' if diff < 1e-4 else 'FAIL'
        os.remove(onnx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['GAT_Stage1'] = f'FAIL: {e}'

    # Stage 2: GATHER_BOTH
    print("\n[GAT Stage 2] GATHER_BOTH: Wx[N,F'], edge_index -> (Wx_i[E,F'], Wx_j[E,F'])")
    try:
        model = GATStage2_GatherBoth()
        model.eval()
        Wx = torch.randn(NUM_NODES, FEATURE_DIM)
        edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))

        # Test PyTorch forward
        with torch.no_grad():
            Wx_i_pt, Wx_j_pt = model(Wx, edge_index)
        print(f"  PyTorch output shapes: Wx_i={Wx_i_pt.shape}, Wx_j={Wx_j_pt.shape}")

        # Export to ONNX
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (Wx, edge_index), onnx_path, opset_version=17,
                         input_names=['Wx', 'edge_index'],
                         dynamic_axes={'Wx': {0: 'num_nodes'}, 'edge_index': {1: 'num_edges'}})
        print(f"  ONNX export: OK")

        # Run ONNX
        sess = ort.InferenceSession(onnx_path)
        outs = sess.run(None, {'Wx': Wx.numpy(), 'edge_index': edge_index.numpy()})
        print(f"  ONNX output shapes: {[o.shape for o in outs]}")

        diff = max(np.abs(Wx_i_pt.numpy() - outs[0]).max(), np.abs(Wx_j_pt.numpy() - outs[1]).max())
        print(f"  Max diff: {diff:.6f}")
        results['GAT_Stage2'] = 'PASS' if diff < 1e-4 else 'FAIL'
        os.remove(onnx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['GAT_Stage2'] = f'FAIL: {e}'

    # Stage 3: ATTENTION_SCORE
    print("\n[GAT Stage 3] ATTENTION_SCORE: Wx_i[E,F'], Wx_j[E,F'] -> e[E]")
    try:
        model = GATStage3_AttentionScore(FEATURE_DIM)
        model.eval()
        Wx_i = torch.randn(NUM_EDGES, FEATURE_DIM)
        Wx_j = torch.randn(NUM_EDGES, FEATURE_DIM)

        # Test PyTorch forward
        with torch.no_grad():
            e_pt = model(Wx_i, Wx_j)
        print(f"  PyTorch output shape: {e_pt.shape}")

        # Export to ONNX
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (Wx_i, Wx_j), onnx_path, opset_version=17,
                         input_names=['Wx_i', 'Wx_j'],
                         dynamic_axes={'Wx_i': {0: 'num_edges'}, 'Wx_j': {0: 'num_edges'}})
        print(f"  ONNX export: OK")

        # Run ONNX
        sess = ort.InferenceSession(onnx_path)
        e_onnx = sess.run(None, {'Wx_i': Wx_i.numpy(), 'Wx_j': Wx_j.numpy()})[0]
        print(f"  ONNX output shape: {e_onnx.shape}")

        diff = np.abs(e_pt.numpy() - e_onnx).max()
        print(f"  Max diff: {diff:.6f}")
        results['GAT_Stage3'] = 'PASS' if diff < 1e-4 else 'FAIL'
        os.remove(onnx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['GAT_Stage3'] = f'FAIL: {e}'

    # Stage 4: ATTENTION_SOFTMAX (contains scatter - may have issues)
    print("\n[GAT Stage 4] ATTENTION_SOFTMAX: e[E], edge_index, num_nodes -> alpha[E]")
    print("  Note: Contains scatter operations")
    try:
        model = GATStage4_AttentionSoftmax(num_nodes_static=NUM_NODES)
        model.eval()
        e = torch.randn(NUM_EDGES)
        edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))

        # Test PyTorch forward
        with torch.no_grad():
            alpha_pt = model(e, edge_index, NUM_NODES)
        print(f"  PyTorch output shape: {alpha_pt.shape}")

        # Export to ONNX (static shape for scatter)
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (e, edge_index, torch.tensor(NUM_NODES)), onnx_path, opset_version=17)
        print(f"  ONNX export: OK")

        # Run ONNX
        sess = ort.InferenceSession(onnx_path)
        inputs = {sess.get_inputs()[i].name: [e.numpy(), edge_index.numpy(), np.array(NUM_NODES)][i]
                  for i in range(len(sess.get_inputs()))}
        alpha_onnx = sess.run(None, inputs)[0]
        print(f"  ONNX output shape: {alpha_onnx.shape}")

        diff = np.abs(alpha_pt.numpy() - alpha_onnx).max()
        print(f"  Max diff: {diff:.6f}")
        results['GAT_Stage4'] = 'PASS' if diff < 1e-4 else 'FAIL'
        os.remove(onnx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['GAT_Stage4'] = f'FAIL: {e}'

    # Stage 5: MESSAGE_WEIGHTED
    print("\n[GAT Stage 5] MESSAGE_WEIGHTED: Wx_j[E,F'], alpha[E] -> msg[E,F']")
    try:
        model = GATStage5_MessageWeighted()
        model.eval()
        Wx_j = torch.randn(NUM_EDGES, FEATURE_DIM)
        alpha = torch.rand(NUM_EDGES)

        # Test PyTorch forward
        with torch.no_grad():
            msg_pt = model(Wx_j, alpha)
        print(f"  PyTorch output shape: {msg_pt.shape}")

        # Export to ONNX
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (Wx_j, alpha), onnx_path, opset_version=17,
                         input_names=['Wx_j', 'alpha'],
                         dynamic_axes={'Wx_j': {0: 'num_edges'}, 'alpha': {0: 'num_edges'}})
        print(f"  ONNX export: OK")

        # Run ONNX
        sess = ort.InferenceSession(onnx_path)
        msg_onnx = sess.run(None, {'Wx_j': Wx_j.numpy(), 'alpha': alpha.numpy()})[0]
        print(f"  ONNX output shape: {msg_onnx.shape}")

        diff = np.abs(msg_pt.numpy() - msg_onnx).max()
        print(f"  Max diff: {diff:.6f}")
        results['GAT_Stage5'] = 'PASS' if diff < 1e-4 else 'FAIL'
        os.remove(onnx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['GAT_Stage5'] = f'FAIL: {e}'

    # Stage 6: REDUCE_SUM (contains scatter)
    print("\n[GAT Stage 6] REDUCE_SUM: msg[E,F'], edge_index, num_nodes -> h[N,F']")
    print("  Note: Contains scatter operations")
    try:
        model = GATStage6_ReduceSum(num_nodes_static=NUM_NODES)
        model.eval()
        msg = torch.randn(NUM_EDGES, FEATURE_DIM)
        edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))

        # Test PyTorch forward
        with torch.no_grad():
            h_pt = model(msg, edge_index, NUM_NODES)
        print(f"  PyTorch output shape: {h_pt.shape}")

        # Export to ONNX (static shape for scatter)
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (msg, edge_index, torch.tensor(NUM_NODES)), onnx_path, opset_version=17)
        print(f"  ONNX export: OK")

        # Run ONNX
        sess = ort.InferenceSession(onnx_path)
        inputs = {sess.get_inputs()[i].name: [msg.numpy(), edge_index.numpy(), np.array(NUM_NODES)][i]
                  for i in range(len(sess.get_inputs()))}
        h_onnx = sess.run(None, inputs)[0]
        print(f"  ONNX output shape: {h_onnx.shape}")

        diff = np.abs(h_pt.numpy() - h_onnx).max()
        print(f"  Max diff: {diff:.6f}")
        results['GAT_Stage6'] = 'PASS' if diff < 1e-4 else 'FAIL'
        os.remove(onnx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['GAT_Stage6'] = f'FAIL: {e}'

    # Stage 7: ACTIVATE
    print("\n[GAT Stage 7] ACTIVATE: h[N,F'] -> out[N,F']")
    try:
        model = GATStage7_Activate()
        model.eval()
        h = torch.randn(NUM_NODES, FEATURE_DIM)

        # Test PyTorch forward
        with torch.no_grad():
            out_pt = model(h)
        print(f"  PyTorch output shape: {out_pt.shape}")

        # Export to ONNX
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (h,), onnx_path, opset_version=17,
                         input_names=['h'], dynamic_axes={'h': {0: 'num_nodes'}})
        print(f"  ONNX export: OK")

        # Run ONNX
        sess = ort.InferenceSession(onnx_path)
        out_onnx = sess.run(None, {'h': h.numpy()})[0]
        print(f"  ONNX output shape: {out_onnx.shape}")

        diff = np.abs(out_pt.numpy() - out_onnx).max()
        print(f"  Max diff: {diff:.6f}")
        results['GAT_Stage7'] = 'PASS' if diff < 1e-4 else 'FAIL'
        os.remove(onnx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['GAT_Stage7'] = f'FAIL: {e}'

    return results


def test_gcn_stages():
    """Test all GCN stages"""
    print("\n" + "=" * 60)
    print("Testing GCN Stages (6 stages)")
    print("=" * 60)

    results = {}
    torch.manual_seed(42)

    # Stage 1: COMPUTE_NORM (contains scatter)
    print("\n[GCN Stage 1] COMPUTE_NORM: edge_index, num_nodes -> norm[E]")
    print("  Note: Contains scatter operations")
    try:
        model = GCNStage1_ComputeNorm(num_nodes_static=NUM_NODES)
        model.eval()
        edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))

        # Test PyTorch forward
        with torch.no_grad():
            norm_pt = model(edge_index, NUM_NODES)
        print(f"  PyTorch output shape: {norm_pt.shape}")

        # Export to ONNX (static shape for scatter)
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (edge_index, torch.tensor(NUM_NODES)), onnx_path, opset_version=17)
        print(f"  ONNX export: OK")

        # Run ONNX
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        inputs = {sess.get_inputs()[i].name: [edge_index.numpy(), np.array(NUM_NODES)][i]
                  for i in range(len(sess.get_inputs()))}
        norm_onnx = sess.run(None, inputs)[0]
        print(f"  ONNX output shape: {norm_onnx.shape}")

        diff = np.abs(norm_pt.numpy() - norm_onnx).max()
        print(f"  Max diff: {diff:.6f}")
        results['GCN_Stage1'] = 'PASS' if diff < 1e-4 else 'FAIL'
        os.remove(onnx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['GCN_Stage1'] = f'FAIL: {e}'

    # Stage 2: GATHER
    print("\n[GCN Stage 2] GATHER: x[N,F], edge_index -> x_j[E,F]")
    try:
        model = GCNStage2_Gather()
        model.eval()
        x = torch.randn(NUM_NODES, FEATURE_DIM)
        edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))

        # Test PyTorch forward
        with torch.no_grad():
            x_j_pt = model(x, edge_index)
        print(f"  PyTorch output shape: {x_j_pt.shape}")

        # Export to ONNX
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (x, edge_index), onnx_path, opset_version=17,
                         input_names=['x', 'edge_index'],
                         dynamic_axes={'x': {0: 'num_nodes'}, 'edge_index': {1: 'num_edges'}})
        print(f"  ONNX export: OK")

        # Run ONNX
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        x_j_onnx = sess.run(None, {'x': x.numpy(), 'edge_index': edge_index.numpy()})[0]
        print(f"  ONNX output shape: {x_j_onnx.shape}")

        diff = np.abs(x_j_pt.numpy() - x_j_onnx).max()
        print(f"  Max diff: {diff:.6f}")
        results['GCN_Stage2'] = 'PASS' if diff < 1e-4 else 'FAIL'
        os.remove(onnx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['GCN_Stage2'] = f'FAIL: {e}'

    # Stage 3: MESSAGE
    print("\n[GCN Stage 3] MESSAGE: x_j[E,F], norm[E] -> msg[E,F]")
    try:
        model = GCNStage3_Message()
        model.eval()
        x_j = torch.randn(NUM_EDGES, FEATURE_DIM)
        norm = torch.rand(NUM_EDGES) + 0.1

        # Test PyTorch forward
        with torch.no_grad():
            msg_pt = model(x_j, norm)
        print(f"  PyTorch output shape: {msg_pt.shape}")

        # Export to ONNX
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (x_j, norm), onnx_path, opset_version=17,
                         input_names=['x_j', 'norm'],
                         dynamic_axes={'x_j': {0: 'num_edges'}, 'norm': {0: 'num_edges'}})
        print(f"  ONNX export: OK")

        # Run ONNX
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        msg_onnx = sess.run(None, {'x_j': x_j.numpy(), 'norm': norm.numpy()})[0]
        print(f"  ONNX output shape: {msg_onnx.shape}")

        diff = np.abs(msg_pt.numpy() - msg_onnx).max()
        print(f"  Max diff: {diff:.6f}")
        results['GCN_Stage3'] = 'PASS' if diff < 1e-4 else 'FAIL'
        os.remove(onnx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['GCN_Stage3'] = f'FAIL: {e}'

    # Stage 4: REDUCE_SUM (contains scatter)
    print("\n[GCN Stage 4] REDUCE_SUM: msg[E,F], edge_index, num_nodes -> agg[N,F]")
    print("  Note: Contains scatter operations")
    try:
        model = GCNStage4_ReduceSum(num_nodes_static=NUM_NODES)
        model.eval()
        msg = torch.randn(NUM_EDGES, FEATURE_DIM)
        edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))

        # Test PyTorch forward
        with torch.no_grad():
            agg_pt = model(msg, edge_index, NUM_NODES)
        print(f"  PyTorch output shape: {agg_pt.shape}")

        # Export to ONNX (static shape for scatter)
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (msg, edge_index, torch.tensor(NUM_NODES)), onnx_path, opset_version=17)
        print(f"  ONNX export: OK")

        # Run ONNX
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        inputs = {sess.get_inputs()[i].name: [msg.numpy(), edge_index.numpy(), np.array(NUM_NODES)][i]
                  for i in range(len(sess.get_inputs()))}
        agg_onnx = sess.run(None, inputs)[0]
        print(f"  ONNX output shape: {agg_onnx.shape}")

        diff = np.abs(agg_pt.numpy() - agg_onnx).max()
        print(f"  Max diff: {diff:.6f}")
        results['GCN_Stage4'] = 'PASS' if diff < 1e-4 else 'FAIL'
        os.remove(onnx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['GCN_Stage4'] = f'FAIL: {e}'

    # Stage 5: TRANSFORM
    print("\n[GCN Stage 5] TRANSFORM: agg[N,F] -> out[N,F']")
    try:
        model = GCNStage5_Transform(FEATURE_DIM, FEATURE_DIM)
        model.eval()
        agg = torch.randn(NUM_NODES, FEATURE_DIM)

        # Test PyTorch forward
        with torch.no_grad():
            out_pt = model(agg)
        print(f"  PyTorch output shape: {out_pt.shape}")

        # Export to ONNX
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (agg,), onnx_path, opset_version=17,
                         input_names=['agg'], dynamic_axes={'agg': {0: 'num_nodes'}})
        print(f"  ONNX export: OK")

        # Run ONNX
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        out_onnx = sess.run(None, {'agg': agg.numpy()})[0]
        print(f"  ONNX output shape: {out_onnx.shape}")

        diff = np.abs(out_pt.numpy() - out_onnx).max()
        print(f"  Max diff: {diff:.6f}")
        results['GCN_Stage5'] = 'PASS' if diff < 1e-4 else 'FAIL'
        os.remove(onnx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['GCN_Stage5'] = f'FAIL: {e}'

    # Stage 6: ACTIVATE
    print("\n[GCN Stage 6] ACTIVATE: out[N,F'] -> h[N,F']")
    try:
        model = GCNStage6_Activate()
        model.eval()
        out = torch.randn(NUM_NODES, FEATURE_DIM)

        # Test PyTorch forward
        with torch.no_grad():
            h_pt = model(out)
        print(f"  PyTorch output shape: {h_pt.shape}")

        # Export to ONNX
        onnx_path = tempfile.mktemp(suffix='.onnx')
        torch.onnx.export(model, (out,), onnx_path, opset_version=17,
                         input_names=['out'], dynamic_axes={'out': {0: 'num_nodes'}})
        print(f"  ONNX export: OK")

        # Run ONNX
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        h_onnx = sess.run(None, {'out': out.numpy()})[0]
        print(f"  ONNX output shape: {h_onnx.shape}")

        diff = np.abs(h_pt.numpy() - h_onnx).max()
        print(f"  Max diff: {diff:.6f}")
        results['GCN_Stage6'] = 'PASS' if diff < 1e-4 else 'FAIL'
        os.remove(onnx_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['GCN_Stage6'] = f'FAIL: {e}'

    return results


def main():
    print("=" * 60)
    print("Stage ONNX Export and Execution Verification")
    print("=" * 60)
    print(f"Test parameters: nodes={NUM_NODES}, edges={NUM_EDGES}, features={FEATURE_DIM}")

    gat_results = test_gat_stages()
    gcn_results = test_gcn_stages()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nGAT Stages:")
    for stage, result in gat_results.items():
        status = "PASS" if result == "PASS" else "FAIL"
        print(f"  {stage}: {status}")

    print("\nGCN Stages:")
    for stage, result in gcn_results.items():
        status = "PASS" if result == "PASS" else "FAIL"
        print(f"  {stage}: {status}")

    # Overall
    all_results = {**gat_results, **gcn_results}
    passed = sum(1 for r in all_results.values() if r == "PASS")
    total = len(all_results)
    print(f"\nOverall: {passed}/{total} passed")

    if passed == total:
        print("\nAll stages verified successfully!")
    else:
        print("\nSome stages failed - check details above")


if __name__ == '__main__':
    main()
