#!/usr/bin/env python3
"""
Verify exported ONNX/IR models are correct single-layer GNN models.

Usage:
    python verify_models.py
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / 'exported_models'


def verify_onnx_models():
    """Verify ONNX model structure"""
    try:
        import onnx
    except ImportError:
        print("WARNING: onnx not installed, skipping ONNX verification")
        return

    print("=" * 70)
    print("ONNX Model Verification")
    print("=" * 70)

    for model_name in ['graphsage', 'gcn', 'gat']:
        onnx_path = MODELS_DIR / f"{model_name}_dynamic.onnx"

        if not onnx_path.exists():
            print(f"\n{model_name}: NOT FOUND")
            continue

        print(f"\n--- {model_name.upper()} ---")
        print(f"File: {onnx_path.name}")

        try:
            model = onnx.load(str(onnx_path))

            # Check inputs
            print("\nInputs:")
            for inp in model.graph.input:
                shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
                dtype = inp.type.tensor_type.elem_type
                print(f"  {inp.name}: shape={shape}, dtype={dtype}")

            # Check outputs
            print("\nOutputs:")
            for out in model.graph.output:
                shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
                dtype = out.type.tensor_type.elem_type
                print(f"  {out.name}: shape={shape}, dtype={dtype}")

            # Count operations
            op_counts = {}
            for node in model.graph.node:
                op_type = node.op_type
                op_counts[op_type] = op_counts.get(op_type, 0) + 1

            print(f"\nOperations ({len(model.graph.node)} total):")
            for op, count in sorted(op_counts.items()):
                print(f"  {op}: {count}")

            # Verify model
            try:
                onnx.checker.check_model(model)
                print("\nModel validation: PASSED")
            except Exception as e:
                print(f"\nModel validation: FAILED - {e}")

        except Exception as e:
            print(f"Error loading model: {e}")


def verify_openvino_models():
    """Verify OpenVINO IR models"""
    try:
        import openvino.runtime as ov
    except ImportError:
        print("\nWARNING: OpenVINO not installed, skipping IR verification")
        return

    print("\n" + "=" * 70)
    print("OpenVINO IR Model Verification")
    print("=" * 70)

    core = ov.Core()

    for model_name in ['graphsage', 'gcn', 'gat']:
        for device in ['cpu', 'gpu']:
            ir_path = MODELS_DIR / f"{model_name}_{device}.xml"

            if not ir_path.exists():
                continue

            print(f"\n--- {model_name.upper()} ({device.upper()}) ---")
            print(f"File: {ir_path.name}")

            try:
                model = core.read_model(str(ir_path))

                # Check inputs
                print("\nInputs:")
                for inp in model.inputs:
                    print(f"  {inp.any_name}: shape={inp.partial_shape}, dtype={inp.element_type}")

                # Check outputs
                print("\nOutputs:")
                for out in model.outputs:
                    print(f"  {out.any_name}: shape={out.partial_shape}, dtype={out.element_type}")

                # Count operations
                op_counts = {}
                for op in model.get_ops():
                    op_type = op.get_type_name()
                    op_counts[op_type] = op_counts.get(op_type, 0) + 1

                print(f"\nOperations ({len(model.get_ops())} total):")
                for op, count in sorted(op_counts.items())[:15]:  # Show top 15
                    print(f"  {op}: {count}")
                if len(op_counts) > 15:
                    print(f"  ... and {len(op_counts) - 15} more op types")

            except Exception as e:
                print(f"Error loading model: {e}")


def test_inference():
    """Test actual inference with the models"""
    try:
        import openvino.runtime as ov
        import numpy as np
    except ImportError:
        print("\nWARNING: Required packages not installed")
        return

    print("\n" + "=" * 70)
    print("Inference Test")
    print("=" * 70)

    core = ov.Core()

    # Test parameters
    num_nodes = 1000
    num_edges = 5000
    feature_dim = 500

    # Generate test input
    np.random.seed(42)
    x = np.random.randn(num_nodes, feature_dim).astype(np.float32)
    edge_index = np.random.randint(0, num_nodes, (2, num_edges)).astype(np.int64)

    print(f"\nTest input: x={x.shape}, edge_index={edge_index.shape}")

    for model_name in ['graphsage', 'gcn', 'gat']:
        ir_path = MODELS_DIR / f"{model_name}_gpu.xml"

        if not ir_path.exists():
            ir_path = MODELS_DIR / f"{model_name}_cpu.xml"

        if not ir_path.exists():
            print(f"\n{model_name}: Model not found")
            continue

        print(f"\n--- {model_name.upper()} ---")

        try:
            model = core.read_model(str(ir_path))
            compiled = core.compile_model(model, 'CPU')  # Use CPU for testing

            result = compiled([x, edge_index])
            output = result[0]

            print(f"Output shape: {output.shape}")
            print(f"Output dtype: {output.dtype}")
            print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
            print(f"Output mean: {output.mean():.4f}")

            # Verify output shape
            expected_shape = (num_nodes, 256)
            if output.shape == expected_shape:
                print(f"Shape verification: PASSED (expected {expected_shape})")
            else:
                print(f"Shape verification: FAILED (expected {expected_shape}, got {output.shape})")

            # Verify ReLU (no negative values)
            if output.min() >= 0:
                print("ReLU verification: PASSED (no negative values)")
            else:
                print(f"ReLU verification: WARNING (min={output.min():.4f})")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    verify_onnx_models()
    verify_openvino_models()
    test_inference()

    print("\n" + "=" * 70)
    print("Verification Complete")
    print("=" * 70)
