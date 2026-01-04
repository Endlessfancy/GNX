"""
Simple GPU Test for Individual Stages

Tests if GPU can run each stage (1-4) individually, then fused stages 1-4.
Based on profiling/profile_stages.py approach.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add paths
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_parent_dir / 'profiling') not in sys.path:
    sys.path.insert(0, str(_parent_dir / 'profiling'))

try:
    import openvino as ov
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("OpenVINO not available")
    sys.exit(1)

# Import stage models from profiling
from profiling.models.Model_sage import (
    SAGEStage1_Gather,
    SAGEStage2_Message,
    SAGEStage3_ReduceSum,
    SAGEStage4_ReduceCount,
)

# ============================================================================
# Helper Functions (from profiling/profile_stages.py)
# ============================================================================

def get_stage_model(stage_id, num_nodes=None):
    """Get stage model."""
    models = {
        1: SAGEStage1_Gather(),
        2: SAGEStage2_Message(),
        3: SAGEStage3_ReduceSum(num_nodes_static=num_nodes),
        4: SAGEStage4_ReduceCount(),
    }
    return models[stage_id]


def generate_dummy_input(stage_id, num_nodes, num_edges, feature_dim=500):
    """Generate dummy input for each stage."""
    torch.manual_seed(42)

    if stage_id == 1:
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        return (x, edge_index)

    elif stage_id == 2:
        return torch.randn(num_edges, feature_dim)

    elif stage_id == 3:
        messages = torch.randn(num_edges, feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        return (messages, edge_index, torch.tensor(num_nodes))

    elif stage_id == 4:
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        return (edge_index, torch.tensor(num_nodes), torch.tensor(num_edges))

    else:
        raise ValueError(f"Invalid stage_id: {stage_id}")


def get_dynamic_axes_for_stage(stage_id):
    """Get dynamic_axes and input_names for each stage."""
    if stage_id == 1:
        dynamic_axes = {
            'x': {0: 'num_nodes'},
            'edge_index': {1: 'num_edges'}
        }
        input_names = ['x', 'edge_index']

    elif stage_id == 2:
        dynamic_axes = {
            'x_j': {0: 'num_edges'}
        }
        input_names = ['x_j']

    elif stage_id == 3:
        dynamic_axes = {
            'messages': {0: 'num_edges'},
            'edge_index': {1: 'num_edges'}
        }
        input_names = ['messages', 'edge_index', 'num_nodes']

    elif stage_id == 4:
        dynamic_axes = {
            'edge_index': {1: 'num_edges'}
        }
        input_names = ['edge_index', 'num_nodes', 'num_edges']

    else:
        raise ValueError(f"Invalid stage_id: {stage_id}")

    return dynamic_axes, input_names


# ============================================================================
# Export Functions
# ============================================================================

def export_stage_onnx(stage_id, output_dir, num_nodes=5000, num_edges=5000):
    """Export a single stage to ONNX."""
    model = get_stage_model(stage_id)
    model.eval()

    dummy_input = generate_dummy_input(stage_id, num_nodes, num_edges)
    dynamic_axes, input_names = get_dynamic_axes_for_stage(stage_id)

    onnx_path = output_dir / f"stage{stage_id}_dynamic.onnx"

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

    return onnx_path


def convert_to_ir(onnx_path, ir_path):
    """Convert ONNX to OpenVINO IR."""
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(ir_path), compress_to_fp16=False)
    return ir_path


# ============================================================================
# Test Functions
# ============================================================================

def test_stage_on_device(stage_id, ir_path, device, num_nodes, num_edges):
    """Test a stage on a specific device."""
    core = Core()

    # Compile model
    model = core.read_model(str(ir_path))
    compiled = core.compile_model(model, device)

    # Generate input
    dummy_input = generate_dummy_input(stage_id, num_nodes, num_edges)

    # Convert to numpy
    if isinstance(dummy_input, tuple):
        inputs = [t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
                  for t in dummy_input]
    else:
        inputs = [dummy_input.numpy() if isinstance(dummy_input, torch.Tensor)
                  else np.array(dummy_input)]

    # Run inference
    request = compiled.create_infer_request()

    # Set inputs by name
    input_names = get_dynamic_axes_for_stage(stage_id)[1]
    for name, data in zip(input_names, inputs):
        request.set_tensor(name, ov.Tensor(data))

    request.infer()

    # Get output
    output = request.get_output_tensor(0).data.copy()
    return output


def test_single_stage(stage_id, output_dir, test_sizes):
    """Test a single stage on CPU and GPU."""
    print(f"\n{'='*60}")
    print(f"Testing Stage {stage_id}")
    print(f"{'='*60}")

    # Export ONNX
    print(f"  Exporting ONNX...")
    try:
        onnx_path = export_stage_onnx(stage_id, output_dir)
        print(f"    ONNX: {onnx_path.name}")
    except Exception as e:
        print(f"    ONNX export FAILED: {e}")
        return False

    # Convert to IR
    print(f"  Converting to IR...")
    ir_path = output_dir / f"stage{stage_id}_dynamic.xml"
    try:
        convert_to_ir(onnx_path, ir_path)
        print(f"    IR: {ir_path.name}")
    except Exception as e:
        print(f"    IR conversion FAILED: {e}")
        return False

    # Test on different sizes
    all_passed = True
    for num_nodes, num_edges in test_sizes:
        print(f"\n  Size: {num_nodes} nodes, {num_edges} edges")

        # Test CPU
        try:
            cpu_output = test_stage_on_device(stage_id, ir_path, "CPU", num_nodes, num_edges)
            print(f"    CPU: OK, shape={cpu_output.shape}")
        except Exception as e:
            print(f"    CPU: FAILED - {e}")
            all_passed = False
            continue

        # Test GPU
        try:
            gpu_output = test_stage_on_device(stage_id, ir_path, "GPU", num_nodes, num_edges)
            print(f"    GPU: OK, shape={gpu_output.shape}")
        except Exception as e:
            print(f"    GPU: FAILED - {e}")
            all_passed = False
            continue

        # Compare
        max_diff = np.max(np.abs(cpu_output - gpu_output))
        if max_diff < 1e-3:
            print(f"    Match: max_diff={max_diff:.6f}")
        else:
            print(f"    MISMATCH: max_diff={max_diff:.6f}")

    return all_passed


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("GPU Test: Individual Stages 1-4")
    print("=" * 60)

    # Check devices
    core = Core()
    print(f"Available devices: {core.available_devices}")

    if "GPU" not in core.available_devices:
        print("ERROR: GPU not available")
        return

    # Output directory
    output_dir = _current_dir / "models" / "gpu_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Test sizes
    test_sizes = [
        (1000, 5000),
        (5000, 25000),
        (10000, 50000),
    ]

    # Test each stage individually
    results = {}
    for stage_id in [1, 2, 3, 4]:
        passed = test_single_stage(stage_id, output_dir, test_sizes)
        results[stage_id] = passed

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for stage_id, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  Stage {stage_id}: {status}")

    # Overall result
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
