"""
Simple GPU Test for Stages 1-5 Fused Model

Tests if GPU can run dynamic IR for stages 1-5 without errors.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent dir to path
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

try:
    import openvino as ov
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("OpenVINO not available")
    sys.exit(1)


def test_gpu_stages_1_5():
    """Test GPU with stages 1-5 dynamic model."""

    print("=" * 60)
    print("GPU Test: Stages 1-5 Fused Model (Dynamic IR)")
    print("=" * 60)

    # Initialize OpenVINO
    core = Core()
    print(f"Available devices: {core.available_devices}")

    if "GPU" not in core.available_devices:
        print("ERROR: GPU not available")
        return False

    # Check for existing model
    models_dir = _current_dir / "models"
    model_path = models_dir / "stages_1_2_3_4_5_GPU_dynamic.xml"

    if not model_path.exists():
        print(f"\nModel not found: {model_path}")
        print("Exporting model first...")

        try:
            from latency.model_exporter import GNNModelExporter
            exporter = GNNModelExporter(output_dir=models_dir)
            # Export stages 1-5 for GPU (dynamic shape)
            exporter.export_stages([1, 2, 3, 4, 5], device="GPU")
            print(f"Model exported to: {model_path}")
        except Exception as e:
            print(f"ERROR exporting model: {e}")
            import traceback
            traceback.print_exc()
            return False

    if not model_path.exists():
        print(f"ERROR: Model still not found after export: {model_path}")
        return False

    print(f"\nLoading model: {model_path}")

    # Load and compile model
    try:
        model = core.read_model(str(model_path))
        print(f"  Model inputs: {[inp.get_any_name() for inp in model.inputs]}")
        print(f"  Model outputs: {[out.get_any_name() for out in model.outputs]}")

        compiled = core.compile_model(model, "GPU")
        print("  Compiled successfully on GPU")
    except Exception as e:
        print(f"ERROR compiling model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create test data
    test_cases = [
        (1000, 10000),   # Small
        (5000, 50000),   # Medium
        (10000, 100000), # Large
    ]

    print("\nRunning inference tests...")

    for num_nodes, num_edges in test_cases:
        print(f"\n  Test: {num_nodes} nodes, {num_edges} edges")

        try:
            # Create random test data
            x = np.random.randn(num_nodes, 500).astype(np.float32)
            edge_index = np.random.randint(0, num_nodes, (2, num_edges)).astype(np.int64)

            # Create infer request
            request = compiled.create_infer_request()

            # Set inputs
            request.set_tensor("x", ov.Tensor(x))
            request.set_tensor("edge_index", ov.Tensor(edge_index))

            # Run inference
            request.infer()

            # Get output
            output = request.get_output_tensor(0).data
            print(f"    SUCCESS: output shape = {output.shape}")

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 60)
    print("All GPU tests PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_gpu_stages_1_5()
    sys.exit(0 if success else 1)
