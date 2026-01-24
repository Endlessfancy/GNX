#!/usr/bin/env python3
"""
Test CPU thread configuration impact on OpenVINO inference

Run on your Intel AI PC to test different CPU configurations.

Usage:
    python test_cpu_threads.py
    python test_cpu_threads.py --threads 8
    python test_cpu_threads.py --throughput
"""

import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
GNX_ROOT = SCRIPT_DIR.parent


def get_cpu_info():
    """Get CPU information"""
    import openvino.runtime as ov
    core = ov.Core()

    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"OS CPU count: {os.cpu_count()}")
    print(f"Available devices: {core.available_devices}")

    try:
        print(f"CPU full name: {core.get_property('CPU', 'FULL_DEVICE_NAME')}")
        print(f"Default threads: {core.get_property('CPU', 'INFERENCE_NUM_THREADS')}")
    except Exception as e:
        print(f"Could not get CPU properties: {e}")
    print()


def test_single_config(num_threads=None, throughput_mode=False):
    """Test a single CPU configuration"""
    import openvino.runtime as ov
    from openvino.tools import mo

    # Find model
    onnx_path = GNX_ROOT / "executer/models/c1_b0_CPU_stages_1_2_3_4.onnx"
    if not onnx_path.exists():
        # Try alternative path
        onnx_path = GNX_ROOT / "compiler/output/models/CPU_stages_1_2_3_4_5_6_7.onnx"

    if not onnx_path.exists():
        print(f"No ONNX model found!")
        return None

    print(f"Model: {onnx_path.name}")

    core = ov.Core()

    # Apply configuration
    config_name = "Default"
    if num_threads is not None:
        core.set_property("CPU", {"INFERENCE_NUM_THREADS": num_threads})
        config_name = f"{num_threads} threads"
    if throughput_mode:
        core.set_property("CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
        config_name = "Throughput mode"

    # Get actual configuration
    actual_threads = core.get_property("CPU", "INFERENCE_NUM_THREADS")
    print(f"Config: {config_name} (actual threads: {actual_threads})")

    # Convert model
    print("Loading model...")
    model = mo.convert_model(str(onnx_path))
    compiled_model = core.compile_model(model, "CPU")

    # Generate test input
    torch.manual_seed(42)
    num_nodes, num_edges = 10000, 100000
    x = torch.randn(num_nodes, 500).numpy()
    edge_index = torch.randint(0, num_nodes, (2, num_edges)).numpy()

    # Create infer request
    infer_request = compiled_model.create_infer_request()
    infer_request.set_input_tensor(0, ov.Tensor(x))
    infer_request.set_input_tensor(1, ov.Tensor(edge_index))

    # Warmup
    print("Warmup...")
    for _ in range(10):
        infer_request.start_async()
        infer_request.wait()

    # Measure
    print("Measuring...")
    latencies = []
    for _ in range(30):
        start = time.perf_counter()
        infer_request.start_async()
        infer_request.wait()
        latencies.append((time.perf_counter() - start) * 1000)

    mean_ms = np.mean(latencies)
    std_ms = np.std(latencies)
    min_ms = np.min(latencies)
    max_ms = np.max(latencies)

    print()
    print(f"Results: {mean_ms:.2f} ms (std: {std_ms:.2f}, min: {min_ms:.2f}, max: {max_ms:.2f})")

    return mean_ms


def main():
    parser = argparse.ArgumentParser(description="Test CPU thread configurations")
    parser.add_argument("--threads", type=int, default=None,
                       help="Number of threads (default: auto)")
    parser.add_argument("--throughput", action="store_true",
                       help="Use throughput mode instead of latency mode")
    parser.add_argument("--all", action="store_true",
                       help="Test all common configurations")
    args = parser.parse_args()

    get_cpu_info()

    if args.all:
        # Test multiple configurations
        configs = [
            (None, False, "Default"),
            (1, False, "1 thread"),
            (4, False, "4 threads"),
            (8, False, "8 threads"),
            (16, False, "16 threads"),
            (None, True, "Throughput"),
        ]

        results = []
        for threads, throughput, name in configs:
            print("=" * 60)
            result = test_single_config(threads, throughput)
            if result:
                results.append((name, result))
            print()

        # Summary
        if results:
            print("=" * 60)
            print("Summary")
            print("=" * 60)
            baseline = results[0][1]
            for name, ms in results:
                speedup = baseline / ms
                print(f"  {name:<20} {ms:>8.2f} ms  ({speedup:.2f}x)")
    else:
        test_single_config(args.threads, args.throughput)


if __name__ == "__main__":
    main()
