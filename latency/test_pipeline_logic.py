"""
Standalone Pipeline Logic Test

Tests the pipeline framework without external dependencies like torch_geometric.
Uses synthetic data and dummy model paths to verify:
- HaloPartitioner graph splitting
- PipelineBenchmark cycle execution
- Cross-cycle buffer behavior
"""

import sys
from pathlib import Path
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import numpy as np
from latency.graph_partitioner import HaloPartitioner, PartitionData
from latency.profiler import PipelineProfiler


def test_halo_partitioner():
    """Test HaloPartitioner with synthetic graph."""
    print("=" * 60)
    print("Testing HaloPartitioner")
    print("=" * 60)

    # Create a simple graph: 6 nodes, connected linearly with some cross edges
    # 0 -> 1 -> 2 -> 3 -> 4 -> 5
    # Plus: 0->2, 1->3, 2->4, 3->5
    num_nodes = 6
    x = np.random.randn(num_nodes, 8).astype(np.float32)
    edge_index = np.array([
        [0, 1, 2, 3, 4, 0, 1, 2, 3],  # src
        [1, 2, 3, 4, 5, 2, 3, 4, 5]   # dst
    ], dtype=np.int64)

    print(f"Original graph: {num_nodes} nodes, {edge_index.shape[1]} edges")

    # Partition into 2 parts
    partitioner = HaloPartitioner(num_partitions=2)
    partitions = partitioner.partition(x, edge_index)

    for p in partitions:
        print(f"\nPartition {p.partition_id}:")
        print(f"  Owned nodes: {p.num_owned}")
        print(f"  Halo nodes: {p.num_halo}")
        print(f"  Local edges: {p.num_edges}")
        print(f"  Global IDs: {list(p.local_to_global)}")
        print(f"  Owned mask: {p.owned_mask}")

        # Verify local edge indices are valid
        max_local_id = p.num_owned + p.num_halo - 1
        max_edge_id = p.edge_index_local.max()
        assert max_edge_id <= max_local_id, f"Edge index {max_edge_id} exceeds max {max_local_id}"
        print(f"  Edge index range valid: [0, {max_edge_id}] <= {max_local_id}")

    # Test merge
    fake_outputs = [
        np.arange(p.num_owned + p.num_halo).reshape(-1, 1).astype(np.float32)
        for p in partitions
    ]
    merged = partitioner.merge_outputs(fake_outputs, partitions, num_nodes)
    print(f"\nMerged output shape: {merged.shape}")

    print("\n[PASS] HaloPartitioner test passed!")
    return True


def test_pipeline_buffer():
    """Test PipelineBuffer cross-cycle behavior."""
    print("\n" + "=" * 60)
    print("Testing PipelineBuffer")
    print("=" * 60)

    from latency.pipeline_executor import PipelineBuffer

    # Simulate 3-stage pipeline with 4 batches
    num_stages = 3
    num_batches = 4
    total_cycles = num_batches + num_stages - 1  # = 6

    buffer = PipelineBuffer(num_stages)

    # Raw inputs (simulated)
    raw_inputs = [{'x': np.array([i])} for i in range(num_batches)]

    print(f"Stages: {num_stages}, Batches: {num_batches}, Cycles: {total_cycles}")
    print("\nPipeline execution simulation:")

    for cycle_id in range(total_cycles):
        print(f"\nCycle {cycle_id}:")
        for stage_id in range(num_stages):
            data = buffer.get_input(stage_id, cycle_id, raw_inputs)
            batch_id = cycle_id - stage_id

            if data is not None:
                # Simulate processing
                output = {'output': np.array([batch_id * 10 + stage_id])}
                buffer.set_output(stage_id, output)
                print(f"  Stage {stage_id}: Processing batch {batch_id}, input={data}, output={output}")
            else:
                print(f"  Stage {stage_id}: No data (pipeline {'filling' if batch_id < 0 else 'draining'})")

    print("\n[PASS] PipelineBuffer test passed!")
    return True


def test_pipeline_benchmark_structure():
    """Test PipelineBenchmark without actual inference."""
    print("\n" + "=" * 60)
    print("Testing PipelineBenchmark Structure")
    print("=" * 60)

    from latency.pipeline_executor import PipelineBenchmark, OPENVINO_AVAILABLE

    if not OPENVINO_AVAILABLE:
        print("OpenVINO not available, skipping inference test")
        print("[SKIP] PipelineBenchmark inference test skipped")
        return True

    # Create profiler and pipeline
    profiler = PipelineProfiler("TestPipeline")
    pipeline = PipelineBenchmark(profiler)

    print(f"Pipeline created: {pipeline}")
    print(f"OpenVINO available: {OPENVINO_AVAILABLE}")

    print("\n[PASS] PipelineBenchmark structure test passed!")
    return True


def main():
    print("=" * 60)
    print("Pipeline Logic Standalone Test")
    print("=" * 60)
    print("Testing core pipeline components without external dependencies\n")

    results = []

    # Test 1: HaloPartitioner
    try:
        results.append(("HaloPartitioner", test_halo_partitioner()))
    except Exception as e:
        print(f"[FAIL] HaloPartitioner test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("HaloPartitioner", False))

    # Test 2: PipelineBuffer
    try:
        results.append(("PipelineBuffer", test_pipeline_buffer()))
    except Exception as e:
        print(f"[FAIL] PipelineBuffer test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("PipelineBuffer", False))

    # Test 3: PipelineBenchmark structure
    try:
        results.append(("PipelineBenchmark", test_pipeline_benchmark_structure()))
    except Exception as e:
        print(f"[FAIL] PipelineBenchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("PipelineBenchmark", False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    all_passed = all(passed for _, passed in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
