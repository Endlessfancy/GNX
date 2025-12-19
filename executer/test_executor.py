"""
Test script for complete pipeline execution
端到端测试：从compilation_result到实际推理
"""

import time
import torch
from pathlib import Path
from executor import PipelineExecutor


def test_flickr_pipeline():
    """
    Complete pipeline test for Flickr dataset
    """
    print("=" * 80)
    print("Pipeline Executor - End-to-End Test")
    print("=" * 80)

    # Path configuration (relative to project root)
    project_root = Path(__file__).parent.parent
    compilation_result_path = project_root / "compiler" / "output" / "compilation_result.json"

    if not compilation_result_path.exists():
        print(f"ERROR: Compilation result not found at {compilation_result_path}")
        print("Please run compiler first: python test_compiler_flickr.py")
        return

    print(f"\n[1/5] Loading compilation result from {compilation_result_path}")

    try:
        # 创建执行器
        executor = PipelineExecutor(
            compilation_result_path=str(compilation_result_path),
            dataset_name='flickr'
        )

        print(f"\n[2/5] Executor initialized successfully")
        print(f"  - Dataset: Flickr")
        print(f"  - Number of subgraphs: {executor.num_subgraphs}")
        print(f"  - Number of clusters: {executor.num_clusters}")
        print(f"  - Compiler estimated makespan: {executor.estimated_makespan:.2f}ms")

        print(f"\n[3/5] Preparing data and models...")
        executor.prepare()

        print(f"\n[4/5] Executing pipeline inference...")
        start_time = time.time()

        result = executor.execute()

        actual_latency = (time.time() - start_time) * 1000  # ms

        print(f"\n[5/5] Execution completed!")

        # 结果分析
        print(f"\n{'=' * 80}")
        print("Results Analysis")
        print(f"{'=' * 80}")

        embeddings = result['embeddings']
        per_subgraph_times = result['per_subgraph_times']

        print(f"\nOutput embeddings:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Dtype: {embeddings.dtype}")
        print(f"  Device: {embeddings.device}")

        print(f"\nPerformance:")
        print(f"  Actual latency: {actual_latency:.2f}ms")
        print(f"  Compiler estimate: {executor.estimated_makespan:.2f}ms")

        error_pct = (actual_latency - executor.estimated_makespan) / executor.estimated_makespan * 100
        print(f"  Estimation error: {error_pct:+.1f}%")

        if abs(error_pct) < 20:
            print(f"  ✓ Estimation is accurate (within 20%)")
        else:
            print(f"  ⚠ Estimation deviates significantly")

        print(f"\nPer-subgraph breakdown:")
        total_sg_time = 0
        for sg_id, sg_time in enumerate(per_subgraph_times):
            print(f"  Subgraph {sg_id}: {sg_time:.2f}ms")
            total_sg_time += sg_time

        print(f"  Sum of subgraph times: {total_sg_time:.2f}ms")

        overhead = actual_latency - total_sg_time
        print(f"  Overhead (data loading, ghost nodes, etc.): {overhead:.2f}ms")

        print(f"\n{'=' * 80}")
        print("Test Passed! ✓")
        print(f"{'=' * 80}\n")

        return result

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    test_flickr_pipeline()
