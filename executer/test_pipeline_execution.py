"""
Pipeline Execution Test Script

Test the executor with custom PEP configurations for pipeline execution.
"""

import sys
from pathlib import Path

# Add executer to path
sys.path.insert(0, str(Path(__file__).parent))

from executor import PipelineExecutor
from test_helper import create_two_pep_test_plan


def main():
    print("="*80)
    print("Pipeline Execution Test")
    print("="*80)
    print()

    # Create custom execution plan with two PEPs
    print("[1/4] Creating custom execution plan...")
    custom_plan = create_two_pep_test_plan()

    print(f"  Created {custom_plan['num_clusters']} clusters:")
    for i, cluster in enumerate(custom_plan['clusters']):
        print(f"    Cluster {i}: {cluster['pep_key']}")
        print(f"      PEP: {cluster['pep']}")
        print(f"      Subgraphs: {cluster['subgraph_ids']}")
    print()

    # Create executor with custom plan
    print("[2/4] Initializing executor...")
    try:
        executor = PipelineExecutor(
            custom_execution_plan=custom_plan,
            dataset_name='flickr'
        )
        print("  ✓ Executor initialized")
    except Exception as e:
        print(f"  ✗ Failed to initialize executor: {e}")
        return
    print()

    # Prepare executor (load data, export models)
    print("[3/4] Preparing executor (loading data and models)...")
    print("  Note: This may take a while on first run (model export)")
    try:
        executor.prepare()
        print("  ✓ Preparation complete")
    except Exception as e:
        print(f"  ✗ Preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    print()

    # Execute (Sequential mode)
    print("[4/5] Executing pipeline (Sequential mode)...")
    try:
        result_seq = executor.execute(use_pipeline_parallelism=False)
        print("  ✓ Sequential execution complete")
    except Exception as e:
        print(f"  ✗ Sequential execution failed: {e}")
        import traceback
        traceback.print_exc()
        return
    print()

    # Execute (Pipeline Parallel mode)
    print("[5/5] Executing pipeline (Pipeline Parallel mode)...")
    try:
        result_par = executor.execute(use_pipeline_parallelism=True)
        print("  ✓ Pipeline parallel execution complete")
    except Exception as e:
        print(f"  ✗ Pipeline parallel execution failed: {e}")
        import traceback
        traceback.print_exc()
        return
    print()

    # Print results
    print("="*80)
    print("Execution Results Comparison")
    print("="*80)
    print()

    print("Sequential Execution:")
    print(f"  Total time: {result_seq['total_time']:.2f}ms")
    print(f"  Output shape: {result_seq['embeddings'].shape}")
    print(f"  Per-cluster times: {[f'{t:.2f}ms' for t in result_seq['per_cluster_times']]}")
    print()

    print("Pipeline Parallel Execution:")
    print(f"  Total time: {result_par['total_time']:.2f}ms")
    print(f"  Output shape: {result_par['embeddings'].shape}")
    print(f"  Per-cluster times: {[f'{t:.2f}ms' for t in result_par['per_cluster_times']]}")
    print()

    print("Performance Comparison:")
    speedup = result_seq['total_time'] / result_par['total_time']
    time_saved = result_seq['total_time'] - result_par['total_time']
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {time_saved:.2f}ms ({time_saved/result_seq['total_time']*100:.1f}%)")
    print()

    # Verify correctness
    import torch
    if torch.allclose(result_seq['embeddings'], result_par['embeddings'], atol=1e-4):
        print("✓ Results verification: PASSED (outputs match)")
    else:
        print("✗ Results verification: FAILED (outputs differ!)")
        max_diff = (result_seq['embeddings'] - result_par['embeddings']).abs().max().item()
        print(f"  Max difference: {max_diff}")
    print()

    print("✓ Pipeline execution test completed successfully!")


if __name__ == "__main__":
    main()
