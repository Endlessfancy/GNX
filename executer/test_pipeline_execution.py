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

    # Execute
    print("[4/4] Executing pipeline...")
    try:
        result = executor.execute()
        print("  ✓ Execution complete")
    except Exception as e:
        print(f"  ✗ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return
    print()

    # Print results
    print("="*80)
    print("Execution Results")
    print("="*80)
    print(f"Total time: {result['total_time']:.2f}ms")
    print(f"Output embeddings shape: {result['embeddings'].shape}")
    print()

    print("Per-cluster times:")
    for i, time_ms in enumerate(result['per_cluster_times']):
        print(f"  Cluster {i}: {time_ms:.2f}ms")
    print()

    print("Per-subgraph times (first 5):")
    for i, time_ms in enumerate(result['per_subgraph_times'][:5]):
        print(f"  Subgraph {i}: {time_ms:.2f}ms")
    print()

    print("✓ Pipeline execution test completed successfully!")


if __name__ == "__main__":
    main()
