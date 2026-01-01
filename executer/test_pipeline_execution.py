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

    # Execute (Pipeline Parallel mode - Sync)
    print("[5/6] Executing pipeline (Pipeline Parallel mode - Sync)...")
    try:
        result_par = executor.execute(use_pipeline_parallelism=True)
        print("  ✓ Pipeline parallel (sync) execution complete")
    except Exception as e:
        print(f"  ✗ Pipeline parallel (sync) execution failed: {e}")
        import traceback
        traceback.print_exc()
        return
    print()

    # Execute (Async Pipeline Parallel mode)
    print("[6/6] Executing pipeline (Async Pipeline Parallel mode)...")
    print("  【阶段3优化】Testing asyncio-based pipeline execution...")
    try:
        result_async = executor.execute(use_async=True)
        print("  ✓ Async pipeline parallel execution complete")
    except Exception as e:
        print(f"  ✗ Async pipeline parallel execution failed: {e}")
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

    print("Pipeline Parallel Execution (Sync):")
    print(f"  Total time: {result_par['total_time']:.2f}ms")
    print(f"  Output shape: {result_par['embeddings'].shape}")
    print(f"  Per-cluster times: {[f'{t:.2f}ms' for t in result_par['per_cluster_times']]}")
    print()

    print("Async Pipeline Parallel Execution:")
    print(f"  Total time: {result_async['total_time']:.2f}ms")
    print(f"  Output shape: {result_async['embeddings'].shape}")
    print(f"  Per-cluster times: {[f'{t:.2f}ms' for t in result_async['per_cluster_times']]}")
    print()

    print("Performance Comparison:")
    speedup_sync = result_seq['total_time'] / result_par['total_time']
    speedup_async = result_seq['total_time'] / result_async['total_time']
    time_saved_sync = result_seq['total_time'] - result_par['total_time']
    time_saved_async = result_seq['total_time'] - result_async['total_time']

    print(f"  Sequential → Sync Pipeline:")
    print(f"    Speedup: {speedup_sync:.2f}x")
    print(f"    Time saved: {time_saved_sync:.2f}ms ({time_saved_sync/result_seq['total_time']*100:.1f}%)")

    print(f"  Sequential → Async Pipeline:")
    print(f"    Speedup: {speedup_async:.2f}x")
    print(f"    Time saved: {time_saved_async:.2f}ms ({time_saved_async/result_seq['total_time']*100:.1f}%)")

    print(f"  Sync Pipeline → Async Pipeline:")
    async_vs_sync_speedup = result_par['total_time'] / result_async['total_time']
    async_vs_sync_saved = result_par['total_time'] - result_async['total_time']
    print(f"    Speedup: {async_vs_sync_speedup:.2f}x")
    print(f"    Time saved: {async_vs_sync_saved:.2f}ms ({async_vs_sync_saved/result_par['total_time']*100:.1f}%)")
    print()

    # Verify correctness
    import torch
    seq_vs_sync = torch.allclose(result_seq['embeddings'], result_par['embeddings'], atol=1e-4)
    seq_vs_async = torch.allclose(result_seq['embeddings'], result_async['embeddings'], atol=1e-4)
    sync_vs_async = torch.allclose(result_par['embeddings'], result_async['embeddings'], atol=1e-4)

    print("Results Verification:")
    if seq_vs_sync and seq_vs_async and sync_vs_async:
        print("  ✓ ALL MODES MATCH: Sequential, Sync Pipeline, and Async Pipeline produce identical results")
    else:
        print("  ✗ VERIFICATION FAILED: Results differ between modes!")
        if not seq_vs_sync:
            max_diff = (result_seq['embeddings'] - result_par['embeddings']).abs().max().item()
            print(f"    Sequential vs Sync Pipeline: Max diff = {max_diff}")
        if not seq_vs_async:
            max_diff = (result_seq['embeddings'] - result_async['embeddings']).abs().max().item()
            print(f"    Sequential vs Async Pipeline: Max diff = {max_diff}")
        if not sync_vs_async:
            max_diff = (result_par['embeddings'] - result_async['embeddings']).abs().max().item()
            print(f"    Sync Pipeline vs Async Pipeline: Max diff = {max_diff}")
    print()

    # 详细统计分析（如果有pipeline executor实例）
    if hasattr(executor, 'last_pipeline_exec') and executor.last_pipeline_exec:
        pipeline_exec = executor.last_pipeline_exec

        # 打印详细时间统计表格
        from gantt_visualizer import print_detailed_timing_table
        print_detailed_timing_table(pipeline_exec.detailed_timing)

        # 性能分析
        perf = pipeline_exec.analyze_performance()
        print("\n" + "="*80)
        print("Pipeline Performance Analysis")
        print("="*80)
        print(f"\nTheoretical Performance:")
        print(f"  Sum of all block execution times (sequential): {perf['sequential_time']:.0f}ms")
        print(f"  Actual pipeline wallclock time:               {perf['pipeline_time']:.0f}ms")
        print(f"\nSpeedup Analysis:")
        print(f"  Theoretical maximum speedup (ideal):          {perf['theoretical_speedup']:.2f}x")
        print(f"  Actual speedup achieved:                      {perf['actual_speedup']:.2f}x")
        print(f"  Pipeline efficiency:                          {perf['efficiency']*100:.1f}%")
        print(f"\nBlock Execution Times (excluding wait):")
        for block_id in sorted(perf['block_exec_times'].keys()):
            exec_time = perf['block_exec_times'][block_id]
            wait_time = perf['block_wait_times'][block_id]
            avg_exec = perf['avg_block_exec_times'][block_id]
            print(f"  Block {block_id}: total_exec={exec_time:.0f}ms, total_wait={wait_time:.0f}ms, avg_exec={avg_exec:.0f}ms")

        # 生成甘特图
        print("\n" + "="*80)
        print("Generating Gantt Chart...")
        print("="*80)
        from gantt_visualizer import generate_gantt_chart
        gantt_output = generate_gantt_chart(pipeline_exec.detailed_timing, 'pipeline_gantt.txt')
        print(gantt_output)
        print("\n✓ Gantt chart saved to: pipeline_gantt.txt")
    else:
        print("\nNote: Detailed pipeline statistics not available (run with use_pipeline_parallelism=True)")

    print("\n✓ Pipeline execution test completed successfully!")


if __name__ == "__main__":
    main()
