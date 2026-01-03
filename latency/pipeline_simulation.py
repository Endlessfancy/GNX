"""
Pipeline Simulation - Complete Multi-Stage GNN Inference Pipeline Profiling

This script simulates a multi-stage GNN inference pipeline with:
- Multiple stages running on different devices (CPU, GPU, NPU)
- Data parallelism support (splitting across devices)
- Precise timing measurement
- Chrome Tracing visualization output

Usage:
    python pipeline_simulation.py [--num-batches 10] [--output trace.json]

Output:
    - Chrome Tracing JSON file (open in chrome://tracing or Perfetto)
    - Console analysis with utilization, bubbles, and latency metrics
"""

import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from profiler import PipelineProfiler
from stage_executor import StageExecutor, create_dummy_model, create_gather_scatter_model

try:
    import openvino as ov
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("WARNING: OpenVINO not available")

try:
    from data_parallel_stage import DataParallelStage, DataParallelStageAsync
    DATA_PARALLEL_AVAILABLE = True
except ImportError:
    DATA_PARALLEL_AVAILABLE = False


class PipelineSimulator:
    """
    Simulates a multi-stage GNN inference pipeline

    Stages:
    1. GraphPrep (CPU): Data preprocessing
    2. Embedding (GPU/NPU): Main GNN computation (can be data parallel)
    3. Classifier (CPU/GPU): Final classification
    """

    def __init__(self,
                 profiler: PipelineProfiler,
                 use_gpu: bool = True,
                 use_npu: bool = False,
                 use_data_parallel: bool = False,
                 dp_ratios: List[float] = None):
        """
        Initialize pipeline

        Args:
            profiler: PipelineProfiler instance
            use_gpu: Use GPU for computation
            use_npu: Use NPU for computation
            use_data_parallel: Enable data parallelism
            dp_ratios: Data parallel split ratios
        """
        self.profiler = profiler
        self.use_gpu = use_gpu
        self.use_npu = use_npu
        self.use_data_parallel = use_data_parallel
        self.dp_ratios = dp_ratios or [0.5, 0.5]

        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO is required for pipeline simulation")

        self.core = Core()
        available_devices = self.core.available_devices
        print(f"Available devices: {available_devices}")

        # Validate device availability
        if use_gpu and "GPU" not in available_devices:
            print("WARNING: GPU not available, falling back to CPU")
            self.use_gpu = False
        if use_npu and "NPU" not in available_devices:
            print("WARNING: NPU not available, falling back to CPU")
            self.use_npu = False

        self._init_stages()

    def _init_stages(self):
        """Initialize all pipeline stages"""
        print("\n=== Initializing Pipeline Stages ===")

        # Stage 1: GraphPrep (CPU) - lightweight preprocessing
        model_prep = create_dummy_model((1000, 128), (1000, 128), compute_ops=1)
        self.stage_prep = StageExecutor(
            core=self.core,
            model=model_prep,
            device="CPU",
            stage_name="Stage1_GraphPrep",
            profiler=self.profiler
        )

        # Stage 2: Embedding - main computation
        model_embed = create_dummy_model((1000, 128), (1000, 256), compute_ops=5)

        if self.use_data_parallel and DATA_PARALLEL_AVAILABLE:
            # Create multiple executors for data parallelism
            embed_executors = []

            if self.use_gpu:
                embed_exec_gpu = StageExecutor(
                    core=self.core,
                    model=model_embed,
                    device="GPU",
                    stage_name="Stage2_Embedding",
                    profiler=self.profiler,
                    stream_id=0
                )
                embed_executors.append(embed_exec_gpu)

            if self.use_npu:
                embed_exec_npu = StageExecutor(
                    core=self.core,
                    model=model_embed,
                    device="NPU",
                    stage_name="Stage2_Embedding",
                    profiler=self.profiler,
                    stream_id=1
                )
                embed_executors.append(embed_exec_npu)

            if len(embed_executors) < 2:
                # Fallback: use two CPU streams
                for i in range(2 - len(embed_executors)):
                    exec_cpu = StageExecutor(
                        core=self.core,
                        model=model_embed,
                        device="CPU",
                        stage_name="Stage2_Embedding",
                        profiler=self.profiler,
                        stream_id=len(embed_executors)
                    )
                    embed_executors.append(exec_cpu)

            self.stage_embed = DataParallelStageAsync(
                name="Stage2_Embedding_DP",
                executors=embed_executors,
                ratios=self.dp_ratios[:len(embed_executors)],
                profiler=self.profiler
            )
            self.embed_is_dp = True
        else:
            # Single device embedding
            device = "GPU" if self.use_gpu else ("NPU" if self.use_npu else "CPU")
            self.stage_embed = StageExecutor(
                core=self.core,
                model=model_embed,
                device=device,
                stage_name="Stage2_Embedding",
                profiler=self.profiler
            )
            self.embed_is_dp = False

        # Stage 3: Classifier (CPU or GPU)
        model_classify = create_dummy_model((1000, 256), (1000, 64), compute_ops=2)
        classify_device = "GPU" if self.use_gpu else "CPU"
        self.stage_classify = StageExecutor(
            core=self.core,
            model=model_classify,
            device=classify_device,
            stage_name="Stage3_Classifier",
            profiler=self.profiler
        )

        print("=== Pipeline Initialized ===\n")

    def run_batch(self, batch_id: int, input_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Run a single batch through the pipeline

        Args:
            batch_id: Batch identifier
            input_data: Input tensor [num_nodes, feature_dim]

        Returns:
            (output, timing_info)
        """
        timing_info = {}

        # Stage 1: GraphPrep
        inputs_prep = {"input": input_data}
        outputs_prep, hw_time_prep = self.stage_prep.run(inputs_prep, batch_id)
        timing_info['stage1_ms'] = hw_time_prep

        # Stage 2: Embedding
        if self.embed_is_dp:
            outputs_embed, dp_timing = self.stage_embed.run(
                {"input": outputs_prep["output"]}, batch_id
            )
            timing_info['stage2_ms'] = dp_timing['stage_total_ms']
            timing_info['stage2_details'] = dp_timing
        else:
            inputs_embed = {"input": outputs_prep["output"]}
            outputs_embed, hw_time_embed = self.stage_embed.run(inputs_embed, batch_id)
            timing_info['stage2_ms'] = hw_time_embed

        # Stage 3: Classifier
        inputs_classify = {"input": outputs_embed["output"]}
        outputs_classify, hw_time_classify = self.stage_classify.run(inputs_classify, batch_id)
        timing_info['stage3_ms'] = hw_time_classify

        # Total
        timing_info['total_ms'] = (
            timing_info['stage1_ms'] +
            timing_info['stage2_ms'] +
            timing_info['stage3_ms']
        )

        return outputs_classify["output"], timing_info

    def run_pipeline(self, num_batches: int, batch_size: int = 1000) -> Dict:
        """
        Run the complete pipeline for multiple batches

        Args:
            num_batches: Number of batches to process
            batch_size: Number of nodes per batch

        Returns:
            Aggregated timing statistics
        """
        print(f"\n=== Running Pipeline: {num_batches} batches ===")

        all_timings = []

        for batch_id in range(num_batches):
            # Generate input data
            input_data = np.random.randn(batch_size, 128).astype(np.float32)

            # Run pipeline
            output, timing = self.run_batch(batch_id, input_data)

            all_timings.append(timing)

            if batch_id % max(1, num_batches // 5) == 0:
                print(f"  Batch {batch_id}/{num_batches}: "
                      f"total={timing['total_ms']:.2f}ms, "
                      f"output shape={output.shape}")

        # Aggregate statistics
        stats = {
            'num_batches': num_batches,
            'avg_total_ms': np.mean([t['total_ms'] for t in all_timings]),
            'avg_stage1_ms': np.mean([t['stage1_ms'] for t in all_timings]),
            'avg_stage2_ms': np.mean([t['stage2_ms'] for t in all_timings]),
            'avg_stage3_ms': np.mean([t['stage3_ms'] for t in all_timings]),
            'min_total_ms': np.min([t['total_ms'] for t in all_timings]),
            'max_total_ms': np.max([t['total_ms'] for t in all_timings]),
        }

        print(f"\n=== Pipeline Summary ===")
        print(f"  Avg Total Latency: {stats['avg_total_ms']:.2f} ms")
        print(f"  Stage 1 (GraphPrep): {stats['avg_stage1_ms']:.2f} ms")
        print(f"  Stage 2 (Embedding): {stats['avg_stage2_ms']:.2f} ms")
        print(f"  Stage 3 (Classifier): {stats['avg_stage3_ms']:.2f} ms")
        print(f"  Latency Range: [{stats['min_total_ms']:.2f}, {stats['max_total_ms']:.2f}] ms")

        return stats

    def shutdown(self):
        """Cleanup resources"""
        if self.embed_is_dp and hasattr(self.stage_embed, 'shutdown'):
            self.stage_embed.shutdown()


def run_simple_simulation():
    """Run a simple 3-stage pipeline simulation"""
    if not OPENVINO_AVAILABLE:
        print("OpenVINO not available, running mock simulation")
        return run_mock_simulation()

    profiler = PipelineProfiler("GNN_Pipeline")

    # Create simulator
    simulator = PipelineSimulator(
        profiler=profiler,
        use_gpu=True,
        use_npu=False,
        use_data_parallel=False
    )

    # Run pipeline
    stats = simulator.run_pipeline(num_batches=10, batch_size=1000)

    # Export and analyze
    profiler.export_chrome_trace("pipeline_trace.json")
    profiler.analyze_metrics()

    simulator.shutdown()

    return stats


def run_data_parallel_simulation():
    """Run simulation with data parallelism"""
    if not OPENVINO_AVAILABLE:
        print("OpenVINO required for data parallel simulation")
        return None

    profiler = PipelineProfiler("GNN_Pipeline_DP")

    simulator = PipelineSimulator(
        profiler=profiler,
        use_gpu=True,
        use_npu=False,
        use_data_parallel=True,
        dp_ratios=[0.5, 0.5]
    )

    stats = simulator.run_pipeline(num_batches=10, batch_size=1000)

    profiler.export_chrome_trace("pipeline_trace_dp.json")
    profiler.analyze_metrics()

    simulator.shutdown()

    return stats


def run_mock_simulation():
    """Run mock simulation without OpenVINO (for testing profiler only)"""
    print("\n=== Running Mock Simulation ===")

    profiler = PipelineProfiler("Mock_Pipeline")

    num_batches = 10

    for batch_id in range(num_batches):
        # Simulate Stage 1 (CPU)
        start = time.perf_counter_ns()
        time.sleep(0.001)  # 1ms
        end = time.perf_counter_ns()
        profiler.log_execution("Stage1_GraphPrep", "CPU", batch_id, start, end, 0.8)

        # Simulate Stage 2 (GPU)
        start = time.perf_counter_ns()
        time.sleep(0.003)  # 3ms
        end = time.perf_counter_ns()
        profiler.log_execution("Stage2_Embedding", "GPU", batch_id, start, end, 2.5)

        # Simulate Stage 3 (CPU)
        start = time.perf_counter_ns()
        time.sleep(0.001)  # 1ms
        end = time.perf_counter_ns()
        profiler.log_execution("Stage3_Classifier", "CPU", batch_id, start, end, 0.9)

        # Small gap between batches
        time.sleep(0.0005)

    profiler.export_chrome_trace("mock_pipeline_trace.json")
    profiler.analyze_metrics()

    return {'num_batches': num_batches}


def main():
    parser = argparse.ArgumentParser(description="GNN Pipeline Simulation")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size (nodes)")
    parser.add_argument("--output", type=str, default="pipeline_trace.json", help="Output trace file")
    parser.add_argument("--mode", choices=["simple", "dp", "mock"], default="simple",
                        help="Simulation mode: simple, dp (data parallel), or mock")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU")
    parser.add_argument("--use-npu", action="store_true", help="Use NPU")

    args = parser.parse_args()

    print("=" * 70)
    print("GNN Pipeline Latency Profiling")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Batches: {args.num_batches}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output}")

    if args.mode == "mock":
        run_mock_simulation()
    elif args.mode == "dp":
        run_data_parallel_simulation()
    else:
        run_simple_simulation()

    print("\n" + "=" * 70)
    print("Profiling Complete!")
    print(f"Open {args.output} in chrome://tracing or https://ui.perfetto.dev")
    print("=" * 70)


if __name__ == "__main__":
    main()
