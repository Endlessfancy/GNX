"""
Latency Profiling Framework for GNN Pipeline

This module provides precise latency measurement for multi-stage
GNN inference pipelines running on heterogeneous hardware (CPU/GPU/NPU).

Features:
- Real Flickr dataset with subgraph partitioning
- 7-Stage GraphSAGE models (self-contained, no external dependencies)
- Multi-device async parallel execution with 1-hop Halo Node graph partitioning
- True pipeline execution with cross-cycle buffer
- PERF_COUNT hardware-level timing
- Chrome Tracing visualization

Components:
- FlickrSubgraphLoader: Load and partition Flickr dataset
- GNNModelExporter: Export 7-stage models (ONNX -> OpenVINO IR)
- HaloPartitioner: 1-hop Halo Node graph partitioning for data parallel
- PipelineBenchmark: Industrial-grade pipeline benchmark with cross-cycle buffer
- PipelineProfiler: Central profiler with Chrome Tracing export

Usage:
    from latency import (
        FlickrSubgraphLoader,
        GNNModelExporter,
        PipelineProfiler,
        PipelineBenchmark,
        HaloPartitioner
    )

    # Load data
    loader = FlickrSubgraphLoader(num_subgraphs=8)
    subgraph = loader.get_subgraph(0)

    # Export models
    exporter = GNNModelExporter()
    exporter.export_for_pep(PEP1, max_nodes, max_edges)

    # Run pipeline benchmark
    profiler = PipelineProfiler("MyTest")
    pipeline = PipelineBenchmark(profiler)
    pipeline.add_single_stage(...)
    results = pipeline.run_pipeline(batches, iterations=10)
    profiler.export_chrome_trace("trace.json")

PEP Testing:
    python latency/test_pep_latency.py --pep pep1 --num-subgraphs 8
"""

from latency.profiler import PipelineProfiler, TraceEvent, TimingContext

try:
    from latency.flickr_loader import FlickrSubgraphLoader
except ImportError as e:
    FlickrSubgraphLoader = None
    print(f"Warning: FlickrSubgraphLoader not available: {e}")

try:
    from latency.model_exporter import GNNModelExporter
except ImportError as e:
    GNNModelExporter = None
    print(f"Warning: GNNModelExporter not available: {e}")

try:
    from latency.stage_executor import (
        StageExecutor,
        MultiDeviceExecutor,
        ProfilingResult,
        analyze_profiling,
        OPENVINO_AVAILABLE
    )
except ImportError as e:
    StageExecutor = None
    OPENVINO_AVAILABLE = False
    print(f"Warning: StageExecutor not available: {e}")

try:
    from latency.async_executor import (
        AsyncBlockExecutor,
        PipelineExecutor,
        BlockExecutionResult
    )
except ImportError as e:
    AsyncBlockExecutor = None
    PipelineExecutor = None

try:
    from latency.npu_utils import (
        pad_array_for_npu,
        unpad_array_from_npu,
        pad_graph_data_for_npu,
        unpad_graph_output_from_npu,
        prepare_npu_inputs,
        unpad_npu_outputs,
        NPUPaddingInfo
    )
except ImportError as e:
    NPUPaddingInfo = None
    print(f"Warning: npu_utils not available: {e}")

try:
    from latency.pep_config import (
        PEP1, PEP2, PEP_CPU_ONLY, PEP_GPU_ONLY, PEP_3BLOCK, PEP_FINE_DP,
        ALL_PEPS, get_two_pep_test_plan, get_single_pep_test_plan,
        analyze_pep, print_pep
    )
except ImportError:
    PEP1 = PEP2 = None

try:
    from latency.graph_partitioner import (
        HaloPartitioner,
        PartitionData
    )
except ImportError as e:
    HaloPartitioner = None
    PartitionData = None
    print(f"Warning: graph_partitioner not available: {e}")

try:
    from latency.pipeline_executor import (
        PipelineBenchmark,
        PipelineBuffer,
        SingleDeviceStage,
        DataParallelStage,
        StageStats,
        OPENVINO_AVAILABLE as PIPELINE_OV_AVAILABLE
    )
except ImportError as e:
    PipelineBenchmark = None
    PipelineBuffer = None
    SingleDeviceStage = None
    DataParallelStage = None
    print(f"Warning: pipeline_executor not available: {e}")

__all__ = [
    # Profiler
    'PipelineProfiler',
    'TraceEvent',
    'TimingContext',
    # Data loading
    'FlickrSubgraphLoader',
    # Model export
    'GNNModelExporter',
    # Graph Partitioning (1-hop Halo Node)
    'HaloPartitioner',
    'PartitionData',
    # Pipeline Benchmark (industrial-grade)
    'PipelineBenchmark',
    'PipelineBuffer',
    'SingleDeviceStage',
    'DataParallelStage',
    'StageStats',
    # Legacy Executors (deprecated, use PipelineBenchmark instead)
    'StageExecutor',
    'MultiDeviceExecutor',
    'AsyncBlockExecutor',
    'PipelineExecutor',
    # Results
    'ProfilingResult',
    'BlockExecutionResult',
    # NPU utilities
    'NPUPaddingInfo',
    'pad_array_for_npu',
    'unpad_array_from_npu',
    'pad_graph_data_for_npu',
    'unpad_graph_output_from_npu',
    'prepare_npu_inputs',
    'unpad_npu_outputs',
    # Utilities
    'analyze_profiling',
    'OPENVINO_AVAILABLE',
    # PEP configurations
    'PEP1',
    'PEP2',
    'ALL_PEPS',
    'get_two_pep_test_plan',
    'analyze_pep',
]
