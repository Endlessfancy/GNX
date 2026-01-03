"""
Latency Profiling Framework for GNN Pipeline

This module provides precise latency measurement for multi-stage
GNN inference pipelines running on heterogeneous hardware (CPU/GPU/NPU).

Components:
- PipelineProfiler: Central profiler with Chrome Tracing export
- StageExecutor: OpenVINO inference wrapper with PERF_COUNT timing
- DataParallelStage: Multi-device parallel execution with partition/merge timing
- PEP Configurations: Same PEP definitions as executer for consistent testing

Usage:
    from latency import PipelineProfiler, StageExecutor

    profiler = PipelineProfiler("MyPipeline")
    executor = StageExecutor(core, model, "GPU", "Stage1", profiler)

    output, hw_time = executor.run(inputs, batch_id=0)

    profiler.export_chrome_trace("trace.json")
    profiler.analyze_metrics()

PEP Testing:
    from latency.pep_config import PEP1, PEP2, get_two_pep_test_plan
    from latency.test_pep_latency import PEPLatencyTester
"""

from .profiler import PipelineProfiler, TraceEvent, TimingContext
from .stage_executor import (
    StageExecutor,
    AsyncStageExecutor,
    create_dummy_model,
    create_gather_scatter_model,
    OPENVINO_AVAILABLE
)

try:
    from .data_parallel_stage import DataParallelStage, DataParallelStageAsync
except ImportError:
    DataParallelStage = None
    DataParallelStageAsync = None

try:
    from .pep_config import (
        PEP1, PEP2, PEP_CPU_ONLY, PEP_GPU_ONLY, PEP_3BLOCK, PEP_FINE_DP,
        ALL_PEPS, get_two_pep_test_plan, get_single_pep_test_plan,
        analyze_pep, print_pep
    )
except ImportError:
    PEP1 = PEP2 = None

__all__ = [
    # Profiler
    'PipelineProfiler',
    'TraceEvent',
    'TimingContext',
    # Executors
    'StageExecutor',
    'AsyncStageExecutor',
    'DataParallelStage',
    'DataParallelStageAsync',
    # Model utilities
    'create_dummy_model',
    'create_gather_scatter_model',
    'OPENVINO_AVAILABLE',
    # PEP configurations
    'PEP1',
    'PEP2',
    'ALL_PEPS',
    'get_two_pep_test_plan',
    'analyze_pep',
]
