"""
Latency Profiling Framework for GNN Pipeline

This module provides precise latency measurement for multi-stage
GNN inference pipelines running on heterogeneous hardware (CPU/GPU/NPU).

Components:
- PipelineProfiler: Central profiler with Chrome Tracing export
- StageExecutor: OpenVINO inference wrapper with PERF_COUNT timing
- DataParallelStage: Multi-device parallel execution with partition/merge timing

Usage:
    from latency import PipelineProfiler, StageExecutor

    profiler = PipelineProfiler("MyPipeline")
    executor = StageExecutor(core, model, "GPU", "Stage1", profiler)

    output, hw_time = executor.run(inputs, batch_id=0)

    profiler.export_chrome_trace("trace.json")
    profiler.analyze_metrics()
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

__all__ = [
    'PipelineProfiler',
    'TraceEvent',
    'TimingContext',
    'StageExecutor',
    'AsyncStageExecutor',
    'DataParallelStage',
    'DataParallelStageAsync',
    'create_dummy_model',
    'create_gather_scatter_model',
    'OPENVINO_AVAILABLE',
]
