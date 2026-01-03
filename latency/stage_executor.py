"""
Stage Executor - OpenVINO Inference Wrapper with Precise Timing

Wraps OpenVINO InferRequest with:
- Hardware time measurement via PERF_COUNT
- Wall clock time measurement (ns precision)
- Data transfer overhead detection
- Async execution support
- NPU static shape padding/unpadding
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

try:
    import openvino as ov
    from openvino.runtime import Core, CompiledModel, InferRequest
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("WARNING: OpenVINO not available")

try:
    from profiler import PipelineProfiler
    from npu_utils import (
        prepare_npu_inputs,
        unpad_npu_outputs,
        NPUPaddingInfo
    )
except ImportError:
    from .profiler import PipelineProfiler
    from .npu_utils import (
        prepare_npu_inputs,
        unpad_npu_outputs,
        NPUPaddingInfo
    )


@dataclass
class ProfilingResult:
    """Results from PERF_COUNT profiling."""
    device_time_ms: float  # Total device time
    compute_time_ms: float  # Actual compute time
    io_time_ms: float  # I/O and reorder time


def analyze_profiling(prof_info) -> ProfilingResult:
    """
    Analyze OpenVINO profiling info to extract timing breakdown.

    Args:
        prof_info: InferRequest.profiling_info

    Returns:
        ProfilingResult with device, compute, and I/O times
    """
    total_time = 0.0
    compute_time = 0.0
    io_time = 0.0

    # Types considered as I/O overhead
    io_types = {'Input', 'Output', 'Reorder', 'Convert', 'Parameter', 'Result'}

    for info in prof_info:
        # Check for EXECUTED status
        if hasattr(info.status, 'name'):
            status = info.status.name
        else:
            status = str(info.status)

        if 'EXECUTED' in status:
            # Get time in milliseconds
            if hasattr(info.real_time, 'total_seconds'):
                time_ms = info.real_time.total_seconds() * 1000
            else:
                # Older API: real_time is in microseconds
                time_ms = info.real_time / 1000.0

            total_time += time_ms

            if info.node_type in io_types:
                io_time += time_ms
            else:
                compute_time += time_ms

    return ProfilingResult(
        device_time_ms=total_time,
        compute_time_ms=compute_time,
        io_time_ms=io_time
    )


class StageExecutor:
    """
    Single-device stage executor with precise timing.

    Features:
    - Loads OpenVINO IR models from file
    - PERF_COUNT for hardware timing
    - Wall clock timing around inference
    - Support for both sync and async execution
    """

    def __init__(self,
                 core: 'Core',
                 model_path: Union[str, Path],
                 device: str,
                 stage_name: str,
                 profiler: PipelineProfiler,
                 stream_id: int = 0,
                 enable_perf_count: bool = True,
                 npu_static_nodes: Optional[int] = None,
                 npu_static_edges: Optional[int] = None):
        """
        Initialize stage executor.

        Args:
            core: OpenVINO Core instance
            model_path: Path to IR model (.xml)
            device: Target device ("CPU", "GPU", "NPU")
            stage_name: Stage identifier for logging
            profiler: PipelineProfiler instance
            stream_id: Stream ID for data parallel execution
            enable_perf_count: Enable hardware performance counters
            npu_static_nodes: Static node count for NPU padding (required for NPU)
            npu_static_edges: Static edge count for NPU padding (required for NPU)
        """
        self.core = core
        self.device = device
        self.stage_name = stage_name
        self.profiler = profiler
        self.stream_id = stream_id
        self.enable_perf_count = enable_perf_count

        # NPU padding configuration
        self.npu_static_nodes = npu_static_nodes
        self.npu_static_edges = npu_static_edges
        self._pending_padding_info: Optional[NPUPaddingInfo] = None

        # Compile model with performance counting
        config = {}
        if enable_perf_count:
            config["PERF_COUNT"] = "YES"

        print(f"  Loading {stage_name} on {device} (stream {stream_id})...")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            ov_model = core.read_model(str(model_path))
            self.compiled_model = core.compile_model(ov_model, device, config=config)
            print(f"    Compiled successfully on {device}")
        except Exception as e:
            print(f"    Failed to compile on {device}: {e}")
            print(f"    Falling back to CPU")
            self.device = "CPU"
            self.compiled_model = core.compile_model(ov_model, "CPU", config=config)

        self.request = self.compiled_model.create_infer_request()

        # Cache input/output info
        self.input_names = [inp.any_name for inp in self.compiled_model.inputs]
        self.output_names = [out.any_name for out in self.compiled_model.outputs]

        # For async timing
        self._async_start_ns: Optional[int] = None

    def run(self, inputs: Dict[str, np.ndarray], batch_id: int) -> Tuple[Dict[str, np.ndarray], ProfilingResult]:
        """
        Execute synchronous inference with timing.

        Args:
            inputs: Input tensors {name: ndarray}
            batch_id: Batch/cycle identifier

        Returns:
            (outputs, profiling_result): Output tensors and profiling results
        """
        # 1. Record wall clock start
        wall_start_ns = time.perf_counter_ns()

        # NPU: pad inputs to static shape
        padding_info = None
        if self.device == 'NPU' and self.npu_static_nodes is not None:
            inputs, padding_info = prepare_npu_inputs(
                inputs,
                self.npu_static_nodes,
                self.npu_static_edges or 0
            )

        # 2. Set inputs - map by order if names don't match
        self._set_inputs(inputs)

        # 3. Run inference
        self.request.infer()

        # 4. Record wall clock end
        wall_end_ns = time.perf_counter_ns()

        # 5. Extract hardware time from profiling info
        profiling = self._extract_profiling()

        # 6. Log to profiler
        self.profiler.log_execution(
            stage_name=self.stage_name,
            device=self.device,
            batch_id=batch_id,
            wall_start_ns=wall_start_ns,
            wall_end_ns=wall_end_ns,
            hw_duration_ms=profiling.device_time_ms,
            stream_id=self.stream_id,
            extra_args={
                "compute_time_ms": profiling.compute_time_ms,
                "io_time_ms": profiling.io_time_ms
            }
        )

        # 7. Get outputs
        outputs = self._get_outputs()

        # NPU: unpad outputs
        if padding_info is not None:
            outputs = unpad_npu_outputs(outputs, padding_info)

        return outputs, profiling

    def _set_inputs(self, inputs: Dict[str, np.ndarray]):
        """Set input tensors, mapping by order if names don't match."""
        input_values = list(inputs.values())

        for i, model_input_name in enumerate(self.input_names):
            if i < len(input_values):
                data = input_values[i]
                # Ensure correct dtype
                if data.dtype != np.float32 and 'edge' not in model_input_name.lower():
                    data = data.astype(np.float32)
                self.request.set_tensor(model_input_name, ov.Tensor(data))

    def _get_outputs(self) -> Dict[str, np.ndarray]:
        """Get output tensors."""
        outputs = {}
        for i, name in enumerate(self.output_names):
            # Use 'output' as key for single output, otherwise use model names
            out_key = 'output' if len(self.output_names) == 1 else name
            outputs[out_key] = self.request.get_tensor(name).data.copy()
        return outputs

    def _extract_profiling(self) -> ProfilingResult:
        """Extract profiling info from last inference."""
        if not self.enable_perf_count:
            return ProfilingResult(0.0, 0.0, 0.0)

        try:
            return analyze_profiling(self.request.profiling_info)
        except Exception:
            return ProfilingResult(0.0, 0.0, 0.0)

    def start_async(self, inputs: Dict[str, np.ndarray]) -> int:
        """
        Start asynchronous inference.

        Args:
            inputs: Input tensors

        Returns:
            wall_start_ns: Start timestamp for later timing calculation
        """
        wall_start_ns = time.perf_counter_ns()
        self._async_start_ns = wall_start_ns

        # NPU: pad inputs to static shape
        if self.device == 'NPU' and self.npu_static_nodes is not None:
            inputs, self._pending_padding_info = prepare_npu_inputs(
                inputs,
                self.npu_static_nodes,
                self.npu_static_edges or 0
            )
        else:
            self._pending_padding_info = None

        self._set_inputs(inputs)
        self.request.start_async()

        return wall_start_ns

    def wait(self, batch_id: int) -> Tuple[Dict[str, np.ndarray], ProfilingResult]:
        """
        Wait for async inference to complete and record timing.

        Args:
            batch_id: Batch identifier

        Returns:
            (outputs, profiling_result): Output tensors and profiling results
        """
        self.request.wait()

        wall_end_ns = time.perf_counter_ns()
        wall_start_ns = self._async_start_ns or wall_end_ns

        # Extract profiling info
        profiling = self._extract_profiling()

        # Log to profiler
        self.profiler.log_execution(
            stage_name=self.stage_name,
            device=self.device,
            batch_id=batch_id,
            wall_start_ns=wall_start_ns,
            wall_end_ns=wall_end_ns,
            hw_duration_ms=profiling.device_time_ms,
            stream_id=self.stream_id,
            extra_args={
                "compute_time_ms": profiling.compute_time_ms,
                "io_time_ms": profiling.io_time_ms
            }
        )

        # Get outputs
        outputs = self._get_outputs()

        # NPU: unpad outputs
        if self._pending_padding_info is not None:
            outputs = unpad_npu_outputs(outputs, self._pending_padding_info)

        # Reset async state
        self._async_start_ns = None
        self._pending_padding_info = None

        return outputs, profiling

    def get_layer_timings(self) -> List[Dict]:
        """
        Get detailed per-layer timing information.

        Returns:
            List of layer timing dictionaries
        """
        layers = []

        if not self.enable_perf_count:
            return layers

        try:
            for info in self.request.profiling_info:
                # Get time in milliseconds
                if hasattr(info.real_time, 'total_seconds'):
                    time_us = info.real_time.total_seconds() * 1e6
                else:
                    time_us = info.real_time

                layers.append({
                    "name": info.node_name,
                    "type": info.node_type,
                    "status": str(info.status),
                    "real_time_us": time_us,
                    "cpu_time_us": info.cpu_time if hasattr(info, 'cpu_time') else 0,
                })
        except Exception:
            pass

        return layers


class MultiDeviceExecutor:
    """
    Multi-device executor for data parallel execution.

    Manages multiple StageExecutors on different devices and
    supports parallel async inference.
    """

    def __init__(self,
                 core: 'Core',
                 model_paths: Dict[str, Path],
                 stage_name: str,
                 profiler: PipelineProfiler):
        """
        Initialize multi-device executor.

        Args:
            core: OpenVINO Core instance
            model_paths: {device: model_path} mapping
            stage_name: Stage name for logging
            profiler: PipelineProfiler instance
        """
        self.executors: Dict[str, StageExecutor] = {}
        self.devices: List[str] = []

        for device, model_path in model_paths.items():
            try:
                executor = StageExecutor(
                    core=core,
                    model_path=model_path,
                    device=device,
                    stage_name=f"{stage_name}_{device}",
                    profiler=profiler,
                    stream_id=len(self.executors)
                )
                self.executors[device] = executor
                self.devices.append(device)
            except Exception as e:
                print(f"  Warning: Failed to initialize {device}: {e}")

    def run_parallel(self,
                     inputs_per_device: Dict[str, Dict[str, np.ndarray]],
                     batch_id: int) -> Dict[str, Tuple[Dict[str, np.ndarray], ProfilingResult]]:
        """
        Run parallel async inference on all devices.

        Args:
            inputs_per_device: {device: {input_name: data}}
            batch_id: Batch identifier

        Returns:
            {device: (outputs, profiling_result)}
        """
        # Start all async inferences
        for device in self.devices:
            if device in inputs_per_device:
                self.executors[device].start_async(inputs_per_device[device])

        # Wait for all and collect results
        results = {}
        for device in self.devices:
            if device in inputs_per_device:
                outputs, profiling = self.executors[device].wait(batch_id)
                results[device] = (outputs, profiling)

        return results


if __name__ == "__main__":
    if not OPENVINO_AVAILABLE:
        print("OpenVINO not available, skipping test")
        exit(0)

    # Test basic execution
    print("Testing StageExecutor...")

    from model_exporter import GNNModelExporter

    # Export a test model
    exporter = GNNModelExporter()
    exporter.export_for_stages([6, 7], "CPU", 1000, 5000)

    # Create profiler
    profiler = PipelineProfiler("Test")
    core = Core()

    # Get model path
    model_path = exporter.get_model_path([6, 7], "CPU")
    if model_path is None:
        print("Model not found!")
        exit(1)

    # Create executor
    executor = StageExecutor(
        core=core,
        model_path=model_path,
        device="CPU",
        stage_name="TestStage",
        profiler=profiler
    )

    # Run some batches
    for i in range(5):
        # Stage 6-7 takes (mean_agg, x)
        inputs = {
            'mean_agg': np.random.randn(1000, 500).astype(np.float32),
            'x': np.random.randn(1000, 500).astype(np.float32)
        }
        outputs, profiling = executor.run(inputs, batch_id=i)
        print(f"  Batch {i}: output shape = {outputs['output'].shape}, "
              f"device_time = {profiling.device_time_ms:.3f} ms, "
              f"compute = {profiling.compute_time_ms:.3f} ms")

    # Analyze
    profiler.export_chrome_trace("test_stage_executor.json")
    profiler.analyze_metrics()
