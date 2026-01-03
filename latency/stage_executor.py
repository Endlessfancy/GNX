"""
Stage Executor - OpenVINO Inference Wrapper with Precise Timing

Wraps OpenVINO InferRequest with:
- Hardware time measurement via PERF_COUNT
- Wall clock time measurement (ns precision)
- Data transfer overhead detection
- Async execution support
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

try:
    import openvino as ov
    from openvino.runtime import Core, CompiledModel, InferRequest, AsyncInferQueue
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("WARNING: OpenVINO not available")

from profiler import PipelineProfiler, TimingContext


class StageExecutor:
    """
    Single-device stage executor with precise timing

    Features:
    - OpenVINO inference with PERF_COUNT for hardware timing
    - Wall clock timing around start_async/wait
    - Data transfer overhead detection via Input/Reorder layers
    - Support for both sync and async execution
    """

    def __init__(self,
                 core: 'Core',
                 model: Union[str, 'ov.Model'],
                 device: str,
                 stage_name: str,
                 profiler: PipelineProfiler,
                 stream_id: int = 0,
                 enable_perf_count: bool = True):
        """
        Initialize stage executor

        Args:
            core: OpenVINO Core instance
            model: Model path (ONNX/IR) or OpenVINO Model object
            device: Target device ("CPU", "GPU", "NPU")
            stage_name: Stage identifier for logging
            profiler: PipelineProfiler instance
            stream_id: Stream ID for data parallel execution
            enable_perf_count: Enable hardware performance counters
        """
        self.core = core
        self.device = device
        self.stage_name = stage_name
        self.profiler = profiler
        self.stream_id = stream_id
        self.enable_perf_count = enable_perf_count

        # Compile model with performance counting
        config = {}
        if enable_perf_count:
            config["PERF_COUNT"] = "YES"

        print(f"  Loading {stage_name} on {device} (stream {stream_id})...")

        if isinstance(model, str):
            ov_model = core.read_model(model)
        else:
            ov_model = model

        try:
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

    def run(self, inputs: Dict[str, np.ndarray], batch_id: int) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Execute synchronous inference with timing

        Args:
            inputs: Input tensors {name: ndarray}
            batch_id: Batch/cycle identifier

        Returns:
            (outputs, hw_time_ms): Output tensors and hardware execution time
        """
        # 1. Record wall clock start
        wall_start_ns = time.perf_counter_ns()

        # 2. Set inputs and run inference
        # Map input dict to model's actual input names by order
        input_values = list(inputs.values())
        for i, model_input_name in enumerate(self.input_names):
            if i < len(input_values):
                self.request.set_tensor(model_input_name, ov.Tensor(input_values[i]))

        self.request.infer()

        # 3. Record wall clock end
        wall_end_ns = time.perf_counter_ns()

        # 4. Extract hardware time from profiling info
        hw_time_ms, transfer_time_ms = self._extract_profiling_info()

        # 5. Log to profiler
        self.profiler.log_execution(
            stage_name=self.stage_name,
            device=self.device,
            batch_id=batch_id,
            wall_start_ns=wall_start_ns,
            wall_end_ns=wall_end_ns,
            hw_duration_ms=hw_time_ms,
            stream_id=self.stream_id,
            extra_args={"transfer_time_ms": transfer_time_ms}
        )

        # 6. Get outputs - use generic 'output' key for compatibility
        outputs = {}
        for i, name in enumerate(self.output_names):
            # Use 'output' as key for single output, otherwise use model names
            out_key = 'output' if len(self.output_names) == 1 else name
            outputs[out_key] = self.request.get_tensor(name).data.copy()

        return outputs, hw_time_ms

    def start_async(self, inputs: Dict[str, np.ndarray]) -> int:
        """
        Start asynchronous inference

        Args:
            inputs: Input tensors

        Returns:
            wall_start_ns: Start timestamp for later timing calculation
        """
        wall_start_ns = time.perf_counter_ns()

        # Map input dict to model's actual input names by order
        input_values = list(inputs.values())
        for i, model_input_name in enumerate(self.input_names):
            if i < len(input_values):
                self.request.set_tensor(model_input_name, ov.Tensor(input_values[i]))

        self.request.start_async()

        return wall_start_ns

    def wait(self, wall_start_ns: int, batch_id: int) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Wait for async inference to complete and record timing

        Args:
            wall_start_ns: Start timestamp from start_async
            batch_id: Batch identifier

        Returns:
            (outputs, hw_time_ms): Output tensors and hardware execution time
        """
        self.request.wait()

        wall_end_ns = time.perf_counter_ns()

        # Extract profiling info
        hw_time_ms, transfer_time_ms = self._extract_profiling_info()

        # Log to profiler
        self.profiler.log_execution(
            stage_name=self.stage_name,
            device=self.device,
            batch_id=batch_id,
            wall_start_ns=wall_start_ns,
            wall_end_ns=wall_end_ns,
            hw_duration_ms=hw_time_ms,
            stream_id=self.stream_id,
            extra_args={"transfer_time_ms": transfer_time_ms}
        )

        # Get outputs - use generic 'output' key for compatibility
        outputs = {}
        for i, name in enumerate(self.output_names):
            out_key = 'output' if len(self.output_names) == 1 else name
            outputs[out_key] = self.request.get_tensor(name).data.copy()

        return outputs, hw_time_ms

    def _extract_profiling_info(self) -> Tuple[float, float]:
        """
        Extract hardware execution time and data transfer time from profiling info

        Returns:
            (hw_time_ms, transfer_time_ms)
        """
        hw_time_ms = 0.0
        transfer_time_ms = 0.0

        if not self.enable_perf_count:
            return hw_time_ms, transfer_time_ms

        try:
            for info in self.request.profiling_info:
                # Check if layer was executed
                if info.status.name == "EXECUTED":
                    layer_time_ms = info.real_time / 1000.0  # us -> ms

                    # Check if this is a data transfer layer
                    layer_name = info.node_name.lower()
                    if "reorder" in layer_name or "input" in layer_name or "output" in layer_name:
                        transfer_time_ms += layer_time_ms
                    else:
                        hw_time_ms += layer_time_ms

        except Exception as e:
            # Profiling may not be available on all devices
            pass

        return hw_time_ms, transfer_time_ms

    def get_layer_timings(self) -> List[Dict]:
        """
        Get detailed per-layer timing information

        Returns:
            List of layer timing dictionaries
        """
        layers = []

        if not self.enable_perf_count:
            return layers

        try:
            for info in self.request.profiling_info:
                layers.append({
                    "name": info.node_name,
                    "type": info.node_type,
                    "status": info.status.name,
                    "real_time_us": info.real_time,
                    "cpu_time_us": info.cpu_time,
                })
        except Exception:
            pass

        return layers


class AsyncStageExecutor:
    """
    Async stage executor using AsyncInferQueue for pipelined execution

    Supports multiple concurrent inference requests for better throughput.
    """

    def __init__(self,
                 core: 'Core',
                 model: Union[str, 'ov.Model'],
                 device: str,
                 stage_name: str,
                 profiler: PipelineProfiler,
                 num_requests: int = 2,
                 stream_id: int = 0):
        """
        Initialize async executor with inference queue

        Args:
            core: OpenVINO Core instance
            model: Model path or OpenVINO Model
            device: Target device
            stage_name: Stage identifier
            profiler: Pipeline profiler
            num_requests: Number of parallel inference requests
            stream_id: Stream ID for data parallel
        """
        self.device = device
        self.stage_name = stage_name
        self.profiler = profiler
        self.stream_id = stream_id
        self.num_requests = num_requests

        # Compile model with performance counting
        config = {"PERF_COUNT": "YES"}

        print(f"  Loading {stage_name} on {device} with {num_requests} async requests...")

        if isinstance(model, str):
            ov_model = core.read_model(model)
        else:
            ov_model = model

        try:
            self.compiled_model = core.compile_model(ov_model, device, config=config)
        except Exception as e:
            print(f"    Failed on {device}, falling back to CPU: {e}")
            self.device = "CPU"
            self.compiled_model = core.compile_model(ov_model, "CPU", config=config)

        # Create async queue
        self.async_queue = AsyncInferQueue(self.compiled_model, num_requests)

        # Timing tracking
        self._pending_starts: Dict[int, int] = {}  # request_id -> start_ns
        self._pending_batches: Dict[int, int] = {}  # request_id -> batch_id
        self._results: Dict[int, Tuple[Dict, float]] = {}  # batch_id -> (outputs, hw_time)

        # Set callback
        self.async_queue.set_callback(self._completion_callback)

        # Cache output names
        self.output_names = [out.any_name for out in self.compiled_model.outputs]

    def _completion_callback(self, request: InferRequest, userdata: Any):
        """Callback when inference completes"""
        request_id = id(request)
        wall_end_ns = time.perf_counter_ns()

        if request_id in self._pending_starts:
            wall_start_ns = self._pending_starts.pop(request_id)
            batch_id = self._pending_batches.pop(request_id, -1)

            # Extract timing
            hw_time_ms = 0.0
            try:
                for info in request.profiling_info:
                    if info.status.name == "EXECUTED":
                        hw_time_ms += info.real_time / 1000.0
            except:
                pass

            # Log to profiler
            self.profiler.log_execution(
                stage_name=self.stage_name,
                device=self.device,
                batch_id=batch_id,
                wall_start_ns=wall_start_ns,
                wall_end_ns=wall_end_ns,
                hw_duration_ms=hw_time_ms,
                stream_id=self.stream_id
            )

            # Store result
            outputs = {}
            for name in self.output_names:
                outputs[name] = request.get_tensor(name).data.copy()
            self._results[batch_id] = (outputs, hw_time_ms)

    def start_async(self, inputs: Dict[str, np.ndarray], batch_id: int):
        """
        Start async inference

        Args:
            inputs: Input tensors
            batch_id: Batch identifier
        """
        wall_start_ns = time.perf_counter_ns()

        # Get next available request
        idle_id = self.async_queue.get_idle_request_id()
        request = self.async_queue[idle_id]

        # Track timing
        request_id = id(request)
        self._pending_starts[request_id] = wall_start_ns
        self._pending_batches[request_id] = batch_id

        # Set inputs
        for name, data in inputs.items():
            request.set_tensor(name, ov.Tensor(data))

        # Start
        self.async_queue.start_async()

    def wait_all(self) -> Dict[int, Tuple[Dict, float]]:
        """
        Wait for all pending inferences

        Returns:
            {batch_id: (outputs, hw_time_ms)}
        """
        self.async_queue.wait_all()
        results = self._results.copy()
        self._results.clear()
        return results


def create_dummy_model(input_shape: Tuple[int, ...] = (1, 128),
                       output_shape: Tuple[int, ...] = (1, 128),
                       compute_ops: int = 1) -> 'ov.Model':
    """
    Create a dummy OpenVINO model for testing

    Args:
        input_shape: Input tensor shape
        output_shape: Output tensor shape (derived from matmul)
        compute_ops: Number of MatMul operations to chain

    Returns:
        OpenVINO Model
    """
    if not OPENVINO_AVAILABLE:
        raise RuntimeError("OpenVINO not available")

    # Create parameter
    param = ov.opset10.parameter(list(input_shape), np.float32, "input")

    current = param
    for i in range(compute_ops):
        # Create weight matrix
        in_features = input_shape[-1] if i == 0 else output_shape[-1]
        out_features = output_shape[-1]

        weight = ov.opset10.constant(
            np.random.randn(in_features, out_features).astype(np.float32) * 0.01
        )
        current = ov.opset10.matmul(current, weight, False, False)

        # Add ReLU for more realistic computation
        if i < compute_ops - 1:
            current = ov.opset10.relu(current)

    result = ov.opset10.result(current)
    return ov.Model([result], [param], "dummy_model")


def create_gather_scatter_model(num_nodes: int = 1000,
                                num_edges: int = 5000,
                                feature_dim: int = 128) -> 'ov.Model':
    """
    Create a model simulating GNN gather-scatter pattern

    This mimics the memory access patterns of GNN operations.

    Args:
        num_nodes: Number of nodes
        num_edges: Number of edges
        feature_dim: Feature dimension

    Returns:
        OpenVINO Model
    """
    if not OPENVINO_AVAILABLE:
        raise RuntimeError("OpenVINO not available")

    # Input: node features [num_nodes, feature_dim]
    x_param = ov.opset10.parameter([num_nodes, feature_dim], np.float32, "x")

    # Simple linear transform (simulating message passing)
    weight = ov.opset10.constant(
        np.random.randn(feature_dim, feature_dim).astype(np.float32) * 0.01
    )
    transformed = ov.opset10.matmul(x_param, weight, False, False)
    activated = ov.opset10.relu(transformed)

    result = ov.opset10.result(activated)
    return ov.Model([result], [x_param], "gnn_stage_model")


if __name__ == "__main__":
    if not OPENVINO_AVAILABLE:
        print("OpenVINO not available, skipping test")
        exit(0)

    # Test basic execution
    print("Testing StageExecutor...")

    core = Core()
    profiler = PipelineProfiler("Test")

    # Create dummy model
    model = create_dummy_model((1, 128), (1, 128), compute_ops=3)

    # Create executor
    executor = StageExecutor(
        core=core,
        model=model,
        device="CPU",
        stage_name="TestStage",
        profiler=profiler
    )

    # Run some batches
    for i in range(5):
        inputs = {"input": np.random.randn(1, 128).astype(np.float32)}
        outputs, hw_time = executor.run(inputs, batch_id=i)
        print(f"  Batch {i}: output shape = {outputs['output'].shape}, hw_time = {hw_time:.3f} ms")

    # Analyze
    profiler.export_chrome_trace("test_stage_executor.json")
    profiler.analyze_metrics()
