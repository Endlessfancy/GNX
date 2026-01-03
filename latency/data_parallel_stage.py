"""
Data Parallel Stage - Multi-Device Parallel Execution with Timing

Handles data parallelism across multiple devices (GPU + NPU, etc.):
1. Partition: Split input data across devices
2. Parallel Execution: Run inference on all devices concurrently
3. Merge: Combine results from all devices

Timing:
- partition_time: Time to split/prepare data
- device_time: Per-device execution time
- sync_merge_time: Wait and merge time
- stage_total_time: partition + MAX(device_times) + merge (critical path)
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future

from profiler import PipelineProfiler, TimingContext
from stage_executor import StageExecutor

try:
    import openvino as ov
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False


class DataParallelStage:
    """
    Data Parallel Stage executor for multi-device inference

    Manages multiple StageExecutor instances and coordinates:
    - Data partitioning across devices
    - Parallel async dispatch
    - Result synchronization and merging
    """

    def __init__(self,
                 name: str,
                 executors: List[StageExecutor],
                 ratios: List[float],
                 profiler: PipelineProfiler,
                 partition_dim: int = 0):
        """
        Initialize data parallel stage

        Args:
            name: Stage name for logging
            executors: List of StageExecutor instances (one per device)
            ratios: Data split ratios for each device (must sum to 1.0)
            profiler: PipelineProfiler instance
            partition_dim: Dimension along which to split data (default 0 = batch/nodes)
        """
        self.name = name
        self.executors = executors
        self.ratios = ratios
        self.profiler = profiler
        self.partition_dim = partition_dim

        # Validate ratios
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
        if len(ratios) != len(executors):
            raise ValueError(f"Number of ratios ({len(ratios)}) must match executors ({len(executors)})")

        self.num_devices = len(executors)
        self.device_names = [exc.device for exc in executors]

        print(f"  DataParallelStage '{name}' initialized:")
        print(f"    Devices: {self.device_names}")
        print(f"    Ratios: {ratios}")

    def run(self,
            inputs: Dict[str, np.ndarray],
            batch_id: int) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Execute data parallel inference with full timing

        Args:
            inputs: Input tensors to partition
            batch_id: Batch/cycle identifier

        Returns:
            (merged_outputs, timing_info): Merged results and timing breakdown
        """
        timing_info = {}

        # ============================================================
        # Phase 1: Partition Data
        # ============================================================
        partition_start_ns = time.perf_counter_ns()

        partitioned_inputs = self._partition_data(inputs)

        partition_end_ns = time.perf_counter_ns()
        partition_time_ms = (partition_end_ns - partition_start_ns) / 1e6
        timing_info['partition_time_ms'] = partition_time_ms

        # Log partition event
        self.profiler.log_event(
            event_name="Partition",
            device="CPU",
            batch_id=batch_id,
            start_ns=partition_start_ns,
            end_ns=partition_end_ns,
            event_type="DataParallel"
        )

        # ============================================================
        # Phase 2: Parallel Dispatch (start_async on all devices)
        # ============================================================
        dispatch_start_ns = time.perf_counter_ns()

        # Start all devices asynchronously
        start_times = []
        for i, executor in enumerate(self.executors):
            start_ns = executor.start_async(partitioned_inputs[i], batch_id)
            start_times.append(start_ns)

        dispatch_end_ns = time.perf_counter_ns()
        dispatch_time_ms = (dispatch_end_ns - dispatch_start_ns) / 1e6
        timing_info['dispatch_time_ms'] = dispatch_time_ms

        # ============================================================
        # Phase 3: Wait for all devices
        # ============================================================
        wait_start_ns = time.perf_counter_ns()

        device_outputs = []
        device_hw_times = []
        device_wall_times = []

        for i, executor in enumerate(self.executors):
            # Wait individually and record per-device timing
            wait_device_start = time.perf_counter_ns()
            outputs, hw_time = executor.wait(start_times[i], batch_id)
            wait_device_end = time.perf_counter_ns()

            device_outputs.append(outputs)
            device_hw_times.append(hw_time)
            device_wall_times.append((wait_device_end - start_times[i]) / 1e6)

        wait_end_ns = time.perf_counter_ns()

        timing_info['device_hw_times_ms'] = device_hw_times
        timing_info['device_wall_times_ms'] = device_wall_times
        timing_info['max_device_wall_time_ms'] = max(device_wall_times)

        # ============================================================
        # Phase 4: Merge Results
        # ============================================================
        merge_start_ns = time.perf_counter_ns()

        merged_outputs = self._merge_data(device_outputs)

        merge_end_ns = time.perf_counter_ns()
        merge_time_ms = (merge_end_ns - merge_start_ns) / 1e6
        timing_info['merge_time_ms'] = merge_time_ms

        # Log merge event
        self.profiler.log_event(
            event_name="Merge",
            device="CPU",
            batch_id=batch_id,
            start_ns=merge_start_ns,
            end_ns=merge_end_ns,
            event_type="DataParallel"
        )

        # ============================================================
        # Calculate Stage Total (Critical Path)
        # ============================================================
        # T_stage = T_partition + MAX(device_wall_times) + T_merge
        stage_total_ms = (
            partition_time_ms +
            max(device_wall_times) +
            merge_time_ms
        )
        timing_info['stage_total_ms'] = stage_total_ms

        # Also track end-to-end wall time
        timing_info['end_to_end_ms'] = (merge_end_ns - partition_start_ns) / 1e6

        return merged_outputs, timing_info

    def _partition_data(self,
                        inputs: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """
        Partition input data according to ratios

        Args:
            inputs: Input tensors to partition

        Returns:
            List of input dicts for each device
        """
        partitioned = [{} for _ in range(self.num_devices)]

        for name, tensor in inputs.items():
            total_size = tensor.shape[self.partition_dim]

            # Calculate split points
            split_points = []
            cumsum = 0
            for ratio in self.ratios[:-1]:  # Skip last ratio
                cumsum += int(total_size * ratio)
                split_points.append(cumsum)

            # Split tensor
            splits = np.split(tensor, split_points, axis=self.partition_dim)

            for i, split_tensor in enumerate(splits):
                partitioned[i][name] = split_tensor

        return partitioned

    def _merge_data(self,
                    device_outputs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Merge outputs from all devices

        Args:
            device_outputs: List of output dicts from each device

        Returns:
            Merged output dict
        """
        merged = {}

        # Get all output names from first device
        output_names = device_outputs[0].keys()

        for name in output_names:
            tensors = [outputs[name] for outputs in device_outputs]
            merged[name] = np.concatenate(tensors, axis=self.partition_dim)

        return merged


class DataParallelStageAsync:
    """
    Async version using ThreadPoolExecutor for true parallel execution

    This version runs each device inference in a separate thread,
    allowing better overlap of computation.
    """

    def __init__(self,
                 name: str,
                 executors: List[StageExecutor],
                 ratios: List[float],
                 profiler: PipelineProfiler,
                 partition_dim: int = 0,
                 max_workers: int = None):
        """
        Initialize async data parallel stage

        Args:
            name: Stage name
            executors: List of executors
            ratios: Split ratios
            profiler: Pipeline profiler
            partition_dim: Partition dimension
            max_workers: Max thread pool workers (default = num devices)
        """
        self.name = name
        self.executors = executors
        self.ratios = ratios
        self.profiler = profiler
        self.partition_dim = partition_dim
        self.num_devices = len(executors)
        self.device_names = [exc.device for exc in executors]

        # Thread pool for parallel execution
        self.executor_pool = ThreadPoolExecutor(
            max_workers=max_workers or self.num_devices
        )

        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0")

        print(f"  DataParallelStageAsync '{name}' initialized:")
        print(f"    Devices: {self.device_names}")
        print(f"    Ratios: {ratios}")

    def run(self,
            inputs: Dict[str, np.ndarray],
            batch_id: int) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Execute with true thread-parallel device inference
        """
        timing_info = {}

        # Phase 1: Partition
        partition_start_ns = time.perf_counter_ns()
        partitioned_inputs = self._partition_data(inputs)
        partition_end_ns = time.perf_counter_ns()
        timing_info['partition_time_ms'] = (partition_end_ns - partition_start_ns) / 1e6

        self.profiler.log_event(
            "Partition", "CPU", batch_id,
            partition_start_ns, partition_end_ns, "DataParallel"
        )

        # Phase 2: Submit all to thread pool
        def run_device(idx: int) -> Tuple[int, Dict, float, float]:
            """Run single device inference"""
            start = time.perf_counter_ns()
            outputs, hw_time = self.executors[idx].run(
                partitioned_inputs[idx], batch_id
            )
            end = time.perf_counter_ns()
            wall_time = (end - start) / 1e6
            return idx, outputs, hw_time, wall_time

        parallel_start_ns = time.perf_counter_ns()

        futures: List[Future] = []
        for i in range(self.num_devices):
            future = self.executor_pool.submit(run_device, i)
            futures.append(future)

        # Phase 3: Collect results
        device_outputs = [None] * self.num_devices
        device_hw_times = [0.0] * self.num_devices
        device_wall_times = [0.0] * self.num_devices

        for future in futures:
            idx, outputs, hw_time, wall_time = future.result()
            device_outputs[idx] = outputs
            device_hw_times[idx] = hw_time
            device_wall_times[idx] = wall_time

        parallel_end_ns = time.perf_counter_ns()

        timing_info['device_hw_times_ms'] = device_hw_times
        timing_info['device_wall_times_ms'] = device_wall_times
        timing_info['max_device_wall_time_ms'] = max(device_wall_times)
        timing_info['parallel_execution_ms'] = (parallel_end_ns - parallel_start_ns) / 1e6

        # Phase 4: Merge
        merge_start_ns = time.perf_counter_ns()
        merged_outputs = self._merge_data(device_outputs)
        merge_end_ns = time.perf_counter_ns()
        timing_info['merge_time_ms'] = (merge_end_ns - merge_start_ns) / 1e6

        self.profiler.log_event(
            "Merge", "CPU", batch_id,
            merge_start_ns, merge_end_ns, "DataParallel"
        )

        # Critical path timing
        timing_info['stage_total_ms'] = (
            timing_info['partition_time_ms'] +
            timing_info['max_device_wall_time_ms'] +
            timing_info['merge_time_ms']
        )
        timing_info['end_to_end_ms'] = (merge_end_ns - partition_start_ns) / 1e6

        return merged_outputs, timing_info

    def _partition_data(self, inputs: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """Partition input data according to ratios"""
        partitioned = [{} for _ in range(self.num_devices)]

        for name, tensor in inputs.items():
            total_size = tensor.shape[self.partition_dim]
            split_points = []
            cumsum = 0
            for ratio in self.ratios[:-1]:
                cumsum += int(total_size * ratio)
                split_points.append(cumsum)

            splits = np.split(tensor, split_points, axis=self.partition_dim)
            for i, split_tensor in enumerate(splits):
                partitioned[i][name] = split_tensor

        return partitioned

    def _merge_data(self, device_outputs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Merge outputs from all devices"""
        merged = {}
        output_names = device_outputs[0].keys()
        for name in output_names:
            tensors = [outputs[name] for outputs in device_outputs]
            merged[name] = np.concatenate(tensors, axis=self.partition_dim)
        return merged

    def shutdown(self):
        """Shutdown thread pool"""
        self.executor_pool.shutdown(wait=True)


if __name__ == "__main__":
    if not OPENVINO_AVAILABLE:
        print("OpenVINO not available")
        exit(0)

    from stage_executor import create_dummy_model

    print("Testing DataParallelStage...")

    core = Core()
    profiler = PipelineProfiler("DataParallelTest")

    # Create model
    model = create_dummy_model((100, 128), (100, 128), compute_ops=2)

    # Create executors for different "devices" (both CPU for testing)
    exec1 = StageExecutor(core, model, "CPU", "DP_Stage", profiler, stream_id=0)
    exec2 = StageExecutor(core, model, "CPU", "DP_Stage", profiler, stream_id=1)

    # Create data parallel stage
    dp_stage = DataParallelStage(
        name="TestDP",
        executors=[exec1, exec2],
        ratios=[0.5, 0.5],
        profiler=profiler
    )

    # Run batches
    for batch_id in range(5):
        inputs = {"input": np.random.randn(100, 128).astype(np.float32)}
        outputs, timing = dp_stage.run(inputs, batch_id)

        print(f"\nBatch {batch_id}:")
        print(f"  Partition: {timing['partition_time_ms']:.3f} ms")
        print(f"  Device times: {timing['device_wall_times_ms']}")
        print(f"  Merge: {timing['merge_time_ms']:.3f} ms")
        print(f"  Stage total (critical path): {timing['stage_total_ms']:.3f} ms")
        print(f"  Output shape: {outputs['output'].shape}")

    # Export trace
    profiler.export_chrome_trace("test_data_parallel.json")
    profiler.analyze_metrics()
