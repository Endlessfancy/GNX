"""
Pipeline Profiler - Trace Event Based Latency Measurement

Core profiling infrastructure for measuring:
- Hardware execution time (via OpenVINO PERF_COUNT)
- Wall clock time (Python timestamps)
- Data transfer overhead (Input/Reorder layer analysis)
- Pipeline bubbles (idle gaps between batches)

Output: Chrome Tracing compatible JSON for visualization in chrome://tracing or Perfetto
"""

import time
import json
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any
from pathlib import Path

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("WARNING: pandas not available, analyze_metrics() will be limited")


@dataclass
class TraceEvent:
    """
    Chrome Tracing compatible event format

    See: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
    """
    name: str           # Event name (e.g., "Stage1_Batch0")
    cat: str            # Category (e.g., "CPU", "GPU", "NPU", "Partition", "Merge")
    ph: str             # Phase: 'X' = complete event, 'B' = begin, 'E' = end
    ts: float           # Timestamp in microseconds (us)
    dur: float          # Duration in microseconds (us)
    pid: int            # Process ID (for grouping in visualization)
    tid: int            # Thread ID (for separate tracks per device)
    args: Dict = field(default_factory=dict)  # Additional metadata


class PipelineProfiler:
    """
    Central profiler for multi-stage pipeline execution

    Features:
    - Trace event logging with nanosecond precision
    - Chrome Tracing JSON export
    - Pandas-based metrics analysis
    - Hardware utilization calculation
    - Bubble (idle gap) detection
    """

    # Device to Track ID mapping for visualization
    DEVICE_TID_MAP = {
        "CPU": 1,
        "GPU": 2,
        "NPU": 3,
        "CPU_Partition": 10,
        "CPU_Merge": 11,
    }

    def __init__(self, name: str = "Pipeline"):
        """
        Initialize profiler

        Args:
            name: Pipeline name for identification
        """
        self.name = name
        self.events: List[TraceEvent] = []
        self.start_time_ref_ns = time.perf_counter_ns()
        self._batch_stage_times: Dict[int, Dict[str, float]] = {}  # batch_id -> {stage: duration_ms}

    def _get_tid(self, device: str, stream_id: int = 0) -> int:
        """
        Get thread ID for visualization track

        Args:
            device: Device name ("CPU", "GPU", "NPU")
            stream_id: Stream ID for data parallel (0, 1, 2...)

        Returns:
            Thread ID for Chrome Tracing
        """
        base_tid = self.DEVICE_TID_MAP.get(device, 0)
        return base_tid + stream_id * 100  # Stream 0: tid=2, Stream 1: tid=102

    def log_execution(self,
                      stage_name: str,
                      device: str,
                      batch_id: int,
                      wall_start_ns: int,
                      wall_end_ns: int,
                      hw_duration_ms: float,
                      stream_id: int = 0,
                      extra_args: Optional[Dict] = None):
        """
        Log a single inference execution

        Args:
            stage_name: Stage identifier (e.g., "Stage1_Gather")
            device: Device name ("CPU", "GPU", "NPU")
            batch_id: Batch/cycle identifier
            wall_start_ns: Wall clock start time (nanoseconds)
            wall_end_ns: Wall clock end time (nanoseconds)
            hw_duration_ms: Hardware execution time from profiling_info (milliseconds)
            stream_id: Stream ID for data parallel execution
            extra_args: Additional metadata to include
        """
        # Convert to microseconds for Chrome Tracing
        start_us = (wall_start_ns - self.start_time_ref_ns) / 1000.0
        duration_us = (wall_end_ns - wall_start_ns) / 1000.0
        wall_duration_ms = duration_us / 1000.0

        # Calculate overhead
        sw_overhead_ms = wall_duration_ms - hw_duration_ms

        args = {
            "batch_id": batch_id,
            "stream_id": stream_id,
            "hw_time_ms": round(hw_duration_ms, 4),
            "wall_time_ms": round(wall_duration_ms, 4),
            "sw_overhead_ms": round(sw_overhead_ms, 4),
        }
        if extra_args:
            args.update(extra_args)

        event = TraceEvent(
            name=f"{stage_name}_B{batch_id}",
            cat=device,
            ph="X",
            ts=start_us,
            dur=duration_us,
            pid=1,
            tid=self._get_tid(device, stream_id),
            args=args
        )
        self.events.append(event)

        # Track for cycle analysis
        if batch_id not in self._batch_stage_times:
            self._batch_stage_times[batch_id] = {}
        stage_key = f"{stage_name}_{device}"
        self._batch_stage_times[batch_id][stage_key] = wall_duration_ms

    def log_event(self,
                  event_name: str,
                  device: str,
                  batch_id: int,
                  start_ns: int,
                  end_ns: int,
                  event_type: str = "operation"):
        """
        Log a generic event (partition, merge, data transfer, etc.)

        Args:
            event_name: Event name (e.g., "Partition", "Merge")
            device: Device where event occurred
            batch_id: Batch/cycle identifier
            start_ns: Start time in nanoseconds
            end_ns: End time in nanoseconds
            event_type: Type category for the event
        """
        start_us = (start_ns - self.start_time_ref_ns) / 1000.0
        duration_us = (end_ns - start_ns) / 1000.0

        # Use special tid for CPU operations like Partition/Merge
        tid_key = f"{device}_{event_name}" if event_name in ["Partition", "Merge"] else device

        event = TraceEvent(
            name=f"{event_name}_B{batch_id}",
            cat=event_type,
            ph="X",
            ts=start_us,
            dur=duration_us,
            pid=1,
            tid=self._get_tid(tid_key) if tid_key in self.DEVICE_TID_MAP else self._get_tid(device),
            args={
                "batch_id": batch_id,
                "duration_ms": round(duration_us / 1000.0, 4),
            }
        )
        self.events.append(event)

    def log_data_transfer(self,
                          direction: str,
                          device: str,
                          batch_id: int,
                          start_ns: int,
                          end_ns: int,
                          data_size_bytes: int = 0):
        """
        Log data transfer event

        Args:
            direction: "H2D" (host to device) or "D2H" (device to host)
            device: Target device
            batch_id: Batch identifier
            start_ns: Start time
            end_ns: End time
            data_size_bytes: Data size in bytes
        """
        start_us = (start_ns - self.start_time_ref_ns) / 1000.0
        duration_us = (end_ns - start_ns) / 1000.0
        duration_ms = duration_us / 1000.0

        # Calculate bandwidth if data size provided
        bandwidth_gbps = 0.0
        if data_size_bytes > 0 and duration_ms > 0:
            bandwidth_gbps = (data_size_bytes / 1e9) / (duration_ms / 1000.0)

        event = TraceEvent(
            name=f"Transfer_{direction}_B{batch_id}",
            cat="DataTransfer",
            ph="X",
            ts=start_us,
            dur=duration_us,
            pid=1,
            tid=self._get_tid(device) + 50,  # Offset for transfer events
            args={
                "batch_id": batch_id,
                "direction": direction,
                "data_size_bytes": data_size_bytes,
                "duration_ms": round(duration_ms, 4),
                "bandwidth_gbps": round(bandwidth_gbps, 4),
            }
        )
        self.events.append(event)

    def export_chrome_trace(self, filename: str = "pipeline_trace.json") -> str:
        """
        Export events to Chrome Tracing JSON format

        Args:
            filename: Output filename

        Returns:
            Path to exported file
        """
        output_path = Path(filename)

        # Convert events to Chrome Tracing format
        chrome_events = []
        for event in self.events:
            chrome_events.append(asdict(event))

        # Add metadata events for better visualization
        metadata = [
            {"name": "process_name", "ph": "M", "pid": 1, "args": {"name": self.name}},
            {"name": "thread_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "CPU"}},
            {"name": "thread_name", "ph": "M", "pid": 1, "tid": 2, "args": {"name": "GPU"}},
            {"name": "thread_name", "ph": "M", "pid": 1, "tid": 3, "args": {"name": "NPU"}},
            {"name": "thread_name", "ph": "M", "pid": 1, "tid": 10, "args": {"name": "CPU_Partition"}},
            {"name": "thread_name", "ph": "M", "pid": 1, "tid": 11, "args": {"name": "CPU_Merge"}},
        ]

        all_events = metadata + chrome_events

        with open(output_path, 'w') as f:
            json.dump(all_events, f, indent=2)

        print(f"Trace exported to {output_path}")
        print(f"  Open in chrome://tracing or https://ui.perfetto.dev")
        return str(output_path)

    def get_cycle_latencies(self) -> Dict[int, float]:
        """
        Calculate cycle latency for each batch

        Cycle Latency = MAX(all stage times in that batch)
        This represents the throughput bottleneck.

        Returns:
            {batch_id: cycle_latency_ms}
        """
        cycle_latencies = {}
        for batch_id, stage_times in self._batch_stage_times.items():
            if stage_times:
                cycle_latencies[batch_id] = max(stage_times.values())
        return cycle_latencies

    def analyze_metrics(self) -> Optional[Dict]:
        """
        Analyze collected metrics using Pandas

        Returns:
            Dictionary with analysis results, or None if pandas unavailable
        """
        if not PANDAS_AVAILABLE:
            print("Pandas not available for detailed analysis")
            return self._analyze_metrics_basic()

        if not self.events:
            print("No events recorded")
            return None

        # Convert to DataFrame
        data = []
        for e in self.events:
            if e.ph != "X":  # Skip metadata events
                continue
            row = {
                "Name": e.name,
                "Stage": e.name.rsplit('_B', 1)[0] if '_B' in e.name else e.name,
                "Device": e.cat,
                "Batch": e.args.get('batch_id', -1),
                "Stream": e.args.get('stream_id', 0),
                "Start_us": e.ts,
                "Duration_us": e.dur,
                "Start_ms": e.ts / 1000.0,
                "End_ms": (e.ts + e.dur) / 1000.0,
                "Duration_Wall_ms": e.dur / 1000.0,
                "Duration_HW_ms": e.args.get('hw_time_ms', e.dur / 1000.0),
                "Overhead_ms": e.args.get('sw_overhead_ms', 0),
            }
            data.append(row)

        df = pd.DataFrame(data)

        if df.empty:
            print("No execution events recorded")
            return None

        results = {}

        print("\n" + "=" * 70)
        print("Pipeline Performance Summary")
        print("=" * 70)

        # 1. Total Runtime
        total_time_ms = df['End_ms'].max() - df['Start_ms'].min()
        results['total_runtime_ms'] = total_time_ms
        print(f"\nTotal Pipeline Runtime: {total_time_ms:.2f} ms")

        # 2. Per-Device Utilization
        print("\n--- Device Utilization ---")
        device_stats = {}
        for device in df['Device'].unique():
            if device in ['operation', 'DataTransfer']:
                continue
            d_df = df[df['Device'] == device]

            hw_time_total = d_df['Duration_HW_ms'].sum()
            wall_time_total = d_df['Duration_Wall_ms'].sum()

            hw_util = (hw_time_total / total_time_ms) * 100 if total_time_ms > 0 else 0
            wall_util = (wall_time_total / total_time_ms) * 100 if total_time_ms > 0 else 0
            avg_overhead = d_df['Overhead_ms'].mean()

            device_stats[device] = {
                'hw_util_pct': hw_util,
                'wall_util_pct': wall_util,
                'avg_overhead_ms': avg_overhead,
                'total_hw_time_ms': hw_time_total,
                'total_wall_time_ms': wall_time_total,
            }

            print(f"[{device}] HW Util: {hw_util:.1f}% | Wall Util: {wall_util:.1f}% | Avg Overhead: {avg_overhead:.3f} ms")

        results['device_stats'] = device_stats

        # 3. Pipeline Bubbles (Idle Gaps)
        print("\n--- Pipeline Bubbles (Idle Gaps) ---")
        bubble_stats = {}
        for device in df['Device'].unique():
            if device in ['operation', 'DataTransfer']:
                continue
            d_df = df[df['Device'] == device].sort_values('Start_ms')

            if len(d_df) > 1:
                d_df = d_df.copy()
                d_df['prev_end'] = d_df['End_ms'].shift(1)
                d_df['bubble_ms'] = d_df['Start_ms'] - d_df['prev_end']

                bubbles = d_df[d_df['bubble_ms'] > 0]['bubble_ms']
                if len(bubbles) > 0:
                    avg_bubble = bubbles.mean()
                    max_bubble = bubbles.max()
                    total_bubble = bubbles.sum()
                else:
                    avg_bubble = max_bubble = total_bubble = 0

                bubble_stats[device] = {
                    'avg_bubble_ms': avg_bubble,
                    'max_bubble_ms': max_bubble,
                    'total_bubble_ms': total_bubble,
                }
                print(f"[{device}] Avg Gap: {avg_bubble:.3f} ms | Max Gap: {max_bubble:.3f} ms | Total Idle: {total_bubble:.3f} ms")

        results['bubble_stats'] = bubble_stats

        # 4. Batch/Cycle Latency
        print("\n--- Batch Latency (End-to-End) ---")
        batch_df = df.groupby('Batch').agg(
            Pipeline_Start=('Start_ms', 'min'),
            Pipeline_End=('End_ms', 'max')
        )
        batch_df['Latency_ms'] = batch_df['Pipeline_End'] - batch_df['Pipeline_Start']

        avg_latency = batch_df['Latency_ms'].mean()
        min_latency = batch_df['Latency_ms'].min()
        max_latency = batch_df['Latency_ms'].max()

        results['batch_latency'] = {
            'avg_ms': avg_latency,
            'min_ms': min_latency,
            'max_ms': max_latency,
        }
        print(f"Avg Batch Latency: {avg_latency:.2f} ms (min: {min_latency:.2f}, max: {max_latency:.2f})")

        # 5. Cycle Latency (bottleneck-based)
        cycle_latencies = self.get_cycle_latencies()
        if cycle_latencies:
            avg_cycle = np.mean(list(cycle_latencies.values()))
            results['cycle_latency_avg_ms'] = avg_cycle
            print(f"Avg Cycle Latency (bottleneck): {avg_cycle:.2f} ms")

        # 6. Per-Stage Statistics
        print("\n--- Per-Stage Statistics ---")
        stage_stats = df.groupby(['Stage', 'Device']).agg(
            Count=('Duration_Wall_ms', 'count'),
            Avg_Wall_ms=('Duration_Wall_ms', 'mean'),
            Avg_HW_ms=('Duration_HW_ms', 'mean'),
            Std_ms=('Duration_Wall_ms', 'std'),
        ).round(4)
        print(stage_stats.to_string())
        results['stage_stats'] = stage_stats.to_dict()

        print("\n" + "=" * 70)

        return results

    def _analyze_metrics_basic(self) -> Dict:
        """Basic analysis without pandas"""
        if not self.events:
            return {}

        results = {'events_count': len(self.events)}

        # Calculate basic stats
        exec_events = [e for e in self.events if e.ph == "X"]
        if exec_events:
            total_duration = sum(e.dur for e in exec_events) / 1000.0  # ms
            results['total_duration_ms'] = total_duration

        cycle_latencies = self.get_cycle_latencies()
        if cycle_latencies:
            results['avg_cycle_latency_ms'] = np.mean(list(cycle_latencies.values()))

        return results

    def reset(self):
        """Clear all recorded events"""
        self.events.clear()
        self._batch_stage_times.clear()
        self.start_time_ref_ns = time.perf_counter_ns()
        print("Profiler reset")


# Convenience function for timing blocks
class TimingContext:
    """Context manager for timing code blocks"""

    def __init__(self):
        self.start_ns = 0
        self.end_ns = 0
        self.duration_ms = 0

    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        self.end_ns = time.perf_counter_ns()
        self.duration_ms = (self.end_ns - self.start_ns) / 1e6


if __name__ == "__main__":
    # Simple test
    profiler = PipelineProfiler("TestPipeline")

    # Simulate some events
    base = time.perf_counter_ns()
    for i in range(5):
        # Simulate Stage 1 on CPU
        start = time.perf_counter_ns()
        time.sleep(0.001)  # 1ms
        end = time.perf_counter_ns()
        profiler.log_execution("Stage1", "CPU", i, start, end, 0.8)

        # Simulate Stage 2 on GPU
        start = time.perf_counter_ns()
        time.sleep(0.002)  # 2ms
        end = time.perf_counter_ns()
        profiler.log_execution("Stage2", "GPU", i, start, end, 1.5)

    # Export and analyze
    profiler.export_chrome_trace("test_trace.json")
    profiler.analyze_metrics()
