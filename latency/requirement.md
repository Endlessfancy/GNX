Role: You are a Senior AI Systems Architect specializing in High-Performance Computing (HPC) and OpenVINO optimization on Intel Heterogeneous Hardware (CPU/GPU/NPU).

Task: Design and implement a robust Python Profiling Framework for a complex Multi-Stage GNN Inference Pipeline running on an Intel AI PC (Meteor Lake).

Context: I have a streaming pipeline with multiple stages (e.g., Graph Prep -> Embedding -> Classification).

Data Flow: Streaming input, where different data flows through stages in cycles.

Hardware: It runs on a unified memory architecture (UMA) where CPU, GPU, and NPU are used concurrently.

Parallelism: Some stages utilize Data Parallelism (e.g., multiple concurrent streams on the same or different PUs).

Problem: I need to precisely measure latency, hardware utilization, and pipeline bubbles (idle gaps) to identify bottlenecks caused by memory bandwidth contention or scheduling overhead.

Requirements for the Code:

Architecture:

Use a Trace Event based approach (logging timestamp, duration, device, stage, batch_id for every inference).

Implement a central PipelineProfiler class to handle logging and exporting.

Implement a modular StageExecutor class that wraps openvino.runtime.AsyncInferQueue or InferRequest.

Instrumentation (The "How"):

Hardware Time: Must use config={"PERF_COUNT": "YES"} during compilation and read request.profiling_info to get the pure hardware execution time (excluding Python/Driver overhead).

Wall Clock Time: Record precise start/end timestamps (ns) around start_async() and wait() calls.

Data Transfer Check: In the profiling logic, check for "Input/Reorder" layer times to estimate data marshalling overhead.

Visualization & Analysis:

Chrome Tracing Export: The profiler must export a .json file compatible with chrome://tracing (Perfetto). Use distinct tid (Thread IDs) for different devices/streams to visualize parallelism and overlaps.

Pandas Analysis: Include a method that converts logs to a Pandas DataFrame to calculate:

HW Utilization %: (Total HW Time / Total Wall Time).

Bubbles: Average time gap between the end of Batch N and start of Batch N+1 on the same device.

Concurrency Penalty: Display the difference between Wall Time and Pure HW Time.

Simulation Mode:

Since the real model isn't provided, include a method to generate a Dummy OpenVINO Model (e.g., MatMul) to make the code runnable immediately.

Simulate a multi-stage loop where different stages run asynchronously.

Output: Provide the complete, commented, and runnable Python code structure.Role: You are a Senior AI Systems Architect specializing in High-Performance Computing (HPC) and OpenVINO optimization on Intel Heterogeneous Hardware (CPU/GPU/NPU).

Task: Design and implement a robust Python Profiling Framework for a complex Multi-Stage GNN Inference Pipeline running on an Intel AI PC (Meteor Lake).

Context: I have a streaming pipeline with multiple stages (e.g., Graph Prep -> Embedding -> Classification).

Data Flow: Streaming input, where different data flows through stages in cycles.

Hardware: It runs on a unified memory architecture (UMA) where CPU, GPU, and NPU are used concurrently.

Parallelism: Some stages utilize Data Parallelism (e.g., multiple concurrent streams on the same or different PUs).

Problem: I need to precisely measure latency, hardware utilization, and pipeline bubbles (idle gaps) to identify bottlenecks caused by memory bandwidth contention or scheduling overhead.

Requirements for the Code:

Architecture:

Use a Trace Event based approach (logging timestamp, duration, device, stage, batch_id for every inference).

Implement a central PipelineProfiler class to handle logging and exporting.

Implement a modular StageExecutor class that wraps openvino.runtime.AsyncInferQueue or InferRequest.

Instrumentation (The "How"):

Hardware Time: Must use config={"PERF_COUNT": "YES"} during compilation and read request.profiling_info to get the pure hardware execution time (excluding Python/Driver overhead).

Wall Clock Time: Record precise start/end timestamps (ns) around start_async() and wait() calls.

Data Transfer Check: In the profiling logic, check for "Input/Reorder" layer times to estimate data marshalling overhead.

Visualization & Analysis:

Chrome Tracing Export: The profiler must export a .json file compatible with chrome://tracing (Perfetto). Use distinct tid (Thread IDs) for different devices/streams to visualize parallelism and overlaps.

Pandas Analysis: Include a method that converts logs to a Pandas DataFrame to calculate:

HW Utilization %: (Total HW Time / Total Wall Time).

Bubbles: Average time gap between the end of Batch N and start of Batch N+1 on the same device.

Concurrency Penalty: Display the difference between Wall Time and Pure HW Time.

Simulation Mode:

Since the real model isn't provided, include a method to generate a Dummy OpenVINO Model (e.g., MatMul) to make the code runnable immediately.

Simulate a multi-stage loop where different stages run asynchronously.

Output: Provide the complete, commented, and runnable Python code structure.




**Role:** You are an Expert AI Systems Architect specializing in Heterogeneous Computing (CPU/GPU/NPU) and Pipeline Optimization.

**Task:** Refine and extend the previous OpenVINO Python Profiling Framework to support **Data Parallel (DP) Stages** and **Pipeline Cycle Analysis**.

**Context Update:**
My GNN inference pipeline consists of multiple stages.
1.  **Complex Data Flow:** Some stages are **Data Parallel (DP)**. A single logical stage (e.g., "Graph Execution") is handled by **two or more devices** (e.g., GPU + NPU) simultaneously.
2.  **DP Overhead:** For a DP stage, the CPU must first **partition** the graph data (e.g., split node features 50/50) and later **merge** the results from both devices.
3.  **Pipeline Cycle:** The input data flows through the pipeline in cycles. The "Time Consumption" of a specific cycle is defined by the **Bottleneck Stage** (the stage that took the longest time).

**New Requirements:**

1.  **Implement a `DataParallelStage` Class:**
    This class should manage multiple `StageExecutor` instances (devices). It must specifically measure and log:
    * **`partition_time`**: Time taken to slice/prepare input data for multiple devices.
    * **`device_time`**: Async execution time for each device.
    * **`sync_merge_time`**: Time taken to wait for all devices and concatenate/merge their outputs.
    * **`stage_total_time`**: Calculated as `partition_time + MAX(device_wall_times) + sync_merge_time`.

2.  **Pipeline Cycle Statistic:**
    * For each input batch (Cycle), calculate the duration of *each* stage.
    * Define the **Cycle Latency** as `MAX(Stage_1_Time, Stage_2_Time, ...)`. This represents the pipeline throughput bottleneck.

3.  **Visualization Update:**
    * In the Chrome Trace (`.json`), the "Partition" and "Merge" operations must be distinct events on the CPU timeline, separate from the GPU/NPU execution bars.

**Code Framework Suggestion:**
Please use a structure similar to this (pseudo-code) for the DP logic:

```python
class DataParallelStage:
    def __init__(self, name, executors):
        self.name = name
        self.executors = executors # List of SingleDeviceExecutors (GPU, NPU)

    def run(self, input_data, batch_id, profiler):
        # 1. Partition Data
        t0 = time.perf_counter()
        # Simulate slicing data (e.g., input_data[:mid], input_data[mid:])
        inputs = self._partition_logic(input_data) 
        t1 = time.perf_counter()
        partition_time = (t1 - t0) * 1000
        
        # Log Partition Event (CPU)
        profiler.log_event("Partition", "CPU", batch_id, t0, t1)

        # 2. Parallel Dispatch
        # Start all devices async
        t_start_run = time.perf_counter()
        for i, exc in enumerate(self.executors):
            exc.start_async(inputs[i])
            
        # 3. Wait & Merge
        results = []
        for exc in self.executors:
            # Wait for individual device
            res = exc.wait() 
            results.append(res)
        
        # 4. Merge Data
        t2 = time.perf_counter()
        final_output = self._merge_logic(results)
        t3 = time.perf_counter()
        merge_time = (t3 - t2) * 1000
        
        # Log Merge Event (CPU)
        profiler.log_event("Merge", "CPU", batch_id, t2, t3)
        
        # 5. Calculate Stage Total (Critical Path)
        # Note: Actual logic should align partition -> max(run) -> merge
        return final_output
Output: Provide the complete, updated Python script. Ensure the Dummy Data generation supports splitting so the code runs without errors.


---

### 这个 Prompt 的核心改进点：

1.  **明确了 `DataParallelStage` 的三段式结构：**
    * **Pre-process (Partition):** 显式要求测量切分时间。
    * **Parallel Execution:** 多设备并发。
    * **Post-process (Merge):** 显式要求测量合并时间（通常涉及 `numpy.concatenate` 或 `torch.cat`，在 CPU 上也很耗时）。

2.  **重新定义了 Stage 时间计算公式：**
    * 不再是简单的 `req.wait()`。
    * 而是 $T_{stage} = T_{partition} + \text{CriticalPath}(Devices) + T_{merge}$。

3.  **明确了 Cycle 的定义：**
    * 告诉 Claude 你关注的是 **Throughput (吞吐率)** 瓶颈，因此 Cycle Time = Max(所有 Stage 时间)，而不是 Sum（End-to-End Latency）。

4.  **Trace 可视化要求：**
    * 要求在 Chrome Tracing 的 CPU 这一行里，必须能看到独立的小方块代表 "Partition" 和 "Merge"，这样你能直观地看到它们有没有阻塞流水线。



    

完整的测试逻辑代码
你可以直接运行这段代码。它模拟了一个 3 阶段流水线（CPU -> GPU -> NPU），并生成分析报告。

Python

import openvino.runtime as ov
import numpy as np
import time
import json
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict

# ==========================================
# 1. 基础架构：事件追踪器 (Trace Logger)
# ==========================================

@dataclass
class TraceEvent:
    name: str          # Stage 名称 (e.g., "Stage1_Preprocess")
    cat: str           # 类别 (e.g., "CPU", "GPU")
    ph: str            # Phase: 'X' 代表区间事件
    ts: int            # 时间戳 (微秒)
    dur: int           # 持续时间 (微秒)
    pid: int           # 用于可视化分组 (Process ID)
    tid: int           # 用于可视化分组 (Thread ID)
    args: Dict         # 额外元数据 (Hardware time, batch_id)

class PipelineProfiler:
    def __init__(self):
        self.events = []
        self.start_time_ref = time.perf_counter_ns()
    
    def log_execution(self, stage_name, device, batch_id, 
                      wall_start_ns, wall_end_ns, hw_duration_ms):
        """
        记录一次推理的完整生命周期
        """
        # 转换为微秒 (us) 用于 Chrome Tracing
        start_us = (wall_start_ns - self.start_time_ref) / 1000
        duration_us = (wall_end_ns - wall_start_ns) / 1000
        
        event = TraceEvent(
            name=f"{stage_name}_Batch{batch_id}",
            cat=device,
            ph="X",
            ts=start_us,
            dur=duration_us,
            pid=1,
            tid=self._get_tid_for_device(device),
            args={
                "batch_id": batch_id,
                "hw_time_ms": hw_duration_ms, # 硬件纯计算时间
                "sw_overhead_ms": (duration_us/1000) - hw_duration_ms # 软件/驱动开销
            }
        )
        self.events.append(event)

    def _get_tid_for_device(self, device):
        # 给不同设备分配不同的轨道 ID，方便在图表中分开显示
        mapping = {"CPU": 1, "GPU": 2, "NPU": 3}
        return mapping.get(device, 0)

    def export_chrome_trace(self, filename="pipeline_trace.json"):
        """导出为 Chrome Tracing 格式，可在 chrome://tracing 打开"""
        chrome_data = [asdict(e) for e in self.events]
        with open(filename, 'w') as f:
            json.dump(chrome_data, f)
        print(f"✅ Trace exported to {filename}. Open in chrome://tracing or ui.perfetto.dev")

    def analyze_metrics(self):
        """使用 Pandas 自动计算延迟、利用率和气泡"""
        data = []
        for e in self.events:
            row = {
                "Stage": e.name.split('_')[0],
                "Device": e.cat,
                "Batch": e.args['batch_id'],
                "Start_ms": e.ts / 1000,
                "End_ms": (e.ts + e.dur) / 1000,
                "Duration_Wall_ms": e.dur / 1000,
                "Duration_HW_ms": e.args['hw_time_ms'],
                "Overhead_ms": e.args['sw_overhead_ms']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        if df.empty:
            print("No data recorded.")
            return

        print("\n=== 📊 Pipeline Performance Summary ===")
        
        # 1. 计算每个设备的利用率 (Utilization)
        total_time = df['End_ms'].max() - df['Start_ms'].min()
        print(f"Total Pipeline Runtime: {total_time:.2f} ms")
        
        for device in df['Device'].unique():
            d_df = df[df['Device'] == device]
            # 简单的利用率计算：所有任务硬件时间之和 / 总挂钟时间
            # 注意：如果同一设备并行跑多个任务，这里可能需要更复杂的区间合并逻辑，但对于单流是准确的
            hw_util = d_df['Duration_HW_ms'].sum() / total_time * 100
            wall_util = d_df['Duration_Wall_ms'].sum() / total_time * 100
            print(f"[{device}] HW Utilization: {hw_util:.1f}% | Wall Utilization (busy): {wall_util:.1f}%")

        # 2. 计算 Pipeline Bubbles (空闲时间)
        print("\n--- Bubbles (Idle Gaps) ---")
        for device in df['Device'].unique():
            d_df = df[df['Device'] == device].sort_values('Start_ms')
            # 计算当前任务开始时间 - 上一个任务结束时间
            d_df['prev_end'] = d_df['End_ms'].shift(1)
            d_df['bubble'] = d_df['Start_ms'] - d_df['prev_end']
            avg_bubble = d_df[d_df['bubble'] > 0]['bubble'].mean()
            print(f"[{device}] Avg Gap betw. tasks: {avg_bubble:.2f} ms")

        # 3. 计算端到端延迟 (Latency)
        # 假设 Stage A 是入口，Stage C 是出口
        # 找到每个 Batch 的最早开始和最晚结束
        batch_stats = df.groupby('Batch').agg(
            Pipeline_Start=('Start_ms', 'min'),
            Pipeline_End=('End_ms', 'max')
        )
        batch_stats['Latency'] = batch_stats['Pipeline_End'] - batch_stats['Pipeline_Start']
        print(f"\nAvg Batch Latency: {batch_stats['Latency'].mean():.2f} ms")

# ==========================================
# 2. 核心逻辑：带打点的执行器 (Stage Executor)
# ==========================================

class StageExecutor:
    def __init__(self, core, model_path, device, stage_name, profiler):
        self.profiler = profiler
        self.device = device
        self.stage_name = stage_name
        
        # 开启 PERF_COUNT 获取硬件时间
        print(f"Loading {stage_name} on {device}...")
        # 这里的 model_path 可以换成你的 get_dummy_model()
        # model = core.read_model(model_path) 
        # 为了演示，创建一个 Dummy Model
        model = self._create_dummy_model(core)
        
        self.compiled_model = core.compile_model(model, device, config={"PERF_COUNT": "YES"})
        self.request = self.compiled_model.create_infer_request()
        self.input_tensor = self.request.input_tensors[0]

    def _create_dummy_model(self, core):
        # 创建一个简单的 MatMul 模型用于演示
        param = ov.opset10.parameter([1, 128], np.float32, "input")
        const = ov.opset10.constant(np.random.rand(128, 128).astype(np.float32))
        matmul = ov.opset10.matmul(param, const, False, False)
        res = ov.opset10.result(matmul)
        return ov.Model([res], [param], "dummy_matmul")

    def run(self, data, batch_id):
        """
        执行推理并记录所有时间指标
        """
        # 1. 记录墙上开始时间
        start_ns = time.perf_counter_ns()
        
        # 2. 异步发射
        self.request.start_async({0: data})
        
        # 3. 同步等待 (在 Pipeline 逻辑中，你可能会把 wait 放在后面，这里为了简化演示放在这里)
        # 如果你的 Pipeline 是完全异步的（fire-and-forget），你需要把 wait 拆分出去
        self.request.wait()
        
        # 4. 记录墙上结束时间
        end_ns = time.perf_counter_ns()
        
        # 5. 获取硬件真实耗时
        hw_time_ms = 0.0
        for info in self.request.profiling_info:
            if info.status == list(ov.ProfilingInfo.Status)[1]: # EXECUTED
                hw_time_ms += (info.real_time / 1000.0) # us -> ms

        # 6. 写入追踪器
        self.profiler.log_execution(
            self.stage_name, self.device, batch_id, start_ns, end_ns, hw_time_ms
        )
        
        return self.request.get_output_tensor(0).data

# ==========================================
# 3. 模拟复杂 Pipeline 逻辑
# ==========================================

def run_pipeline_simulation():
    core = ov.Core()
    profiler = PipelineProfiler()
    
    # 定义 3 个 Stage，模拟不同的 PU
    # 注意：在真实代码中请换成你的实际模型路径
    stage1 = StageExecutor(core, "dummy", "CPU", "Stage1_GraphPrep", profiler)
    stage2 = StageExecutor(core, "dummy", "GPU", "Stage2_Embedding", profiler)
    # 假设你的 AI PC 有 NPU，如果没有，OpenVINO 会自动 fallback，或者改回 CPU 测试
    try:
        stage3 = StageExecutor(core, "dummy", "NPU", "Stage3_Classifier", profiler)
    except:
        print("NPU not found, using CPU for Stage 3")
        stage3 = StageExecutor(core, "dummy", "CPU", "Stage3_Classifier", profiler)

    print("\n🚀 Starting Pipeline Simulation (Streaming Mode)...")
    
    dummy_input = np.random.rand(1, 128).astype(np.float32)
    num_batches = 10
    
    # 模拟流水线：简单串行模拟 (为了演示代码逻辑)
    # 在真实的 Pipeline 中，你会用线程或队列让它们重叠运行
    # 这里我们用简单的循环来生成数据，重点是展示 Profiling 结果
    
    for i in range(num_batches):
        # 模拟 Stage 1 处理 Batch i
        out1 = stage1.run(dummy_input, batch_id=i)
        
        # 模拟 Stage 2 处理 Batch i (实际上可能正在并行处理 Batch i-1)
        out2 = stage2.run(out1, batch_id=i)
        
        # 模拟 Stage 3 处理 Batch i
        stage3.run(out2, batch_id=i)
        
        # 模拟不同 Stage 之间的 CPU 处理/Bubble
        time.sleep(0.002) 

    # 导出和分析
    profiler.export_chrome_trace()
    profiler.analyze_metrics()

if __name__ == "__main__":
    run_pipeline_simulation()
如何使用这个方案分析结果？
这个脚本最强大的地方在于生成的 pipeline_trace.json 和控制台输出的 Bubble 分析。

1. 可视化分析 (Visual Analysis)
运行脚本，得到 .json 文件。

打开 Chrome 浏览器，输入 chrome://tracing（或者使用更现代的 ui.perfetto.dev）。

加载 JSON 文件。

你将看到什么：

Timeline: 也就是你想要的“每个 Cycle 的执行情况”。

Bubbles: 你会清楚地看到 GPU 条带上的色块（任务）之间是否有空隙。空隙就是 Bubble。

Overhead: 你会看到色块的总长度（Wall Time）比你在 Args 里看到的 hw_time 要长。

色块长度 = Python + Driver + Hardware。

色块里的 hw_time = 纯 Hardware。

如果色块很长但 hw_time 很短，说明 CPU 调度或数据传输卡住了。

2. 数据指标分析 (Metric Analysis)
控制台会输出类似这样的表格：

Utilization (利用率):

HW Utilization: 比如 40%。这意味着在整个 Pipeline 运行期间，GPU 只有 40% 的时间在真正做计算。

Wall Utilization: 比如 60%。这意味着 GPU 驱动有 60% 的时间是忙碌的（包含了数据搬运）。

优化目标： 提高 HW Utilization，使其接近 Wall Utilization。

Avg Gap (Bubbles):

如果 Avg Gap 是 5ms，而你的 GPU 推理只需要 2ms。

结论： 你的上游（Stage 1）太慢了，或者中间的数据传输（Host overhead）太高，导致 GPU "吃不饱"。

对于你的特殊需求：并行 PU (Data Parallel)
如果你的某个 Stage 有多个 PU 并行（比如 2 个 GPU stream 同时跑）：

只需实例化多个 StageExecutor，例如 executor_gpu_1 和 executor_gpu_2。

在 log_execution 时，传入相同的 stage_name 和 device。

_get_tid_for_device 方法可以改进一下，给每个实例分配不同的 tid (Thread ID)，这样在 Chrome Tracing 里它们会显示在不同的行，你能直观地看到它们是否真的在并行重叠。

这个逻辑能够最清楚、最快速地计算出你想要的 Latency, Utilization 和 Bubbles