# Flickr Pipeline Scheduling Results - JSON Schema

## 文件路径
`/home/haoyang/private/GNX_final/experiments/pipeline_time/flickr_results.json`

## 整体结构

```json
{
  "K=2": { ... },
  "K=4": { ... },
  "K=6": { ... },
  "K=8": { ... },
  "K=10": { ... }
}
```

每个 `K=N` 键对应一个分区配置的完整调度结果。

---

## 单个K值的结构

```json
{
  "k": 10,                           // 分区数
  "num_partitions": 10,              // 分区数量
  "optimal_order": [5, 6, 2, ...],   // Johnson's Rule 最优执行顺序
  "makespan_ms": 1362.84,            // 总执行时间 (毫秒)
  "cycle_times": [104.21, 118.20, ...],  // 每个cycle的时间 (N+1个cycle)
  "avg_utilization": {               // 平均设备利用率
    "cpu": 0.804,
    "gpu": 0.880,
    "npu": 0.918
  },
  "cycle_device_times": [ ... ],     // 每个cycle的详细设备时间
  "subgraphs": [ ... ]               // 每个子图的详细信息
}
```

---

## cycle_device_times 结构

每个cycle包含Stage 1和Stage 2的并行执行信息。对于N个子图，共有N+1个cycle：
- Cycle 0: 只有Stage 1 (流水线预热)
- Cycle 1 ~ N-1: Stage 1 和 Stage 2 并行
- Cycle N: 只有Stage 2 (流水线排空)

```json
{
  "cycle_time": 118.20,              // cycle总时间 (同步点)
  "stage1_subgraph_id": 6,           // 当前cycle执行Stage 1的子图ID (-1表示无)
  "stage2_subgraph_id": 5,           // 当前cycle执行Stage 2的子图ID (-1表示无)
  "dp_split_ratio": 0.433,           // Stage 1的CPU/GPU数据并行分割比例 (α)

  "cpu": {
    "compute": 85.62,                // CPU计算时间 (ms)
    "transfer": 20.65,               // CPU传输总时间 (ms)
    "transfer_in": {
      "time_ms": 10.38,              // CPU数据传入时间 (ms)
      "size_bytes": 25520000         // CPU数据传入大小 (bytes)
    },
    "transfer_out": {
      "time_ms": 10.27,              // CPU数据传出时间 (ms)
      "size_bytes": 25250000         // CPU数据传出大小 (bytes)
    },
    "total": 106.28,                 // CPU总时间 = compute + transfer
    "utilization": 0.899             // CPU利用率 = total / cycle_time
  },

  "gpu": {
    "compute": 81.41,                // GPU计算时间 (ms)
    "transfer": 24.82,               // GPU传输总时间 (ms)
    "transfer_in": {
      "time_ms": 12.48,              // GPU数据传入时间 (ms)
      "size_bytes": 33420000         // GPU数据传入大小 (bytes)
    },
    "transfer_out": {
      "time_ms": 12.34,              // GPU数据传出时间 (ms)
      "size_bytes": 33060000         // GPU数据传出大小 (bytes)
    },
    "total": 106.23,                 // GPU总时间 = compute + transfer
    "utilization": 0.899             // GPU利用率 = total / cycle_time
  },

  "npu": {
    "compute": 52.93,                // NPU计算时间 (ms)
    "transfer": 60.27,               // NPU传输总时间 (ms)
    "transfer_in": {
      "time_ms": 40.19,              // NPU数据传入时间 (ms)
      "size_bytes": 116510000        // NPU数据传入大小 (bytes)
    },
    "transfer_out": {
      "time_ms": 20.08,              // NPU数据传出时间 (ms)
      "size_bytes": 58200000         // NPU数据传出大小 (bytes)
    },
    "total": 113.20,                 // NPU总时间 = compute + transfer
    "utilization": 1.0               // NPU利用率 = total / cycle_time
  }
}
```

---

## subgraphs 结构

每个子图包含其Stage 1和Stage 2的完整时间分解。

```json
{
  "id": 6,                           // 子图ID
  "total_nodes": 29098,              // 总节点数
  "internal_edges": 46424,           // 内部边数

  "stage1": {
    "time": 106.28,                  // Stage 1总时间 (ms)
    "compute": 85.62,                // Stage 1计算时间 (ms)
    "transfer": 24.82                // Stage 1传输时间 (ms)
  },

  "stage2": {
    "time": 113.20,                  // Stage 2总时间 (ms)
    "compute": 52.93,                // Stage 2计算时间 (ms)
    "transfer": 60.27                // Stage 2传输时间 (ms)
  },

  "dp_split": {
    "ratio": 0.433,                  // CPU处理的比例 (α)

    "cpu": {
      "compute": 85.62,              // CPU计算时间 (ms)
      "transfer": 20.65,             // CPU传输总时间 (ms)
      "transfer_in": {
        "time_ms": 10.38,            // CPU数据传入时间 (ms)
        "size_bytes": 25520000       // CPU数据传入大小 (bytes)
      },
      "transfer_out": {
        "time_ms": 10.27,            // CPU数据传出时间 (ms)
        "size_bytes": 25250000       // CPU数据传出大小 (bytes)
      },
      "total": 106.28                // CPU总时间 = compute + transfer
    },

    "gpu": {
      "compute": 81.41,              // GPU计算时间 (ms)
      "transfer": 24.82,             // GPU传输总时间 (ms)
      "transfer_in": {
        "time_ms": 12.48,            // GPU数据传入时间 (ms)
        "size_bytes": 33420000       // GPU数据传入大小 (bytes)
      },
      "transfer_out": {
        "time_ms": 12.34,            // GPU数据传出时间 (ms)
        "size_bytes": 33060000       // GPU数据传出大小 (bytes)
      },
      "total": 106.23                // GPU总时间 = compute + transfer
    },

    "npu": {
      "transfer_in": {
        "time_ms": 40.19,            // NPU数据传入时间 (ms)
        "size_bytes": 116510000      // NPU数据传入大小 (bytes)
      },
      "transfer_out": {
        "time_ms": 20.08,            // NPU数据传出时间 (ms)
        "size_bytes": 58200000       // NPU数据传出大小 (bytes)
      }
    }
  }
}
```

---

## 数据传输计算说明

### Stage 1 (CPU/GPU) 数据传输

**输入数据**:
- `x[n, feat_dim]`: 节点特征矩阵，大小 = n × feat_dim × 4 bytes
- `edge_index[2, m]`: 边索引，大小 = 2 × m × 8 bytes (int64)

**输出数据**:
- `sum_agg[n, feat_dim]`: 聚合结果，大小 = n × feat_dim × 4 bytes
- `count[n]`: 计数，大小 = n × 4 bytes

### Stage 2 (NPU) 数据传输

**输入数据** (与边数无关):
- `sum_agg[n, feat_dim]`: 大小 = n × feat_dim × 4 bytes
- `count[n]`: 大小 = n × 4 bytes
- `x[n, feat_dim]`: 大小 = n × feat_dim × 4 bytes

**输出数据**:
- `activated[n, feat_dim]`: 大小 = n × feat_dim × 4 bytes

### 传输时间计算

```
transfer_time_ms = data_size_bytes / (bandwidth_gbps × 10^9) × 1000
```

当前使用的带宽:
- CPU: 2.46 GB/s
- GPU: 2.68 GB/s
- NPU: 2.90 GB/s

---

## 流水线模型说明

### 同步流水线模型

对于N个子图，使用N+1个cycle的同步流水线:

```
Cycle 0:   [S0.Stage1]  [  idle  ]    <- 预热
Cycle 1:   [S1.Stage1]  [S0.Stage2]   <- 并行执行
Cycle 2:   [S2.Stage1]  [S1.Stage2]
  ...
Cycle N-1: [SN-1.Stage1] [SN-2.Stage2]
Cycle N:   [  idle  ]   [SN-1.Stage2] <- 排空
```

### Cycle时间计算

```
cycle_time[0] = stage1_time[0]                              // 预热
cycle_time[i] = max(stage1_time[i], stage2_time[i-1])       // 1 <= i <= N-1
cycle_time[N] = stage2_time[N-1]                            // 排空

makespan = sum(cycle_times)
```

### Stage 1 时间计算 (数据并行)

```
cpu_total = cpu_compute + cpu_transfer_in + cpu_transfer_out
gpu_total = gpu_compute + gpu_transfer_in + gpu_transfer_out
stage1_time = max(cpu_total, gpu_total)
```

### Stage 2 时间计算

```
npu_total = npu_compute + npu_transfer_in + npu_transfer_out
stage2_time = npu_total
```

---

## Flickr数据集结果摘要

| K | Makespan (ms) | CPU Util | GPU Util | NPU Util |
|---|---------------|----------|----------|----------|
| 2 | 1831.83 | 65.3% | 76.6% | 36.5% |
| 4 | 1391.58 | 71.8% | 80.7% | 69.5% |
| 6 | 1372.03 | 76.2% | 74.7% | 80.4% |
| 8 | 1408.07 | 79.6% | 84.3% | 85.7% |
| **10** | **1362.84** | **80.4%** | **88.0%** | **91.8%** |

**最优配置**: K=10, Makespan = 1362.84 ms
