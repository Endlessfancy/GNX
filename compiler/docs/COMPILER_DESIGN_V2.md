# GNN Compiler 设计文档 V2
## Pipeline-Aware Global Optimization

---

## 概述

本编译器实现了基于Pipeline-Aware优化的GNN编译系统，主要特点：

1. **准确的Pipeline Makespan估算**：模拟真实的pipeline执行时间轴
2. **Bubble-aware优化**：识别并减少pipeline中的空闲时间
3. **PEP聚类**：将相同PEP的subgraph聚类以减少plan switching
4. **迭代优化**：通过切换PEP来持续改善makespan

---

## 系统架构

```
Input: Graph(nodes, edges), Profiling Data
  ↓
Phase 1: Graph Partition (METIS)
  → k个subgraphs，每个有(n, m, n_pad, m_pad)
  ↓
Phase 2: PEP Generation & Cost Estimation
  → 每个subgraph的Top-K候选PEP
  ↓
Phase 3: Global Optimization (Pipeline-Aware)
  → 聚类 → 排序减少bubble → 估算makespan → 调整PEP
  ↓
Phase 4: Model Codegen
  → 导出ONNX/IR模型并缓存
  ↓
Output: Execution Plan (partition + PEPs + order)
```

---

## Phase 3核心算法：Pipeline-Aware优化

### 算法流程

```python
1. 初始分配：每个subgraph选最快的PEP
2. 迭代优化（最多max_iterations次）:
   a. 按PEP结构聚类
   b. 在cluster内排序subgraph以减少bubble
   c. 估算真实pipeline makespan
   d. 找bubble最大的subgraph，尝试切换到次优PEP
   e. 如果makespan改善，保持切换；否则恢复
   f. 如果无改善或达到patience，停止
3. 返回最优assignment和clusters
```

### Pipeline Makespan估算

**关键创新**：准确模拟pipeline执行时间轴

```python
对每个cluster:
  block_end_times = [0, 0, ..., 0]  # 每个block的结束时间

  对cluster中的每个subgraph（按排序后的顺序）:
    对每个block:
      # 计算开始时间
      if block_idx == 0:
        start_time = block_end_times[0]  # 等前一个sg完成
      else:
        # 等两个条件
        prev_block_ready = block_end_times[block_idx-1] + transfer_time
        prev_sg_done = block_end_times[block_idx]
        start_time = max(prev_block_ready, prev_sg_done)

      # 计算bubble
      bubble = start_time - (block_end_times[block_idx-1] + transfer_time)

      # 更新结束时间
      block_end_times[block_idx] = start_time + compute_time

  cluster_completion = max(block_end_times)

total_makespan = sum(所有cluster的completion_time)
```

### Subgraph排序策略

**目标**：减少pipeline bubble

```python
计算block时间比 = Block_0_time / Block_1_time

if ratio > 1.2:  # Block 0明显慢
  先执行大的subgraph（让Block 1有更多工作）
elif ratio < 0.8:  # Block 1明显慢
  先执行小的subgraph（避免Block 1积压）
else:  # 比较平衡
  按大小排序
```

### Bottleneck调整

**策略**：找bubble最大的subgraph，尝试切换到次优PEP

```python
for bubble in top_5_worst_bubbles:
  sg = find_subgraph(bubble.sg_id)

  for alt_pep in sg的备选PEPs:
    临时切换 → 重新聚类 → 重新排序 → 重新估算makespan

    if new_makespan < current_makespan - threshold:
      保持切换
      return True
    else:
      恢复

return False  # 无法改善
```

---

## 关键数据结构

### PEP (Parallel Execution Plan)

```python
@dataclass
class PEPBlock:
    devices: List[str]    # 例如: ['CPU', 'GPU']
    stages: List[int]     # 例如: [1, 2, 3]
    ratios: List[float]   # 例如: [0.5, 0.5]

@dataclass
class PEP:
    blocks: List[PEPBlock]

    # 转换为executor格式
    def to_executor_format():
        return [[block.devices, block.stages, block.ratios] for block in blocks]
```

### Subgraph

```python
@dataclass
class Subgraph:
    id: int
    n: int       # 节点数
    m: int       # 边数
    n_pad: int   # NPU padding后节点数
    m_pad: int   # NPU padding后边数
```

---

## 与现有系统集成

### 1. 使用Profiling数据

```python
profiling = ProfilingLoader('/path/to/profiling_results')

# 查询stage时间（带插值）
time = profiling.get_stage_time('GPU', stage=3, n=2500, m=5000)

# 查询block时间
time = profiling.get_block_time('NPU', stages=[5,6,7], n=5000, m=8000)

# 查询带宽
bw = profiling.get_bandwidth('NPU', stage=5)
```

### 2. 生成Executor可用的执行计划

```python
execution_plan = {
  'partition_config': {...},
  'execution_plan': {
    'clusters': [
      {
        'pep': [
          [['CPU', 'GPU'], [1, 2, 3], [0.5, 0.5]],
          [['NPU'], [4, 5, 6, 7]]
        ],
        'subgraph_ids': [0, 3, 5],
        'model_refs': {...}
      },
      ...
    ]
  }
}
```

---

## 使用示例

### 命令行使用

```bash
cd /home/haoyang/private/GNX_final/compiler

# 基本编译
python compile_graph.py --nodes 50000 --edges 100000

# 自定义配置
python compile_graph.py \
  --nodes 100000 \
  --edges 200000 \
  --k-min 10 \
  --k-max 15 \
  --max-blocks 2 \
  --top-k 5 \
  --verbose
```

### Python API使用

```python
from compiler import GNNCompiler, CompilerConfig

# 创建配置
config = CompilerConfig(
    k_set=[8, 10, 12],
    max_pipeline_blocks=2,
    verbose=True
)

# 创建编译器
compiler = GNNCompiler(config)

# 执行编译
result = compiler.compile(total_nodes=50000, total_edges=100000)

# 访问结果
print(f"Best k: {result['partition_config']['k']}")
print(f"Makespan: {result['statistics']['makespan']:.2f}ms")
```

---

## 性能优化点

1. **PEP候选裁剪**：只保留Top-K，减少搜索空间
2. **Cluster聚类**：减少plan switching overhead
3. **Subgraph排序**：减少pipeline bubble
4. **迭代早停**：达到patience后停止
5. **Model缓存**：相同shape的NPU模型复用

---

## 已知限制与未来改进

### 当前限制

1. **图划分**：使用random mock，未集成真实METIS
2. **模型导出**：使用占位文件，需集成executor的model_exporter
3. **Pipeline段数**：默认最多2段，可扩展到3段
4. **DP比例**：只枚举固定几种(0.3/0.7, 0.5/0.5, 0.7/0.3)

### 未来改进方向

1. **集成真实METIS**：更精确的图划分
2. **动态DP比例**：根据设备负载动态调整
3. **Multi-cluster并行**：某些cluster可并行执行
4. **Cost model改进**：加入通信contention、memory限制等
5. **Auto-tuning**：基于实际执行反馈调整

---

## 文件结构

```
compiler/
├── core/
│   ├── graph_partitioner.py      # Phase 1
│   ├── pep_generator.py          # Phase 2
│   ├── cost_estimator.py         # Phase 2
│   ├── global_optimizer.py       # Phase 3 ⭐
│   └── model_codegen.py          # Phase 4
├── utils/
│   ├── config.py
│   ├── profiling_loader.py
│   └── interpolator.py
├── compiler.py                   # 主编译器
├── compile_graph.py              # CLI工具
└── test_compiler_simple.py       # 测试
```

---

**版本**: V2.0
**日期**: 2024-12-14
**核心改进**: Pipeline-aware makespan估算和bubble reduction优化
