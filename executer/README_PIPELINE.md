# Pipeline Execution Implementation

## 概述

这个实现支持在 GNX executor 中进行**多 block pipeline 执行**和**data parallelism**。

## 核心功能

1. **Block 间正确的数据传递**
   - Stage 1-5 输出 mean_agg 传递给 Stage 6-7
   - 自动处理中间结果的格式转换

2. **Data Parallelism (数据并行)**
   - 支持 CPU+GPU, GPU+NPU 等多设备组合
   - 按 ratio 分割节点进行并行处理
   - 自动合并多设备的输出

3. **NPU Padding 支持**
   - 自动为 NPU 静态模型添加 padding
   - 推理后自动去除 padding

4. **多 Cluster 执行**
   - 支持不同 subgraph 使用不同的 PEP
   - Cluster 间顺序执行，Cluster 内 subgraph 顺序执行

## 文件结构

```
executer/
├── node_partitioner.py          # 节点分区器（data parallelism）
├── npu_utils.py                  # NPU padding 工具
├── subgraph_executor.py          # 子图执行器（已修改）
├── executor.py                   # 主执行器（已修改）
├── model_manager.py             # 模型管理器（已修改）
├── test_helper.py                # 测试辅助函数
├── test_pipeline_execution.py    # 测试脚本
└── README_PIPELINE.md            # 本文件
```

## 测试配置

测试脚本使用以下两种 PEP 配置：

### PEP1 (Subgraphs 0-7)
```python
[
    [['CPU', 'GPU'], [1, 2, 3, 4, 5], [0.5, 0.5]],  # Block 0
    [['NPU'], [6, 7]]                                 # Block 1
]
```
- Block 0: CPU 和 GPU 各处理 50% 的数据，执行 Stage 1-5
- Block 1: NPU 处理全部数据，执行 Stage 6-7

### PEP2 (Subgraphs 8-15)
```python
[
    [['CPU'], [1, 2, 3, 4]],                         # Block 0
    [['GPU', 'NPU'], [5, 6, 7], [0.7, 0.3]]          # Block 1
]
```
- Block 0: CPU 处理全部数据，执行 Stage 1-4
- Block 1: GPU 处理 70%，NPU 处理 30%，执行 Stage 5-7

## 如何运行

### 方法 1: 使用测试脚本（推荐）

```bash
cd executer
python test_pipeline_execution.py
```

### 方法 2: 在代码中使用

```python
from executer.executor import PipelineExecutor
from executer.test_helper import create_two_pep_test_plan

# 创建自定义执行计划
custom_plan = create_two_pep_test_plan()

# 初始化 executor
executor = PipelineExecutor(
    custom_execution_plan=custom_plan,
    dataset_name='flickr'
)

# 准备数据和模型
executor.prepare()

# 执行
result = executor.execute()

# 查看结果
print(f"Total time: {result['total_time']:.2f}ms")
print(f"Embeddings shape: {result['embeddings'].shape}")
```

## 预期输出

```
======================================================================
Cluster 0: custom_pep_0
  PEP: [[['CPU', 'GPU'], [1, 2, 3, 4, 5], [0.5, 0.5]], [['NPU'], [6, 7]]]
  Subgraphs: [0, 1, 2, 3, 4, 5, 6, 7]
======================================================================

  Subgraph 0... 12.34ms
  Subgraph 1... 11.89ms
  ...

✓ Cluster 0 completed in 185.23ms

======================================================================
Cluster 1: custom_pep_1
  PEP: [[['CPU'], [1, 2, 3, 4]], [['GPU', 'NPU'], [5, 6, 7], [0.7, 0.3]]]
  Subgraphs: [8, 9, 10, 11, 12, 13, 14, 15]
======================================================================

  Subgraph 8... 10.45ms
  ...

✓ Cluster 1 completed in 142.67ms

✓ All clusters executed
  Total time: 327.90ms
```

## 技术细节

### Data Parallelism 实现

1. **按目标节点分割**
   ```python
   # CPU 负责节点 0-499
   # GPU 负责节点 500-999
   # 边按目标节点分配
   ```

2. **全图特征访问**
   - 每个设备持有完整的 `x_full` (用于 Gather)
   - 只输出自己负责的节点

3. **输出合并**
   ```python
   merged = torch.cat([cpu_output, gpu_output], dim=0)
   ```

### Block 间数据传递

```python
# Block 0 (Stage 1-5) → Block 1 (Stage 6-7)
block0_output = {'mean_agg': ..., 'x': ...}
block1_input = [mean_agg, x]
```

## 注意事项

1. **模型导出**
   - 首次运行会导出模型，可能需要几分钟
   - 模型保存在 `executer/models/` 目录

2. **NPU 设备**
   - 如果没有真实 NPU，代码会回退到 CPU 执行
   - 确保 NPU 驱动正确安装

3. **内存占用**
   - Data parallelism 会在每个设备上保存全图特征
   - 确保有足够的 GPU/NPU 内存

## 验证正确性

可以与全 GPU 方案对比输出：

```python
# 全 GPU 方案
executor_full = PipelineExecutor(
    compilation_result_path='compiler/output/compilation_result.json',
    dataset_name='flickr'
)
result_full = executor_full.execute()

# Pipeline 方案
result_pipeline = executor.execute()

# 对比
diff = torch.abs(result_full['embeddings'] - result_pipeline['embeddings'])
print(f"Max difference: {diff.max().item()}")  # 应该 < 1e-5
```

## 扩展和定制

### 创建自定义 PEP

```python
custom_plan = create_custom_execution_plan([
    {
        'pep': [
            [['CPU'], [1, 2, 3]],
            [['GPU'], [4, 5, 6, 7]]
        ],
        'subgraph_ids': [0, 1, 2, 3]
    }
])
```

### 修改分割比例

```python
pep = [
    [['CPU', 'GPU'], [1, 2, 3, 4, 5], [0.3, 0.7]],  # CPU 30%, GPU 70%
    ...
]
```

## 故障排查

### 问题 1: 模型输入维度不匹配

**现象**: `RuntimeError: shape mismatch`

**解决**: 检查 model_export_utils.py 中 Stage 6 Transform 的初始化

### 问题 2: NPU padding 错误

**现象**: NPU 输出形状不对

**解决**: 检查 npu_utils.py 中的 padding/unpadding 逻辑

### 问题 3: Data parallelism 输出重复

**现象**: 合并后的输出有重复节点

**解决**: 检查 NodePartitioner 的 node_range 是否不重叠

## 贡献者

- 实现基于 GNX_new (executor copy) 的成熟代码
- 移植并适配到 GNX_final/executer
