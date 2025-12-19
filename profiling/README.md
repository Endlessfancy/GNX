# GNN Stage Profiling System

这是GNX系统的离线profiling模块，用于生成性能查找表（lookup table）和带宽估计表（bandwidth table）。

## 概述

Profiling系统采用**增量执行方式**，分两个阶段完成：

### Phase 1: CPU/GPU（优先执行）
1. **导出CPU/GPU动态模型**：14个模型（每个stage各2个）
2. **测量CPU/GPU延迟**：210次测量（15个尺寸 × 2个PU × 7个stage）
3. **保存checkpoint**：立即保存CPU/GPU数据（防止NPU失败时数据丢失）

### Phase 2: NPU（后续执行）
4. **导出NPU静态模型**：105个模型（7个stage × 15个测试尺寸）
5. **测量NPU延迟**：105次测量（15个尺寸 × 1个PU × 7个stage）
6. **合并并分析**：合并所有数据，生成最终结果

### Phase 3: 分析和生成
7. **估计带宽**：使用线性回归分离计算时间和数据传输时间
8. **生成结果**：输出lookup_table.json和bandwidth_table.json

**优势**：即使NPU失败，您仍然有完整的CPU/GPU数据可用！

## 文件结构

```
profiling/
├── profile_stages.py          # 主脚本（695行）
├── test_cases.json            # 测试配置（15个测试用例）
├── run_profiling.bat          # Windows批处理脚本
├── README.md                  # 本文档
├── exported_models/           # 导出的ONNX和IR模型（自动创建）
└── results/                   # Profiling结果（自动创建）
    ├── checkpoint_cpugpu.json # CPU/GPU checkpoint（防数据丢失）
    ├── checkpoint_npu.json    # NPU checkpoint
    ├── lookup_table.json      # 计算时间查找表
    ├── bandwidth_table.json   # 带宽估计表
    ├── profiling_report.txt   # 统计报告
    └── raw_measurements.json  # 原始测量数据
```

## 测试用例配置

`test_cases.json`定义了15个测试用例，覆盖不同的节点和边数量组合：

| 节点数 | 边数量组合 |
|--------|-----------|
| 1000   | 1k, 2k, 3k, 5k, 10k |
| 2000   | 2k, 3k, 5k, 10k |
| 3000   | 3k, 5k, 10k |
| 5000   | 5k, 10k |
| 10000  | 10k |

- **特征维度**：500
- **预热迭代**：10次
- **测量迭代**：50次
- **预计时间**：3-4小时

## 使用方法

### 方法1：使用批处理脚本（推荐）

在Windows上双击运行：
```
run_profiling.bat
```

### 方法2：手动运行

激活conda环境后运行：
```bash
# 激活MIX环境
conda activate MIX

# 运行完整profiling流程（增量执行）
python profile_stages.py --all
```

### 增量执行机制（新特性）

脚本会自动按阶段执行：
1. **Phase 1**：导出并测量CPU/GPU → 立即保存checkpoint
2. **Phase 2**：导出并测量NPU → 保存checkpoint
3. **Phase 3**：合并所有数据并分析

**断点续跑**：
- 如果Phase 1完成后中断，重新运行会跳过Phase 1
- 如果Phase 2失败，可以单独重跑NPU部分
- checkpoint文件保存在 `results/checkpoint_*.json`

### 分步执行

你也可以分步执行profiling流程：

```bash
# 步骤1：只导出模型
python profile_stages.py --export

# 步骤2：只测量延迟
python profile_stages.py --measure

# 步骤3：只分析数据并生成表格
python profile_stages.py --analyze
```

### 只运行CPU/GPU（跳过NPU）

如果NPU有问题，可以只获取CPU/GPU数据：

```bash
# 运行到Phase 1完成
python profile_stages.py --all

# 观察到Phase 1完成后，按Ctrl+C中断
# 或者等待Phase 2 NPU失败

# 使用CPU/GPU checkpoint生成结果
python profile_stages.py --analyze
```

## 输出文件说明

### 1. lookup_table.json

计算时间查找表，格式如下：
```json
{
  "1000,1000,CPU,1": {
    "compute_time_ms": 10.5,      // 纯计算时间
    "transfer_time_ms": 2.0,      // 数据传输时间
    "total_time_ms": 12.5,        // 总延迟
    "std_ms": 0.3                 // 标准差
  },
  ...
}
```

**Key格式**：`{nodes},{edges},{PU},{stage_id}`
- nodes: 节点数量
- edges: 边数量
- PU: 处理单元（CPU, GPU, NPU）
- stage_id: 阶段编号（1-7）

### 2. bandwidth_table.json

带宽估计表，格式如下：
```json
{
  "CPU_stage1": 15000.5,   // MB/s
  "GPU_stage1": 25000.7,
  "NPU_stage5": 12000.1,
  ...
}
```

**Key格式**：`{PU}_stage{stage_id}`

### 3. profiling_report.txt

人类可读的统计报告，包括：
- 测试配置汇总
- 带宽估计（按stage和PU）
- 计算时间统计（平均值、标准差、范围）

## 技术细节

### GraphSAGE 7阶段分解

1. **Stage 1 - GATHER**：收集源节点特征
2. **Stage 2 - MESSAGE**：计算消息
3. **Stage 3 - REDUCE_SUM**：求和聚合
4. **Stage 4 - REDUCE_COUNT**：计数归一化
5. **Stage 5 - NORMALIZE**：归一化
6. **Stage 6 - TRANSFORM**：线性变换
7. **Stage 7 - ACTIVATE**：ReLU激活

### 带宽估计方法

使用线性回归分离计算时间和传输时间：

```
latency = compute_time + transfer_time
        = compute_time + (data_size / bandwidth)
```

通过拟合 `latency vs data_size`：
- **斜率（slope）**：1 / bandwidth
- **截距（intercept）**：≈ compute_time（基础计算时间）

### 数据大小估计

根据每个stage的输入类型估计数据大小：
- Stage 1: x(nodes×500) + edge_index(2×edges)
- Stage 2: x_j(edges×500) + edge_index(2×edges)
- Stage 3: messages(edges×500) + edge_index(2×edges)
- Stage 4: aggregated(nodes×500) + edge_index(2×edges)
- Stage 5: aggregated(nodes×500) + degree(nodes)
- Stage 6/7: x(nodes×500)

## 依赖要求

- **Python**: 3.8+
- **PyTorch**: 1.x+
- **PyTorch Geometric**: 最新版
- **OpenVINO**: 2023.x+（包含Python API）
- **NumPy**: 任意版本
- **scikit-learn**: 用于线性回归（可选，若无则跳过带宽估计）

## 故障排除

### 问题1：OpenVINO导入失败

**错误**：`ModuleNotFoundError: No module named 'openvino'`

**解决**：
```bash
pip install openvino openvino-dev
```

### 问题2：IR转换失败

**错误**：`⚠ IR conversion failed`

**解决**：
- 检查OpenVINO是否正确安装
- 确认ONNX模型导出成功
- 查看详细错误信息

### 问题3：NPU测量失败

**错误**：`⚠ NPU measurement failed`

**可能原因**：
- NPU驱动未安装
- OpenVINO未配置NPU支持
- 静态模型shape不匹配

**解决**：
- 脚本会自动fallback到PyTorch测量
- 若大量NPU测量失败，请检查NPU环境

### 问题4：sklearn不可用

**警告**：`⚠ sklearn not available, skipping bandwidth regression`

**影响**：
- 无法估计带宽
- lookup_table只包含原始延迟（无compute/transfer分离）

**解决**：
```bash
pip install scikit-learn
```

## 后续步骤

完成profiling后，使用生成的lookup_table.json和bandwidth_table.json进行：

1. **编译阶段（Compilation）**：
   - 生成PEP候选方案（28种组合）
   - 使用lookup table估计每个方案的延迟
   - 选择最优PEP

2. **运行时阶段（Runtime）**：
   - 根据实际输入大小插值查找计算时间
   - 使用bandwidth估计数据传输时间
   - 执行选定的PEP

## 性能预期

基于15个测试用例的精简配置（增量执行）：

### Phase 1: CPU/GPU
- **模型导出**：~5-8分钟（14个动态模型）
- **延迟测量**：~1.5-2小时（210次测量）
- **Checkpoint保存**：~1秒
- **小计**：~1.5-2小时

### Phase 2: NPU
- **模型导出**：~10-15分钟（105个静态模型）
- **延迟测量**：~1.5-2小时（105次测量）
- **小计**：~1.5-2小时

### Phase 3: 分析
- **带宽估计**：~30秒
- **结果生成**：~30秒
- **小计**：~1分钟

**总时间**：~3-4.5小时

**优势**：
- Phase 1完成后立即有CPU/GPU数据（~2小时）
- NPU失败不影响CPU/GPU结果
- 可分两次运行（先跑Phase 1，确认OK后再跑Phase 2）

## 参考

- Task.md：完整的GNX系统设计文档
- pep_model_exporter.py：PEP模型导出器
- models/Model_sage.py：GraphSAGE阶段模型定义
