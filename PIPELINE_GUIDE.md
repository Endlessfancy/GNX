# GNN Pipeline 完整运行指南

## ⚡ 重要更新

**✅ 完全独立化**: `executer/`目录现已完全独立，**无需**`executor copy/`目录！

- 新增: `executer/model_export_utils.py` - 独立模型导出工具
- 移除: 对`executor copy/`的所有依赖
- 简化: Windows部署只需`compiler/`和`executer/`两个目录

详见: `executer/STANDALONE_MIGRATION.md`

---

## 📁 自动化脚本说明

我们提供了4个自动化运行脚本，选择最适合您的：

### 1️⃣ **run_pipeline.py** ⭐ 推荐
**跨平台Python脚本**，功能最完整，输出最美观。

```bash
python run_pipeline.py
```

**优点**:
- ✅ 跨平台（Linux/Windows/macOS）
- ✅ 彩色输出，易读
- ✅ 自动保存日志
- ✅ 生成完整summary
- ✅ 错误处理完善

**输出**:
- 控制台: 彩色格式化输出
- 日志: `logs/compiler_output.log`, `logs/executor_output.log`
- 总结: `pipeline_summary.txt`

---

### 2️⃣ **run_full_pipeline.sh**
**Linux/macOS Shell脚本**，功能完整。

```bash
bash run_full_pipeline.sh
# 或
./run_full_pipeline.sh
```

**优点**:
- ✅ 原生Shell，速度快
- ✅ 彩色输出
- ✅ 详细的时间统计
- ✅ 自动验证结果

**输出**:
- 控制台: 彩色格式化输出
- 日志: `/tmp/compiler_output.log`, `/tmp/executor_output.log`
- 总结: `pipeline_summary.txt`

---

### 3️⃣ **run_full_pipeline.bat**
**Windows批处理脚本**。

```cmd
run_full_pipeline.bat
```

**优点**:
- ✅ Windows原生支持
- ✅ 双击即可运行
- ✅ 自动暂停查看结果

**输出**:
- 控制台: 格式化输出
- 日志: `%TEMP%\compiler_output.log`, `%TEMP%\executor_output.log`
- 总结: `pipeline_summary.txt`

---

### 4️⃣ **quick_run.sh**
**快速运行脚本**，最简洁。

```bash
bash quick_run.sh
```

**优点**:
- ✅ 最简洁的输出
- ✅ 只显示关键信息
- ✅ 快速验证流程

**适用场景**: 开发调试时快速验证

---

## 🚀 完整Pipeline流程

所有脚本都执行相同的3阶段流程：

```
┌─────────────────────────────────────────────────┐
│ Phase 1: Compiler (编译器)                       │
│                                                 │
│  [1.1] 清理旧的输出文件                           │
│  [1.2] 运行 test_compiler_flickr.py             │
│  [1.3] 生成 compilation_result.json              │
│  [1.4] 输出统计信息                               │
│                                                 │
│  输出:                                           │
│  - compilation_result.json                      │
│  - 占位符模型文件 (*.onnx, 126 bytes)            │
│                                                 │
│  耗时: ~10-15秒                                  │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ Phase 2: Model Export (模型导出)                 │
│                                                 │
│  由Executor自动处理:                             │
│  - 检测占位符模型 (<200 bytes)                   │
│  - 调用 pep_model_exporter.py                   │
│  - 生成真实ONNX模型 (~2-3 MB each)              │
│                                                 │
│  耗时: ~5-10秒 (首次运行)                        │
│        ~0秒 (模型已存在)                         │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ Phase 3: Executor (执行器)                       │
│                                                 │
│  [3.1] 加载Flickr图数据                          │
│  [3.2] 按partition_config分区                    │
│  [3.3] 收集ghost node特征                       │
│  [3.4] 加载和编译模型                             │
│  [3.5] 顺序执行8个subgraph                       │
│  [3.6] 合并结果                                  │
│  [3.7] 对比compiler估算                          │
│                                                 │
│  输出:                                           │
│  - embeddings [89250, 256]                      │
│  - 实际延迟测量                                  │
│  - 性能对比                                      │
│                                                 │
│  耗时: ~3-5秒                                    │
└─────────────────────────────────────────────────┘
```

---

## 📊 预期输出示例

### 终端输出（run_pipeline.py）

```
================================================================================
        GNN Complete Pipeline - Compiler → Executor → Verification
================================================================================

[Phase 1/3] Running Compiler...
  - Graph partitioning with METIS
  - PEP generation and optimization
  - Execution plan generation

  Cleaning old results...
  Running compiler...
✓ Compiler completed in 12.3s

  Compilation Summary:
    - Subgraphs: 8
    - Unique models: 2
    - Estimated makespan: 449.78ms

[Phase 2/3] Model Export...
  - Will be handled automatically by executor

[Phase 3/3] Running Executor...
  - Loading graph data and partitions
  - Collecting ghost node features
  - Exporting real ONNX/IR models (if needed)
  - Executing inference on all subgraphs

✓ Executor completed in 8.5s

================================================================================
                          PIPELINE SUMMARY
================================================================================

Execution Time Breakdown:
  ┌─────────────────────────────────────────────────────────┐
  │ Phase 1: Compiler                     12.3s             │
  │ Phase 2: Model Export                (auto)             │
  │ Phase 3: Executor                      8.5s             │
  ├─────────────────────────────────────────────────────────┤
  │ Total Pipeline Time:                  20.8s             │
  └─────────────────────────────────────────────────────────┘

Performance Results:
  ┌─────────────────────────────────────────────────────────┐
  │ Compiler Estimated Makespan:         449.78ms           │
  │ Actual Measured Latency:             412.53ms           │
  │ Estimation Error:                    -8.3%              │
  └─────────────────────────────────────────────────────────┘

✓ Estimation is accurate (within 20%)

Output Files:
  - Compilation result: compiler/output/compilation_result.json
  - ONNX models: compiler/output/models/*.onnx
  - Compiler log: logs/compiler_output.log
  - Executor log: logs/executor_output.log

================================================================================
                    Pipeline completed successfully!
================================================================================

Summary saved to: pipeline_summary.txt
```

### Summary文件（pipeline_summary.txt）

```
GNN Pipeline Execution Summary
Generated: 2025-12-17 14:32:15
================================================================================

TIMING BREAKDOWN
----------------
Compiler:                 12.3s
Executor:                 8.5s
Total:                    20.8s

PERFORMANCE METRICS
-------------------
Estimated Makespan:       449.78ms
Actual Latency:           412.53ms
Estimation Error:         -8.3%

CONFIGURATION
-------------
Dataset:                  Flickr (89,250 nodes, 899,756 edges)
Subgraphs:                8
Unique Models:            2

OUTPUT FILES
------------
- compiler/output/compilation_result.json
- compiler/output/models/*.onnx
- logs/compiler_output.log
- logs/executor_output.log

================================================================================
```

---

## 🔍 输出文件说明

### 主要输出

| 文件 | 描述 | 大小 |
|------|------|------|
| `compiler/output/compilation_result.json` | Compiler编译结果 | ~3 KB |
| `compiler/output/models/CPU_stages_*.onnx` | CPU模型 | ~2.3 MB |
| `compiler/output/models/GPU_stages_*.onnx` | GPU模型 | ~2.3 MB |
| `pipeline_summary.txt` | 性能总结 | ~1 KB |

### 日志文件

| 文件 | 描述 |
|------|------|
| `logs/compiler_output.log` | Compiler详细日志 |
| `logs/executor_output.log` | Executor详细日志 |

---

## ⚙️ 环境要求

### 必需依赖

```bash
conda activate hybridKGRAG  # 或您的环境名

# Python包
pip install torch torch_geometric onnxruntime
pip install numpy scipy

# 可选（用于NPU）
pip install openvino
```

### 验证环境

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
python -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"
```

---

## 🐛 常见问题

### Q1: "ModuleNotFoundError: No module named 'torch_geometric'"

**解决**:
```bash
conda activate hybridKGRAG
pip install torch_geometric
```

### Q2: "FileNotFoundError: compilation_result.json not found"

**原因**: Compiler未成功运行

**解决**:
```bash
# 查看compiler日志
cat logs/compiler_output.log
# 或
cat /tmp/compiler_output.log
```

### Q3: "Model export failed"

**原因**: 可能缺少依赖或模型导出错误

**解决**: 检查依赖是否完整
```bash
pip install torch torch_geometric torch-scatter onnxruntime
```

查看详细错误日志:
```bash
cat logs/executor_output.log
```

### Q4: Windows下脚本无法运行

**解决**:
```cmd
# 使用Python版本（跨平台）
python run_pipeline.py

# 或直接运行bat文件
run_full_pipeline.bat
```

### Q5: 实际延迟与估算差异大（>30%）

**可能原因**:
1. 模型质量问题（未训练）
2. 硬件差异（CPU/GPU性能）
3. 首次运行有缓存开销

**验证**:
```bash
# 运行第二次，消除缓存影响
python run_pipeline.py
```

---

## 📈 性能基准

### 预期性能（Flickr数据集）

| 阶段 | 时间 | 说明 |
|------|------|------|
| Compiler | 10-15s | 取决于k值数量 |
| Model Export | 5-10s | 首次运行，后续缓存 |
| Executor | 3-5s | 取决于硬件 |
| **总计** | **18-30s** | 首次完整运行 |

### 推理性能

| 指标 | 预期值 |
|------|--------|
| Compiler估算 | 449.78ms |
| 实际延迟 | 400-500ms |
| 误差 | ±10-20% |
| Per-subgraph | 50-60ms |
| Throughput | ~200K nodes/sec |

---

## 🎯 使用场景

### 场景1: 开发调试

```bash
# 使用快速脚本
bash quick_run.sh
```

### 场景2: 性能验证

```bash
# 使用完整脚本，获取详细报告
python run_pipeline.py
```

### 场景3: 论文实验

```bash
# 运行多次取平均
for i in {1..5}; do
    python run_pipeline.py
    mv pipeline_summary.txt pipeline_summary_$i.txt
done
```

### 场景4: CI/CD集成

```bash
# 自动化测试
python run_pipeline.py
if [ $? -eq 0 ]; then
    echo "Pipeline test passed"
else
    echo "Pipeline test failed"
    exit 1
fi
```

---

## 📝 下一步

### 验证完成后

1. **查看日志**: 检查详细输出
   ```bash
   cat logs/compiler_output.log
   cat logs/executor_output.log
   ```

2. **分析结果**: 对比estimated vs actual
   ```bash
   cat pipeline_summary.txt
   ```

3. **优化性能**: 根据瓶颈调整
   - Compiler: 减少k值测试范围
   - Executor: 实现pipeline并行

4. **论文数据**: 收集多次运行的统计数据

---

## ✅ 快速检查清单

运行前检查:
- [ ] conda环境已激活
- [ ] 依赖包已安装（torch, torch_geometric, onnxruntime）
- [ ] compiler目录存在
- [ ] executer目录存在
- [ ] executor copy目录存在（用于模型导出）

运行后验证:
- [ ] compilation_result.json已生成
- [ ] ONNX模型文件>1MB（不是占位符）
- [ ] pipeline_summary.txt已生成
- [ ] 实际延迟在合理范围内（400-500ms）
- [ ] 输出shape正确 [89250, 256]

---

**推荐**: 首次运行使用`python run_pipeline.py`，获取最详细的输出和报告！
