# Windows 部署完整指南

## 📦 需要的文件（完整清单）

### ✅ 最小部署包

```
GNX_final/
├── compiler/                          # 编译器目录（完整）
│   ├── core/                          # 核心算法
│   │   ├── __init__.py
│   │   ├── graph_partitioner.py
│   │   ├── pep_generator.py
│   │   ├── global_optimizer.py
│   │   └── execution_plan.py
│   ├── output/                        # 输出目录（会自动创建）
│   ├── test_compiler_flickr.py        # 编译器测试脚本
│   └── README.md
│
├── executer/                          # 执行器目录（完整，独立）
│   ├── __init__.py
│   ├── executor.py
│   ├── subgraph_executor.py
│   ├── data_loader.py
│   ├── model_manager.py
│   ├── model_export_utils.py         # ⭐ 独立模型导出工具
│   ├── ghost_node_handler.py
│   ├── test_executor.py
│   ├── README.md
│   └── STANDALONE_MIGRATION.md
│
├── run_pipeline.py                    # ⭐ 推荐：跨平台Python脚本
├── run_full_pipeline.bat              # Windows批处理脚本
├── run_full_pipeline.sh               # Linux脚本（Windows不需要）
├── quick_run.sh                       # 快速脚本（Windows不需要）
│
├── PIPELINE_GUIDE.md                  # 完整使用指南
├── WINDOWS_DEPLOYMENT.md              # 本文档
└── logs/                              # 日志目录（会自动创建）
```

### ❌ 不需要的文件/目录

```
❌ executor copy/        # 已移除依赖，无需此目录
❌ profiling/            # 仅开发时需要
❌ *.sh                  # Linux脚本，Windows不需要
❌ .git/                 # Git仓库（如果有）
```

---

## 🚀 快速部署步骤

### 步骤1: 准备文件

#### 方式A: 压缩包传输（推荐）

在Linux服务器上：

```bash
cd /home/haoyang/private/GNX_final

# 只打包需要的目录
tar -czf GNX_Windows.tar.gz \
    compiler/ \
    executer/ \
    run_pipeline.py \
    run_full_pipeline.bat \
    PIPELINE_GUIDE.md \
    WINDOWS_DEPLOYMENT.md
```

下载`GNX_Windows.tar.gz`到Windows，解压（使用7-Zip或WinRAR）。

#### 方式B: 最小化部署（只要核心文件）

```bash
# 创建最小包
mkdir GNX_minimal
cp -r compiler/ GNX_minimal/
cp -r executer/ GNX_minimal/
cp run_pipeline.py GNX_minimal/
cp PIPELINE_GUIDE.md GNX_minimal/

tar -czf GNX_minimal.tar.gz GNX_minimal/
```

---

### 步骤2: 安装Python环境

#### 选项A: 使用Anaconda（推荐）

1. 下载安装Anaconda: https://www.anaconda.com/download

2. 打开**Anaconda Prompt**，创建环境:

```cmd
conda create -n gnn_pipeline python=3.9
conda activate gnn_pipeline
```

#### 选项B: 使用标准Python

1. 下载安装Python 3.9+: https://www.python.org/downloads/

2. 打开**命令提示符 (CMD)**:

```cmd
python --version
# 应显示 Python 3.9 或更高
```

---

### 步骤3: 安装依赖包

在Anaconda Prompt或CMD中：

```cmd
# 激活环境（如果使用Anaconda）
conda activate gnn_pipeline

# 安装PyTorch (CPU版本)
pip install torch torchvision torchaudio

# 安装PyTorch Geometric
pip install torch-geometric

# 安装PyG扩展（Windows需要从whl安装）
# 方式1: 使用官方wheel
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# 方式2: 如果上面失败，使用conda
conda install pyg -c pyg

# 安装其他依赖
pip install onnxruntime numpy scipy networkx
```

**注意**: 如果需要GPU支持，安装CUDA版本:

```cmd
# PyTorch GPU版本（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyG GPU版本
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

### 步骤4: 安装METIS（可选但推荐）

METIS用于图分区，Windows安装稍复杂：

#### 方式A: 使用conda（最简单）

```cmd
conda install -c conda-forge metis
```

#### 方式B: 使用pip

```cmd
pip install metis-python
```

#### 方式C: 跳过METIS

如果安装失败，可以暂时跳过（compiler会使用Python实现的fallback）

---

### 步骤5: 验证安装

```cmd
# 检查Python版本
python --version

# 检查依赖
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
python -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"

# 检查torch-scatter
python -c "import torch_scatter; print('torch-scatter:', torch_scatter.__version__)"
```

**预期输出**:
```
Python 3.9.x
PyTorch: 2.0.x
PyG: 2.3.x
ONNX Runtime: 1.15.x
torch-scatter: 2.1.x
```

---

## 🎯 运行Pipeline

### 方式1: Python脚本（推荐）

```cmd
cd C:\path\to\GNX_final
python run_pipeline.py
```

**优点**:
- ✅ 跨平台兼容
- ✅ 彩色输出
- ✅ 详细日志
- ✅ 自动生成summary

### 方式2: Windows批处理

```cmd
cd C:\path\to\GNX_final
run_full_pipeline.bat
```

或直接双击`run_full_pipeline.bat`文件。

**优点**:
- ✅ 双击即可运行
- ✅ 自动暂停查看结果

---

## 📊 预期输出

### 控制台输出（run_pipeline.py）

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
  - Exporting real ONNX models (if needed)
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

### 生成的文件

运行成功后，会生成以下文件：

```
GNX_final/
├── compiler/output/
│   ├── compilation_result.json       # ~3 KB
│   └── models/
│       ├── block_0_CPU.onnx         # ~2.3 MB
│       └── block_0_GPU.onnx         # ~2.3 MB
│
├── logs/
│   ├── compiler_output.log          # 详细日志
│   └── executor_output.log
│
└── pipeline_summary.txt              # 性能总结
```

---

## ⚠️ 常见Windows问题

### 问题1: "Python不是内部或外部命令"

**原因**: Python未添加到PATH

**解决**:
1. 重新安装Python，勾选"Add Python to PATH"
2. 或手动添加到PATH:
   - 右键"此电脑" → 属性 → 高级系统设置 → 环境变量
   - 在"系统变量"中找到"Path"，添加Python安装路径

### 问题2: "No module named 'torch'"

**原因**: 依赖未正确安装

**解决**:
```cmd
# 重新安装
pip install torch torchvision torchaudio
pip install torch-geometric
```

### 问题3: "torch-scatter安装失败"

**原因**: Windows需要编译或预编译wheel

**解决方式A** (推荐):
```cmd
# 使用conda
conda install pytorch-scatter -c pyg
```

**解决方式B**:
```cmd
# 使用官方wheel
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### 问题4: 路径包含空格

**问题**: `C:\Program Files\GNX_final\` 路径有空格

**解决**: 使用短路径或引号
```cmd
cd "C:\Program Files\GNX_final"
python run_pipeline.py
```

或移动到无空格路径：
```cmd
C:\GNX\
```

### 问题5: 编码错误（中文乱码）

**解决**: 设置控制台编码
```cmd
chcp 65001
python run_pipeline.py
```

### 问题6: 权限问题

**解决**: 以管理员身份运行
- 右键CMD → "以管理员身份运行"

### 问题7: 长路径限制

**问题**: Windows默认260字符路径限制

**解决**:
1. **启用长路径支持**（Windows 10+）:
   - Win+R → `gpedit.msc`
   - 计算机配置 → 管理模板 → 系统 → 文件系统
   - 启用"启用 Win32 长路径"

2. 或使用短路径:
   ```cmd
   C:\GNX\
   ```

---

## 🔍 依赖检查清单

运行前完整检查：

```cmd
@echo off
echo === GNN Pipeline Dependency Check ===
echo.

echo [1/6] Python version:
python --version
echo.

echo [2/6] PyTorch:
python -c "import torch; print('  Version:', torch.__version__); print('  CUDA:', torch.cuda.is_available())"
echo.

echo [3/6] PyTorch Geometric:
python -c "import torch_geometric; print('  Version:', torch_geometric.__version__)"
echo.

echo [4/6] ONNX Runtime:
python -c "import onnxruntime; print('  Version:', onnxruntime.__version__)"
echo.

echo [5/6] torch-scatter:
python -c "import torch_scatter; print('  Version:', torch_scatter.__version__)"
echo.

echo [6/6] Directory structure:
dir /B compiler
dir /B executer
echo.

echo === All checks completed ===
pause
```

保存为`check_dependencies.bat`并运行。

---

## 📋 完整部署清单

### ✅ 软件安装
- [ ] Python 3.9+ 或 Anaconda
- [ ] PyTorch 2.0+
- [ ] PyTorch Geometric 2.3+
- [ ] torch-scatter
- [ ] ONNX Runtime
- [ ] (可选) METIS

### ✅ 文件部署
- [ ] `compiler/` 目录
- [ ] `executer/` 目录
- [ ] `run_pipeline.py`
- [ ] `PIPELINE_GUIDE.md`

### ✅ 验证测试
- [ ] `python --version` 显示3.9+
- [ ] 所有依赖import成功
- [ ] `python run_pipeline.py` 运行成功
- [ ] 生成`compilation_result.json`
- [ ] 生成ONNX模型 (>1MB)
- [ ] 生成`pipeline_summary.txt`

---

## 🎉 快速开始命令（复制即用）

```cmd
REM 1. 解压文件
cd C:\
REM 解压 GNX_Windows.tar.gz 到 C:\GNX_final\

REM 2. 创建环境
conda create -n gnn_pipeline python=3.9 -y
conda activate gnn_pipeline

REM 3. 安装依赖
pip install torch torchvision torchaudio
pip install torch-geometric
conda install pytorch-scatter -c pyg
pip install onnxruntime numpy scipy

REM 4. 运行Pipeline
cd C:\GNX_final
python run_pipeline.py

REM 5. 查看结果
type pipeline_summary.txt
```

---

## 📞 支持

遇到问题？

1. **查看日志**:
   ```cmd
   type logs\compiler_output.log
   type logs\executor_output.log
   ```

2. **检查文档**:
   - `PIPELINE_GUIDE.md` - 完整使用指南
   - `executer/STANDALONE_MIGRATION.md` - 独立化说明
   - `executer/README.md` - 执行器文档

3. **验证依赖**:
   ```cmd
   python -c "import torch, torch_geometric, onnxruntime, torch_scatter; print('All dependencies OK')"
   ```

---

## ✨ 总结

**Windows部署现在非常简单**:

1. ✅ **只需2个目录**: `compiler/` + `executer/`
2. ✅ **无外部依赖**: 无需`executor copy/`
3. ✅ **一键运行**: `python run_pipeline.py`
4. ✅ **完整功能**: 编译 → 导出 → 执行 → 验证

**文件总大小**: ~50 KB (代码) + 自动下载数据集

**运行时间**: ~20-30秒（首次运行）
