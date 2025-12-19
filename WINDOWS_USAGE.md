# GNN Profiling Windows使用指南

## 📦 下载和解压

### 下载文件
```
profiling_package.tar.gz (16KB)
```

### 解压到Windows

**方法1 - 使用7-Zip:**
1. 下载7-Zip: https://www.7-zip.org/
2. 右键 `profiling_package.tar.gz` → 7-Zip → Extract Here
3. 再次右键 `profiling_package.tar` → 7-Zip → Extract Here

**方法2 - 使用WSL/Git Bash:**
```bash
tar -xzf profiling_package.tar.gz
```

### 解压后的结构

```
C:\your\path\profiling_package\
├── models\
│   └── Model_sage.py         ← GNN模型定义
└── profiling\
    ├── profile_stages.py     ← 主profiling脚本
    ├── run_profiling.bat     ← Windows启动脚本
    ├── test_cases.json       ← 测试配置
    ├── README.md             ← 详细文档
    └── PROFILING_SUMMARY.md  ← 技术说明
```

---

## 🚀 运行Profiling

### 前置要求

1. **Anaconda/Miniconda** 已安装
2. **MIX conda环境** 已创建并包含：
   - Python 3.x
   - PyTorch
   - NumPy
   - OpenVINO Runtime (用于NPU测试)

### 快速运行

```cmd
cd C:\your\path\profiling_package\profiling
run_profiling.bat
```

### 手动运行

```cmd
REM 1. 激活conda环境
conda activate MIX

REM 2. 进入profiling目录
cd profiling_package\profiling

REM 3. 运行profiling
python profile_stages.py --all

REM 或分步运行
python profile_stages.py --export    REM 只导出模型
python profile_stages.py --measure   REM 只测量
python profile_stages.py --analyze   REM 只分析
```

---

## 📊 输出结果

运行完成后，在 `profiling\results\` 目录下会生成：

| 文件 | 说明 |
|------|------|
| `lookup_table.json` | 性能查找表（给编译器用） |
| `bandwidth_table.json` | 设备间带宽数据 |
| `profiling_report.txt` | 人类可读的分析报告 |
| `checkpoint_cpugpu.json` | CPU/GPU测量checkpoint |
| `checkpoint_npu.json` | NPU测量checkpoint |

---

## ⏱️ 预计运行时间

- **完整profiling**: 3-4小时
  - CPU/GPU测量: ~1.5小时
  - NPU测量: ~1.5小时
  - 分析生成: ~5分钟

- **只CPU/GPU**: ~1.5小时
  ```cmd
  python profile_stages.py --measure --pu CPU GPU
  ```

---

## 📐 测试配置

### 默认测试（15个大小组合）

- 节点数: 1K, 2K, 3K, 5K, 10K
- 边数: 1K - 10K
- 特征维度: 500
- 每次测量: 10次预热 + 50次迭代

### 自定义测试

编辑 `test_cases.json`:
```json
{
  "test_cases": [
    {"nodes": 1000, "edges": 3000}
  ],
  "config": {
    "feature_dim": 500,
    "num_warmup": 10,
    "num_iterations": 50
  }
}
```

---

## ⚠️ NPU注意事项

### NPU自动跳过Stage 3/4

**原因**: NPU不支持scatter_add操作

**测试范围**:
- ✅ CPU: Stage 1-7（完整7个）
- ✅ GPU: Stage 1-7（完整7个）
- ✅ NPU: Stage 1, 2, 5, 6, 7（跳过3/4，共5个）

**输出中的提示**:
```
Total: 5 stages × 15 sizes = 75 models (skipping Stage 3/4)
```

这是**正常行为**，不是错误！

---

## 🔧 故障排除

### 问题1: "Cannot find models/Model_sage.py"

**原因**: 解压时models目录丢失

**解决**:
1. 确认目录结构正确：`models/` 和 `profiling/` 是同级目录
2. 检查 `models/Model_sage.py` 文件存在

### 问题2: "conda: command not found"

**原因**: Anaconda未安装或未添加到PATH

**解决**:
1. 安装Anaconda: https://www.anaconda.com/download
2. 重启cmd/PowerShell
3. 或使用完整路径: `C:\Env\Anaconda\Scripts\activate.bat MIX`

### 问题3: "MIX environment not found"

**原因**: MIX conda环境不存在

**解决**:
```cmd
REM 查看现有环境
conda env list

REM 如果没有MIX，创建它
conda create -n MIX python=3.10
conda activate MIX
conda install pytorch numpy
pip install openvino
```

### 问题4: GPU测试失败

**原因**: CUDA不可用

**解决**:
1. 检查CUDA: `nvidia-smi`
2. 只测试CPU/NPU:
   ```cmd
   python profile_stages.py --measure --pu CPU NPU
   ```

### 问题5: NPU测试全部失败

**原因**: OpenVINO或NPU驱动未安装

**解决**:
1. 安装OpenVINO: https://docs.openvino.ai/
2. 安装NPU驱动（Intel AI PC专用）
3. 或跳过NPU测试:
   ```cmd
   python profile_stages.py --measure --pu CPU GPU
   ```

---

## 📖 进阶用法

### 只导出模型（不测量）

```cmd
python profile_stages.py --export
```

生成的模型在 `profiling\exported_models\`

### 从checkpoint恢复

如果中途中断：
```cmd
python profile_stages.py --resume
```

### 自定义输出路径

修改 `profile_stages.py` 中的配置：
```python
MODELS_DIR = PROFILING_DIR / 'exported_models'
RESULTS_DIR = PROFILING_DIR / 'results'
```

---

## 📞 获取帮助

详细文档位于：
- `profiling/README.md` - 完整使用说明
- `profiling/PROFILING_SUMMARY.md` - 技术实现细节

---

## ✅ 快速检查清单

运行前确认：
- [ ] Anaconda已安装
- [ ] MIX环境已创建
- [ ] models/Model_sage.py存在
- [ ] 在profiling目录下运行
- [ ] 有足够磁盘空间（约5GB用于模型）

运行后检查：
- [ ] profiling/results/目录已创建
- [ ] lookup_table.json已生成
- [ ] NPU Stage 3/4显示"SKIP"（正常）
- [ ] 无其他错误信息

---

**最后更新**: 2024-12-14
**包版本**: profiling_v8 (NPU Stage 3/4 skipped)
