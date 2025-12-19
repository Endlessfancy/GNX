# Standalone Migration - 独立化迁移说明

## 📝 背景

之前的实现依赖于`executor copy/`目录中的以下文件：
- `pep_model_exporter.py` - 模型导出器
- `gnn_model.py` / `models/Model_sage.py` - GNN模型定义
- `pep_validator.py` - PEP验证器
- `stage_dependency_analyzer.py` - 阶段依赖分析（不存在）

这导致部署时需要携带整个`executor copy/`目录，增加了复杂性。

---

## ✅ 解决方案

### 创建了独立的模型导出工具

**新文件**: `executer/model_export_utils.py`

**功能**:
1. ✅ **完整的7-Stage模型定义** - 从`executor copy/models/Model_sage.py`提取
2. ✅ **简化的Combined Model** - 支持stages 1-7的组合模型
3. ✅ **独立的模型导出器** - 不依赖外部文件
4. ✅ **ONNX导出和验证** - 完整的导出流程

### 更新了model_manager.py

**修改**:
- ❌ 移除: `sys.path.append('/home/haoyang/private/GNX_final/executor copy')`
- ❌ 移除: `from pep_model_exporter import PEPModelExporter`
- ✅ 新增: `from model_export_utils import SimpleModelExporter`

**导出逻辑**:
```python
# 旧版 (需要executor copy/)
from pep_model_exporter import PEPModelExporter
exporter = PEPModelExporter()
exporter.export_combined_model(device, stages, output_path, dynamic=True)

# 新版 (完全独立)
from model_export_utils import SimpleModelExporter
exporter = SimpleModelExporter()
exporter.export_combined_model(
    device=device,
    stages=stages,
    output_path=output_path,
    num_nodes=89250,
    num_edges=899756,
    num_features=500,
    dynamic=True
)
```

---

## 📦 现在的文件结构

### 最小部署包 (不需要executor copy/)

```
GNX_final/
├── compiler/                    # 编译器（完整目录）
├── executer/                    # 执行器（完整目录）
│   ├── __init__.py
│   ├── executor.py
│   ├── subgraph_executor.py
│   ├── data_loader.py
│   ├── model_manager.py
│   ├── model_export_utils.py   # ⭐ 新增：独立模型导出
│   ├── ghost_node_handler.py
│   ├── test_executor.py
│   └── README.md
├── run_pipeline.py              # 跨平台脚本
├── run_full_pipeline.sh         # Linux脚本
├── run_full_pipeline.bat        # Windows脚本
└── PIPELINE_GUIDE.md            # 使用指南
```

**无需executor copy目录！**

---

## 🔍 技术细节

### model_export_utils.py 包含的类

#### 1. 7个Stage模块类

```python
SAGEStage1_Gather      # Neighbor gathering
SAGEStage2_Message     # Message computation
SAGEStage3_ReduceSum   # Sum aggregation
SAGEStage4_ReduceCount # Count neighbors
SAGEStage5_Normalize   # Mean normalization
SAGEStage6_Transform   # Linear transformation
SAGEStage7_Activate    # ReLU activation
```

#### 2. CombinedStagesModel

```python
class CombinedStagesModel(nn.Module):
    """
    组合多个stage成一个模型
    支持stages 1-7的任意组合
    """
    def forward(self, *args):
        # 执行完整的7-stage pipeline
        # x, edge_index → ... → output
```

#### 3. SimpleModelExporter

```python
class SimpleModelExporter:
    """
    简化版模型导出器
    - 初始化7个stage
    - 组合成combined model
    - 导出为ONNX
    - 验证正确性
    """
    def export_combined_model(self, device, stages, output_path, ...):
        # 完整的导出流程
```

---

## 📊 功能对比

| 功能 | 旧版 (executor copy/) | 新版 (model_export_utils.py) |
|------|----------------------|------------------------------|
| 7-Stage模型 | ✅ | ✅ |
| Combined Model | ✅ 复杂依赖分析 | ✅ 简化版（支持1-7） |
| ONNX导出 | ✅ | ✅ |
| 动态/静态模型 | ✅ | ✅ (仅动态) |
| OpenVINO IR | ✅ | ❌ (暂不支持) |
| Multi-output | ✅ | ❌ (暂不需要) |
| 依赖分析 | ✅ 复杂 | ❌ (简化为单pipeline) |
| **外部依赖** | ❌ 需要多个文件 | ✅ **完全独立** |

---

## ⚙️ 使用方法

### 方式1: 直接使用SimpleModelExporter

```python
from model_export_utils import SimpleModelExporter

exporter = SimpleModelExporter()
exporter.export_combined_model(
    device="CPU",
    stages=[1, 2, 3, 4, 5, 6, 7],
    output_path="model.onnx",
    num_nodes=89250,
    num_edges=899756,
    num_features=500,
    dynamic=True
)
```

### 方式2: 通过ModelManager（自动调用）

```python
from executer import PipelineExecutor

executor = PipelineExecutor(
    compilation_result_path='compilation_result.json',
    dataset_name='flickr'
)
executor.prepare()  # 自动检查并导出模型
executor.execute()
```

### 方式3: 使用便捷函数

```python
from model_export_utils import check_and_export_model

pep_block = {'devices': ['CPU'], 'stages': [1, 2, 3, 4, 5, 6, 7]}
success = check_and_export_model(
    model_path="model.onnx",
    pep_block=pep_block,
    num_nodes=89250,
    num_edges=899756,
    num_features=500
)
```

---

## 🎯 简化的地方

### 1. 移除复杂的依赖分析

**旧版**: 使用`stage_dependency_analyzer`分析stage之间的数据依赖，支持multi-output缓存
**新版**: 简化为单pipeline执行（stages 1-7），无需复杂依赖管理

**原因**:
- Compiler生成的PEP目前都是完整的1-7 pipeline
- Multi-output主要用于pipeline parallelism（阶段2优化）
- 当前阶段1只需要sequential execution

### 2. 移除OpenVINO IR转换

**旧版**: 自动将ONNX转换为OpenVINO IR（用于NPU）
**新版**: 只导出ONNX（足够CPU/GPU使用）

**原因**:
- 当前测试主要在CPU/GPU
- IR转换需要额外依赖（openvino）
- 可以后续添加

### 3. 统一输入参数

**旧版**: 从PyG Data对象获取图参数
**新版**: 直接传入num_nodes, num_edges, num_features

**原因**:
- 更清晰的接口
- 便于测试和调试
- 减少对PyG的依赖

---

## ✅ 验证清单

运行测试确认功能正常：

```bash
cd /home/haoyang/private/GNX_final/executer

# 1. 测试独立导出工具
python model_export_utils.py

# 2. 测试完整pipeline
python test_executor.py

# 3. 使用自动化脚本
cd ..
python run_pipeline.py
```

**预期结果**:
- ✅ model_export_utils.py成功导出test_model.onnx (>1MB)
- ✅ test_executor.py成功运行并输出性能对比
- ✅ run_pipeline.py完成完整流程

---

## 🚀 部署优势

### Windows部署

**旧版** (需要executor copy/):
```
GNX_final/
├── compiler/
├── executer/
├── executor copy/        # ❌ 需要这个目录
│   ├── models/
│   ├── pep_model_exporter.py
│   ├── pep_validator.py
│   └── ...
└── run_pipeline.py
```

**新版** (完全独立):
```
GNX_final/
├── compiler/
├── executer/             # ✅ 自包含，无外部依赖
└── run_pipeline.py
```

### 文件减少

- **旧版**: ~15个文件 (executor copy/)
- **新版**: 0个额外文件
- **减少**: 100% 外部依赖移除

---

## 📝 后续扩展

如果需要完整功能，可以逐步添加：

### 阶段1 (当前)
- ✅ 基础7-stage模型导出
- ✅ Sequential execution
- ✅ ONNX格式

### 阶段2 (优化)
- [ ] Pipeline parallelism支持
- [ ] Multi-output缓存
- [ ] 复杂依赖分析

### 阶段3 (高级)
- [ ] OpenVINO IR转换
- [ ] NPU静态模型
- [ ] 自动shape推导

---

## 🎉 总结

**迁移完成！** executer/目录现在是**完全独立**的，无需任何外部依赖。

**关键改进**:
1. ✅ 移除了对`executor copy/`的依赖
2. ✅ 创建了独立的`model_export_utils.py`
3. ✅ 简化了模型导出流程
4. ✅ 保持了所有核心功能
5. ✅ 更易于部署和维护

**Windows部署**: 只需复制`compiler/`和`executer/`目录即可！
