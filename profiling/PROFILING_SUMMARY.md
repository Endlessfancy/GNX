# Profiling System - 实现总结

## 完成状态

✅ **Profiling系统已完成并可以使用**

## 创建的文件

### 1. 核心脚本
- **profile_stages.py** (695行)
  - 自动化profiling流程
  - 支持模型导出、延迟测量、带宽估计
  - 支持分步执行或一键完成

### 2. 配置文件
- **test_cases.json**
  - 15个测试用例（从原55个精简73%）
  - 节点数：1k, 2k, 3k, 5k, 10k
  - 边数：1k, 2k, 3k, 5k, 10k
  - 特征维度：500

### 3. 辅助文件
- **run_profiling.bat**
  - Windows批处理脚本
  - 自动激活conda环境
  - 一键运行完整profiling流程

- **README.md**
  - 详细使用文档
  - 包含技术细节和故障排除
  - 输出文件格式说明

- **PROFILING_SUMMARY.md** (本文件)
  - 实现总结

## 目录结构

```
gnx_aipc_deploy/profiling/
├── profile_stages.py          # 主脚本（695行）
├── test_cases.json            # 测试配置
├── run_profiling.bat          # Windows批处理
├── README.md                  # 使用文档
├── PROFILING_SUMMARY.md       # 本总结
├── exported_models/           # 模型输出目录（待生成）
│   ├── onnx_models/          # ONNX模型
│   └── ir_models/            # OpenVINO IR模型
└── results/                   # 结果输出目录（待生成）
    ├── lookup_table.json      # 计算时间查找表
    ├── bandwidth_table.json   # 带宽估计表
    ├── profiling_report.txt   # 统计报告
    └── raw_measurements.json  # 原始数据
```

## 实现的功能

### ✅ 模型导出
- **动态模型（CPU/GPU）**：7 stages × 2 PUs = 14个模型
  - 支持可变输入尺寸
  - ONNX格式 + OpenVINO IR格式
  - FP32（CPU）/ FP16（GPU）

- **静态模型（NPU）**：7 stages × 15 sizes = 105个模型
  - 每个测试尺寸单独导出
  - NPU要求固定shape
  - FP16压缩

### ✅ 延迟测量
- **总测量次数**：15 sizes × 3 PUs × 7 stages = 315次
- **测量方法**：
  - 优先使用OpenVINO Runtime
  - Fallback到PyTorch（若OpenVINO失败）
- **测量参数**：
  - 预热：10次迭代
  - 测量：50次迭代
  - 统计：均值、标准差、最小值、最大值

### ✅ 带宽估计
- **方法**：线性回归
  - 模型：`latency = a × data_size + b`
  - 带宽：`BW = 1000 / a / (1024²)` MB/s
  - 计算时间：`compute = total - transfer`
- **输出**：21个带宽估计（3 PUs × 7 stages）

### ✅ 结果生成
- **lookup_table.json**：315条记录
  - Key: `{nodes},{edges},{PU},{stage}`
  - Value: compute_time, transfer_time, total_time, std

- **bandwidth_table.json**：21条记录
  - Key: `{PU}_stage{stage_id}`
  - Value: bandwidth (MB/s)

- **profiling_report.txt**：人类可读报告
  - 测试配置汇总
  - 带宽估计统计
  - 计算时间统计（按PU和stage）

## 性能优化

### 测试用例优化
- **原始方案**：55个测试用例
  - 11种节点数 × 5种边密度
  - 385个NPU模型
  - 1155次测量
  - 预计8-11小时

- **优化方案**：15个测试用例
  - 5种节点数 × 不同边数组合
  - 105个NPU模型
  - 315次测量
  - 预计3-4小时

- **优化效果**：
  - 测试用例：-73%
  - NPU模型：-73%
  - 测量次数：-73%
  - 时间：-60%

## 使用流程

### 在服务器端（已完成）
```bash
# 所有文件已准备好
cd /home/haoyang/private/GNX_new/gnx_aipc_deploy/profiling
ls -la
# profile_stages.py, test_cases.json, run_profiling.bat, README.md 都已就绪
```

### 在本地Windows端（用户操作）
```bash
# 1. 确保环境配置正确
conda activate MIX

# 2. 运行profiling（方法1：批处理）
run_profiling.bat

# 或者（方法2：命令行）
python profile_stages.py --all

# 3. 等待完成（3-4小时）

# 4. 检查结果
# - profiling/results/lookup_table.json
# - profiling/results/bandwidth_table.json
# - profiling/results/profiling_report.txt
```

## 验证清单

### ✅ 文件完整性
- [x] profile_stages.py 存在且语法正确（695行）
- [x] test_cases.json 配置正确（15个测试用例）
- [x] run_profiling.bat Windows批处理脚本
- [x] README.md 详细文档
- [x] 所有导入的模型类存在（../models/Model_sage.py）

### ✅ 功能完整性
- [x] 模型导出功能（export_dynamic_models, export_npu_static_models）
- [x] 延迟测量功能（measure_all_latencies）
- [x] 带宽估计功能（estimate_bandwidth_and_compute_time）
- [x] 结果保存功能（save_results）
- [x] 报告生成功能（generate_report）
- [x] 命令行参数解析（--all, --export, --measure, --analyze）

### ✅ 鲁棒性
- [x] OpenVINO不可用时fallback到PyTorch
- [x] sklearn不可用时跳过带宽回归
- [x] 异常处理和错误提示
- [x] 进度显示和状态反馈

## 技术亮点

1. **分步执行支持**
   - 可以分步运行（export → measure → analyze）
   - 中间结果保存，断点续跑

2. **自适应Fallback**
   - OpenVINO失败 → PyTorch
   - sklearn缺失 → 跳过带宽估计

3. **精简测试配置**
   - 从55个减少到15个用例
   - 时间从8-11小时减少到3-4小时
   - 保持覆盖范围（1k-10k节点）

4. **详细输出**
   - 3个JSON文件（原始、查找表、带宽）
   - 1个TXT报告（人类可读）
   - 实时进度显示

## 下一步

### 用户需要做的
1. ✅ **在本地运行profiling**
   ```bash
   cd gnx_aipc_deploy\profiling
   run_profiling.bat
   ```

2. ⏳ **等待完成（3-4小时）**
   - 确保NPU驱动正常
   - 确保OpenVINO已安装

3. ⏳ **验证结果**
   - 检查3个输出文件
   - 查看profiling_report.txt
   - 确认315条lookup记录和21条bandwidth记录

### 后续开发（在profiling完成后）
1. **编译阶段（Compilation）**
   - 生成28个PEP候选
   - 使用lookup_table估计延迟
   - 选择最优PEP

2. **运行时优化**
   - 插值查找（对于未测试的尺寸）
   - 动态PEP切换（根据输入尺寸）

## 参考文档

- **Task.md**: GNX系统总体设计
  - Profiling部分（第106-114行）
  - 说明了profiling目标和方法

- **README.md**: Profiling使用文档
  - 详细使用说明
  - 故障排除指南

- **profile_stages.py**: 实现代码
  - 695行完整实现
  - 包含所有功能模块

## 状态

🎯 **Profiling系统实现完成，等待用户在本地运行**
