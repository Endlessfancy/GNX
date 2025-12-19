# Profiling Package Download

## 📦 下载文件

**文件路径**:
```
/home/haoyang/private/GNX_final/profiling_package.tar
```

**文件大小**: 70KB (tar格式，未压缩)

## 📥 下载方法

### 方法1: SCP (推荐)

在Windows PowerShell/cmd中：
```cmd
scp username@server:/home/haoyang/private/GNX_final/profiling_package.tar C:\Downloads\
```

### 方法2: WinSCP/FileZilla

1. 连接到服务器
2. 导航到 `/home/haoyang/private/GNX_final/`
3. 下载 `profiling_package.tar`

### 方法3: VS Code Remote

1. 打开Remote Explorer
2. 右键 `profiling_package.tar`
3. Download...

## 📂 解压后结构

```
profiling/                          ← 解压得到profiling文件夹
├── models/                        ← models在profiling内
│   └── Model_sage.py              (7.6KB)
├── profile_stages.py              (35KB) - 主脚本（已修改）
├── run_profiling.bat              (1.5KB) - Windows启动
├── test_cases.json                (1.2KB) - 测试配置
├── README.md                      (8.1KB) - 详细文档
├── PROFILING_SUMMARY.md           (6.5KB) - 技术文档
└── bug.md                         (1.2KB) - 已知问题
```

### 与compiler、executor并列

在你的GNX_final目录下：
```
GNX_final/
├── profiling/     ← 这个包
├── compiler/
└── executor/
```

## ✅ 修改内容

相比原始版本的改进：

1. **✅ 删除NPU Stage 3/4测试** - 自动跳过不兼容的stage
2. **✅ 修复bug1模块导入问题** - 简化导入逻辑
3. **✅ 移除Linux绝对路径** - 支持Windows本地运行
4. **✅ models在profiling内** - 结构清晰，开箱即用
5. **✅ tar格式** - 无需解压两次

## 🚀 Windows使用

### 解压

**Windows自带tar (Windows 10+)**:
```cmd
tar -xf profiling_package.tar
```

**7-Zip**:
右键 → 7-Zip → Extract Here

### 运行

```cmd
cd profiling
run_profiling.bat
```

就这么简单！

## 📊 测试范围

- ✅ **CPU**: Stage 1-7（完整7个）
- ✅ **GPU**: Stage 1-7（完整7个）
- ✅ **NPU**: Stage 1, 2, 5, 6, 7（自动跳过3/4，共5个）

NPU Stage 3/4 会显示：
```
Total: 5 stages × 15 sizes = 75 models (skipping Stage 3/4)
```

这是**正常的**！

## 📖 详细文档

解压后查看：
- `README.md` - 详细技术文档
- `PROFILING_SUMMARY.md` - 实现细节

---

**最后更新**: 2024-12-14
**包版本**: profiling_v8 (models inside, NPU Stage 3/4 skipped)
**格式**: tar (未压缩，方便Windows)
