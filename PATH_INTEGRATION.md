# Path Integration & Module Collaboration

This document explains how different modules work together using relative paths for GitHub portability.

## Overview

All modules now use **relative paths** to ensure the project works anywhere without hardcoded absolute paths.

---

## Path Configuration

### Central Configuration: `project_paths.py`

The project includes a central path configuration module that all other modules can import:

```python
from project_paths import (
    PROJECT_ROOT,
    PROFILING_RESULTS_DIR,
    COMPILER_OUTPUT_DIR,
    COMPILATION_RESULT_FILE,
    get_model_path,
    get_log_path
)
```

**Key Features**:
- All paths relative to project root
- Automatic directory creation
- Helper functions for common paths
- Cross-platform compatibility (Windows/Linux/macOS)

---

## Module Integration

### 1. Profiling → Compiler

**Profiling Output**:
```
profiling/
└── results/
    ├── lookup_table.json      # Latency lookup table
    └── bandwidth_table.json   # Bandwidth estimates
```

**Compiler Input**:
```python
# compiler/utils/config.py
profiling_dir = Path(__file__).parent.parent.parent / 'profiling' / 'results'
```

**Integration**:
- Compiler loads profiling results from `../profiling/results/`
- Uses `ProfilingLoader` class to read JSON files
- Interpolates latencies for different graph sizes

**Verification**:
```bash
# Check profiling results exist
ls profiling/results/lookup_table.json

# Run compiler (will find profiling data automatically)
cd compiler
python test_compiler_flickr.py
```

---

### 2. Compiler → Executor

**Compiler Output**:
```
compiler/
└── output/
    ├── compilation_result.json    # Main result
    └── models/
        ├── CPU_stages_*.onnx      # Placeholder models
        └── GPU_stages_*.onnx
```

**Executor Input**:
```python
# executer/test_executor.py
project_root = Path(__file__).parent.parent
compilation_result_path = project_root / "compiler" / "output" / "compilation_result.json"
```

**Integration**:
- Executor loads `compilation_result.json` from `../compiler/output/`
- Reads partition config, PEP assignments, model references
- Auto-exports real ONNX models if placeholders detected

**Verification**:
```bash
# Check compilation result exists
ls compiler/output/compilation_result.json

# Run executor (will find compilation result automatically)
cd executer
python test_executor.py
```

---

### 3. Executor → Compiler Graph Loader

**Executor Need**: Load same graph data as compiler

**Solution**:
```python
# executer/data_loader.py
compiler_path = str(Path(__file__).parent.parent / 'compiler')
sys.path.insert(0, compiler_path)
from utils.graph_loader import GraphLoader
```

**Integration**:
- Executor dynamically adds compiler to Python path
- Imports `GraphLoader` to ensure same dataset
- Maintains consistency between compilation and execution

---

## Directory Structure

```
GNX_final/                           # PROJECT_ROOT
├── project_paths.py                 # ⭐ Central path configuration
│
├── profiling/                       # Profiling module
│   ├── results/                     # OUTPUT
│   │   ├── lookup_table.json        # → Used by compiler
│   │   └── bandwidth_table.json
│   ├── exported_models/
│   └── profile_stages.py
│
├── compiler/                        # Compiler module
│   ├── utils/
│   │   └── config.py                # Uses: ../profiling/results/
│   ├── output/                      # OUTPUT
│   │   ├── compilation_result.json  # → Used by executor
│   │   └── models/
│   │       ├── *.onnx
│   │       └── *.ir
│   └── test_compiler_flickr.py
│
├── executer/                        # Executor module
│   ├── data_loader.py               # Uses: ../compiler/utils/graph_loader.py
│   ├── model_manager.py             # Uses: ../compiler/output/models/
│   └── test_executor.py             # Uses: ../compiler/output/compilation_result.json
│
├── logs/                            # OUTPUT
│   ├── compiler_output.log
│   └── executor_output.log
│
├── run_pipeline.py                  # Uses relative paths
└── pipeline_summary.txt             # OUTPUT
```

---

## Path Resolution Examples

### Example 1: Profiling Results

```python
# Old (hardcoded)
profiling_dir = Path('/home/haoyang/private/GNX_final/result/profiling_results')

# New (relative)
profiling_dir = Path(__file__).parent.parent.parent / 'profiling' / 'results'

# Using central config
from project_paths import PROFILING_RESULTS_DIR
profiling_dir = PROFILING_RESULTS_DIR
```

### Example 2: Compilation Result

```python
# Old (hardcoded)
result_path = Path("/home/haoyang/private/GNX_final/compiler/output/compilation_result.json")

# New (relative)
project_root = Path(__file__).parent.parent
result_path = project_root / "compiler" / "output" / "compilation_result.json"

# Using central config
from project_paths import COMPILATION_RESULT_FILE
result_path = COMPILATION_RESULT_FILE
```

### Example 3: Model Files

```python
# Old (hardcoded in config)
model_path = '/home/haoyang/private/GNX_final/compiler/output/models/CPU_stages_1_2_3.onnx'

# New (relative in result JSON)
{
    "model_refs": {
        "block_0_CPU": "output/models/CPU_stages_1_2_3.onnx"  # Relative to compiler/
    }
}

# Resolved by executor
compiler_dir = Path(__file__).parent.parent / 'compiler'
model_path = compiler_dir / result['model_refs']['block_0_CPU']
```

---

## Testing Integration

### Test Script

```bash
#!/bin/bash
# test_integration.sh - Verify all modules can find each other

echo "=== Testing Module Integration ==="

# Test 1: Check profiling results accessible
echo "[1/4] Profiling → Compiler"
if [ -f "profiling/results/lookup_table.json" ]; then
    echo "  ✓ Profiling results found"
else
    echo "  ✗ Profiling results missing"
    exit 1
fi

# Test 2: Run compiler
echo "[2/4] Running Compiler"
cd compiler
python test_compiler_flickr.py > /dev/null 2>&1
if [ -f "output/compilation_result.json" ]; then
    echo "  ✓ Compilation result generated"
else
    echo "  ✗ Compilation failed"
    exit 1
fi
cd ..

# Test 3: Check executor can find result
echo "[3/4] Compiler → Executor"
if [ -f "compiler/output/compilation_result.json" ]; then
    echo "  ✓ Compilation result accessible to executor"
else
    echo "  ✗ Result not found"
    exit 1
fi

# Test 4: Run executor
echo "[4/4] Running Executor"
cd executer
python test_executor.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ Executor completed successfully"
else
    echo "  ✗ Executor failed"
    exit 1
fi
cd ..

echo "=== All Integration Tests Passed ✓ ==="
```

---

## Common Issues & Solutions

### Issue 1: "Profiling results not found"

**Error**:
```
FileNotFoundError: profiling/results/lookup_table.json not found
```

**Solution**:
```bash
# Run profiling first
cd profiling
python profile_stages.py --all
```

### Issue 2: "Compilation result not found"

**Error**:
```
ERROR: Compilation result not found at compiler/output/compilation_result.json
```

**Solution**:
```bash
# Run compiler first
cd compiler
python test_compiler_flickr.py
```

### Issue 3: "Module not found: utils.graph_loader"

**Error**:
```
ModuleNotFoundError: No module named 'utils.graph_loader'
```

**Cause**: Executor trying to import from compiler but path not added

**Solution**: Already fixed in `data_loader.py`:
```python
compiler_path = str(Path(__file__).parent.parent / 'compiler')
sys.path.insert(0, compiler_path)
```

---

## GitHub Deployment

### What Works Out of the Box

When someone clones from GitHub:

```bash
git clone https://github.com/your-repo/GNN_Compiler.git
cd GNN_Compiler

# Install dependencies
pip install torch torch-geometric torch-scatter onnxruntime numpy scipy networkx metis

# Run profiling (first time only)
cd profiling
python profile_stages.py --all
cd ..

# Run complete pipeline
python run_pipeline.py
```

**All paths work automatically!** No configuration needed.

### What Gets Created

First run creates:
```
GNX_final/
├── profiling/results/        # Created by profiling
├── compiler/output/          # Created by compiler
├── logs/                     # Created by run_pipeline.py
└── pipeline_summary.txt      # Created by run_pipeline.py
```

---

## Best Practices

### 1. Always Use Relative Paths

```python
# ✓ Good
project_root = Path(__file__).parent.parent
data_path = project_root / "compiler" / "output" / "data.json"

# ✗ Bad
data_path = Path("/home/username/project/compiler/output/data.json")
```

### 2. Use Central Configuration

```python
# ✓ Good
from project_paths import COMPILATION_RESULT_FILE
result = load(COMPILATION_RESULT_FILE)

# ✓ Also good (if can't import)
script_dir = Path(__file__).parent
result_path = script_dir.parent / "compiler" / "output" / "compilation_result.json"
```

### 3. Store Relative Paths in JSON

```json
{
    "model_refs": {
        "block_0_CPU": "output/models/CPU_stages_1_2_3.onnx"
    }
}
```

Then resolve at runtime:
```python
compiler_dir = Path("compiler")
full_path = compiler_dir / model_ref
```

---

## Verification Checklist

Before pushing to GitHub, verify:

- [ ] No hardcoded absolute paths in Python files
- [ ] All modules use `Path(__file__).parent` for relative paths
- [ ] JSON files store relative paths, not absolute
- [ ] `project_paths.py` creates all necessary directories
- [ ] Integration test passes
- [ ] Works from different working directories

---

## Migration from Old Paths

If you find hardcoded paths:

```python
# 1. Find them
grep -r "/home/haoyang" ./*.py

# 2. Replace with relative
# Before
path = Path("/home/haoyang/private/GNX_final/compiler/output")

# After
path = Path(__file__).parent.parent / "compiler" / "output"

# Or use central config
from project_paths import COMPILER_OUTPUT_DIR
path = COMPILER_OUTPUT_DIR
```

---

## Summary

✅ **All modules now use relative paths**
✅ **Central configuration in `project_paths.py`**
✅ **Profiling → Compiler → Executor chain works automatically**
✅ **GitHub portable (works anywhere)**
✅ **Cross-platform (Windows/Linux/macOS)**

**No manual configuration needed!**
