# File Dependencies - Visual Guide

This document shows which files depend on which, helping you understand what you need to download.

---

## Executor Dependencies

### To Run Executor (`test_executor.py`)

```
test_executor.py
    │
    ├─→ project_paths.py                    [REQUIRED]
    │
    ├─→ executor.py
    │   │
    │   ├─→ data_loader.py
    │   │   │
    │   │   ├─→ project_paths.py
    │   │   └─→ compiler/utils/graph_loader.py   [REQUIRED]
    │   │
    │   ├─→ ghost_node_handler.py
    │   │   └─→ data_loader.py
    │   │
    │   ├─→ model_manager.py
    │   │   └─→ model_export_utils.py
    │   │
    │   └─→ subgraph_executor.py
    │
    └─→ compiler/output/compilation_result.json  [REQUIRED - INPUT FILE]
```

### Executor File List (Minimum)

```
Required Python Files:
✓ project_paths.py                      (5 KB)
✓ executer/__init__.py                  (1 KB)
✓ executer/executor.py                  (10 KB)
✓ executer/subgraph_executor.py         (15 KB)
✓ executer/data_loader.py               (12 KB)
✓ executer/model_manager.py             (8 KB)
✓ executer/model_export_utils.py        (35 KB)
✓ executer/ghost_node_handler.py        (6 KB)
✓ executer/test_executor.py             (8 KB)

Required from Compiler:
✓ compiler/__init__.py                  (1 KB)
✓ compiler/utils/__init__.py            (1 KB)
✓ compiler/utils/graph_loader.py        (10 KB)

Required Input:
✓ compiler/output/compilation_result.json   (3-5 KB, from compiler)

Total: ~13 files, ~115 KB
```

---

## Compiler Dependencies

### To Run Compiler (`test_compiler_flickr.py`)

```
test_compiler_flickr.py
    │
    ├─→ project_paths.py                    [REQUIRED]
    │
    ├─→ compiler.py
    │   │
    │   ├─→ core/graph_partitioner.py
    │   │   └─→ utils/graph_loader.py
    │   │
    │   ├─→ core/pep_generator.py
    │   │   └─→ utils/config.py
    │   │
    │   ├─→ core/cost_estimator.py
    │   │   │
    │   │   ├─→ utils/profiling_loader.py
    │   │   │   └─→ profiling/results/lookup_table.json   [REQUIRED - INPUT FILE]
    │   │   │
    │   │   └─→ utils/interpolator.py
    │   │
    │   ├─→ core/global_optimizer.py
    │   │   └─→ utils/config.py
    │   │
    │   └─→ core/model_codegen.py
    │
    └─→ OUTPUT: compiler/output/compilation_result.json
```

### Compiler File List (Minimum)

```
Required Python Files:
✓ project_paths.py                          (5 KB)
✓ compiler/__init__.py                      (1 KB)
✓ compiler/compiler.py                      (20 KB)
✓ compiler/test_compiler_flickr.py          (5 KB)

Core Algorithms:
✓ compiler/core/__init__.py                 (1 KB)
✓ compiler/core/graph_partitioner.py        (15 KB)
✓ compiler/core/pep_generator.py            (20 KB)
✓ compiler/core/cost_estimator.py           (18 KB)
✓ compiler/core/global_optimizer.py         (25 KB)
✓ compiler/core/model_codegen.py            (10 KB)

Utilities:
✓ compiler/utils/__init__.py                (1 KB)
✓ compiler/utils/config.py                  (8 KB)
✓ compiler/utils/graph_loader.py            (10 KB)
✓ compiler/utils/profiling_loader.py        (8 KB)
✓ compiler/utils/interpolator.py            (6 KB)

Required Input:
✓ profiling/results/lookup_table.json       (50-100 KB, from profiling)

Total: ~15 files, ~200 KB
```

---

## Profiling Dependencies

### To Run Profiling (`profile_stages.py`)

```
profile_stages.py
    │
    ├─→ models/Model_sage.py                [REQUIRED]
    │   └─→ Contains all 7 stage definitions
    │
    ├─→ test_cases.json                     [REQUIRED]
    │   └─→ Test configurations
    │
    └─→ OUTPUT: results/
        ├─→ lookup_table.json               (Used by compiler)
        └─→ bandwidth_table.json
```

### Profiling File List (Minimum)

```
Required Python Files:
✓ profiling/profile_stages.py               (25 KB)
✓ profiling/models/Model_sage.py            (12 KB)

Required Config:
✓ profiling/test_cases.json                 (5 KB)

Total: 3 files, ~42 KB
```

---

## Complete Pipeline Dependencies

### To Run Complete Pipeline (`run_pipeline.py`)

```
run_pipeline.py
    │
    ├─→ project_paths.py
    │
    ├─→ Calls: compiler/test_compiler_flickr.py
    │   └─→ (See Compiler Dependencies above)
    │
    └─→ Calls: executer/test_executor.py
        └─→ (See Executor Dependencies above)
```

---

## Dependency Graph (Visual)

```
┌─────────────────────────────────────────────────────────────┐
│ PROFILING MODULE                                             │
│                                                              │
│  profile_stages.py                                          │
│       ↓                                                      │
│  results/lookup_table.json ───────────────────┐            │
└──────────────────────────────────────────────│─────────────┘
                                                 │
                                                 │ Used by
                                                 ↓
┌─────────────────────────────────────────────────────────────┐
│ COMPILER MODULE                                              │
│                                                              │
│  test_compiler_flickr.py                                    │
│       ↓                                                      │
│  compiler.py                                                │
│       ↓                                                      │
│  ├─ core/graph_partitioner.py                              │
│  ├─ core/pep_generator.py                                  │
│  ├─ core/cost_estimator.py ← lookup_table.json            │
│  └─ core/global_optimizer.py                               │
│       ↓                                                      │
│  output/compilation_result.json ─────────────┐             │
└──────────────────────────────────────────────│─────────────┘
                                                 │
                                                 │ Used by
                                                 ↓
┌─────────────────────────────────────────────────────────────┐
│ EXECUTOR MODULE                                              │
│                                                              │
│  test_executor.py                                           │
│       ↓                                                      │
│  executor.py ← compilation_result.json                      │
│       ↓                                                      │
│  ├─ data_loader.py                                          │
│  ├─ ghost_node_handler.py                                  │
│  ├─ model_manager.py                                        │
│  │   └─ model_export_utils.py                              │
│  └─ subgraph_executor.py                                   │
│       ↓                                                      │
│  Final embeddings [N, 256]                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Critical Input Files

These files are **generated** by running earlier stages:

### 1. `profiling/results/lookup_table.json`

**Generated by**: `profiling/profile_stages.py`

**Used by**: `compiler/core/cost_estimator.py`

**Content**: Stage latency measurements at different graph sizes

**Example**:
```json
{
    "CPU_stage_1": {
        "nodes": [1000, 5000, 10000, 50000],
        "edges": [5000, 25000, 50000, 250000],
        "latencies_ms": [10.2, 45.3, 89.1, 412.5]
    }
}
```

**Size**: ~50-100 KB

---

### 2. `compiler/output/compilation_result.json`

**Generated by**: `compiler/test_compiler_flickr.py`

**Used by**: `executer/test_executor.py`

**Content**: Complete compilation result with partition, PEP, and model references

**Example**:
```json
{
    "partition_config": {
        "k": 8,
        "num_subgraphs": 8,
        "subgraphs": [...]
    },
    "execution_plan": {
        "clusters": [...]
    },
    "statistics": {
        "makespan": 449.78
    }
}
```

**Size**: ~3-5 KB

---

## Dependency Matrix

| File/Module | Needs Profiling | Needs Compiler | Needs Executor | Needs project_paths.py |
|-------------|----------------|----------------|----------------|------------------------|
| **profiling/profile_stages.py** | - | ❌ | ❌ | ✅ |
| **compiler/test_compiler_flickr.py** | ✅ `lookup_table.json` | - | ❌ | ✅ |
| **executer/test_executor.py** | ❌ | ✅ `compilation_result.json` | - | ✅ |
| **run_pipeline.py** | ✅ | ✅ | ✅ | ✅ |

---

## What You Need for Each Scenario

### Scenario 1: Run Executor with Existing Compilation Result

```bash
Download:
✓ project_paths.py
✓ executer/ (all files)
✓ compiler/utils/graph_loader.py
✓ compiler/output/compilation_result.json    # Must have!
```

**Total**: ~15 files, ~115 KB

---

### Scenario 2: Run Compiler with Existing Profiling Results

```bash
Download:
✓ project_paths.py
✓ compiler/ (all files)
✓ profiling/results/lookup_table.json        # Must have!
```

**Total**: ~20 files, ~250 KB

---

### Scenario 3: Run Everything from Scratch

```bash
Download:
✓ All files in repository
```

**Total**: ~100 files, ~5 MB

Then run:
```bash
# 1. Profiling (generates lookup_table.json)
cd profiling
python profile_stages.py --all

# 2. Compiler (generates compilation_result.json)
cd ../compiler
python test_compiler_flickr.py

# 3. Executor (uses compilation_result.json)
cd ../executer
python test_executor.py
```

---

## Import Dependencies

### Python Import Chain

```python
# executer/test_executor.py imports:
from executor import PipelineExecutor
    # executor.py imports:
    from data_loader import GraphDataLoader
        # data_loader.py imports:
        from compiler.utils.graph_loader import GraphLoader  # Cross-module!
    from ghost_node_handler import GhostNodeHandler
    from model_manager import ModelManager
        # model_manager.py imports:
        from model_export_utils import SimpleModelExporter
    from subgraph_executor import SubgraphExecutor
```

**Key**: `data_loader.py` imports from `compiler/`, so you need:
- `compiler/__init__.py`
- `compiler/utils/__init__.py`
- `compiler/utils/graph_loader.py`

---

## File Size Summary

| Component | Files | Code Size | With Data |
|-----------|-------|-----------|-----------|
| **project_paths.py** | 1 | 5 KB | 5 KB |
| **profiling/** | 20 | 500 KB | 50 MB* |
| **compiler/** | 30 | 1 MB | 5 MB** |
| **executer/** | 15 | 300 KB | 300 KB |
| **Documentation** | 15 | 500 KB | 500 KB |
| **Scripts** | 5 | 50 KB | 50 KB |
| **Total** | ~100 | **~2.5 MB** | **~56 MB** |

*Including exported ONNX models
**Including compilation results and placeholder models

---

## Quick Reference Commands

### Check if you have all dependencies:

```bash
# For executor
ls project_paths.py executer/executor.py compiler/utils/graph_loader.py compiler/output/compilation_result.json

# For compiler
ls project_paths.py compiler/compiler.py profiling/results/lookup_table.json

# For complete pipeline
ls project_paths.py run_pipeline.py profiling/profile_stages.py compiler/test_compiler_flickr.py executer/test_executor.py
```

### Verify imports work:

```bash
# Test executor imports
python -c "import sys; sys.path.insert(0, 'executer'); from executor import PipelineExecutor; print('✓ Executor OK')"

# Test compiler imports
python -c "import sys; sys.path.insert(0, 'compiler'); from compiler import GNNCompiler; print('✓ Compiler OK')"
```

---

## Summary

### Key Takeaways

1. **Executor needs**: Its own files + `graph_loader.py` + `compilation_result.json`
2. **Compiler needs**: Its own files + `lookup_table.json`
3. **Complete pipeline**: All files in repo (~5 MB, trivial to download)

### Recommendation 🌟

**Just clone the entire repository!**

```bash
git clone <your-repo-url>
```

**Why**:
- ✅ Only ~5 MB (smaller than most photos)
- ✅ No missing dependencies
- ✅ All paths work automatically
- ✅ Can run any component
- ✅ Easy to update with `git pull`

**Don't overthink it** - download everything! 😊
