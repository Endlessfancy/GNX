# Troubleshooting Guide

## Issue 1: Compiler Returns None / TypeError: 'NoneType' object is not subscriptable

**Error Message:**
```
Traceback (most recent call last):
  File "D:\Research\GNX\compiler\test_compiler_flickr.py", line 262, in <module>
    result = test_flickr_compilation()
  File "D:\Research\GNX\compiler\test_compiler_flickr.py", line 68, in test_flickr_compilation
    partition_config = result['partition_config']
TypeError: 'NoneType' object is not subscriptable
```

**Root Cause:**
The compiler failed during execution and returned `None` instead of a result dictionary.

**NEW in v1.1**: The compiler now shows detailed error messages before returning None:
- `✗ Graph partitioning failed: [error details]`
- `✗ No valid partitions generated!`
- `✗ Compilation failed for k=[value]: [error details]`
- `✗ All compilation attempts failed!`

**Look for these error messages in your output to identify the exact failure point.**

**Common Causes:**

### 1. Missing or Empty Profiling Data

**Symptoms:**
- `profiling/results/lookup_table.json` is missing or nearly empty (< 1KB)
- Compiler crashes during cost estimation

**Solution:**
```bash
# Option 1: Run profiling to generate lookup_table.json
cd profiling
python profile_stages.py --all

# Option 2: Use the backup lookup table (if available)
cp profiling/results/"lookup_table copy.json" profiling/results/lookup_table.json

# Verify the file size (should be ~15-20 KB)
ls -lh profiling/results/lookup_table.json
```

**Expected lookup_table.json structure:**
```json
{
  "stage_1": {
    "CPU": {
      "data": {
        "1000_5000": 5.14,
        "1000_10000": 9.42,
        ...
      }
    },
    "GPU": {...},
    "NPU": {...}
  },
  "stage_2": {...},
  ...
}
```

---

### 2. Missing PyTorch Geometric

**Symptoms:**
```
ModuleNotFoundError: No module named 'torch_geometric'
ImportError: torch_geometric not installed
```

**Solution:**
```bash
# Install PyTorch Geometric and dependencies
pip install torch torch-geometric torch-scatter onnxruntime numpy scipy networkx

# For METIS graph partitioning support
pip install metis  # or conda install metis

# Verify installation
python -c "import torch_geometric; print('✓ PyTorch Geometric installed')"
```

**Platform-specific installation:**

**Linux/macOS:**
```bash
pip install torch torch-geometric torch-scatter
```

**Windows:**
```bash
# May need to install from wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

### 3. METIS Not Available

**Symptoms:**
```
Warning: METIS partitioning failed, using random partitioning
Graph partitioning quality is poor
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libmetis-dev
pip install metis

# macOS
brew install metis
pip install metis

# Windows
# Download pre-built binary or use conda
conda install -c conda-forge metis
pip install metis
```

---

### 4. Missing Compiler Output Directory

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'compiler/output/compilation_result.json'
```

**Solution:**
```bash
# Create output directory
mkdir -p compiler/output

# Or run from project root to auto-create directories
python project_paths.py
```

---

## Issue 2: Dataset Download Fails

**Symptoms:**
```
Error downloading Flickr dataset
URLError or TimeoutError
```

**Solution:**
```bash
# PyTorch Geometric will auto-download on first run
# If download fails, manually download and place in:
# ~/.pyg/datasets/Flickr/  (Linux/macOS)
# C:\Users\<username>\.pyg\datasets\Flickr\  (Windows)

# Or specify custom data directory
export TORCH_HOME=/path/to/data
```

---

## Issue 3: Path Errors on Windows

**Symptoms:**
```
FileNotFoundError: compiler/output/compilation_result.json
Paths with backslashes not working
```

**Solution:**
The project uses `pathlib.Path` which should handle cross-platform paths automatically.

If you encounter path issues:
```python
# In project_paths.py, verify paths are created
from project_paths import ensure_dirs
ensure_dirs()  # Creates all necessary directories
```

---

## Issue 4: Running on Windows (D:\Research\GNX)

**Current Working Directory Issue:**

The error shows you're running from `D:\Research\GNX>`, but the script expects relative paths from project root.

**Solution:**
```bash
# Ensure you're in the project root
cd D:\Research\GNX

# Verify project structure
dir
# Should see: compiler/, executer/, profiling/, project_paths.py, run_pipeline.py

# Run from project root
python run_pipeline.py
```

---

## Quick Diagnostic Checklist

Run this before executing `run_pipeline.py`:

```bash
# 1. Check profiling data exists and is not empty
ls -lh profiling/results/lookup_table.json
# Expected: 15-20 KB

# 2. Verify Python dependencies
python -c "import torch; import torch_geometric; print('✓ Dependencies OK')"

# 3. Test project paths
python project_paths.py
# Should create directories and print paths

# 4. Check compiler can import
python -c "from compiler import GNNCompiler; print('✓ Compiler imports OK')"

# 5. Check executer can import
python -c "from executer import PipelineExecutor; print('✓ Executor imports OK')"
```

---

## Complete Dependency List

**Required Python packages:**
```bash
pip install torch>=2.0.0
pip install torch-geometric
pip install torch-scatter
pip install onnxruntime
pip install numpy
pip install scipy
pip install networkx
pip install metis  # Optional but recommended for quality partitioning
```

**System dependencies (Linux):**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    build-essential \
    libmetis-dev
```

---

## Verifying the Fix

After fixing the issues, test the complete pipeline:

```bash
# 1. Test profiling data is loaded
cd compiler
python -c "from utils import ProfilingLoader; p = ProfilingLoader('../profiling/results'); print('✓ Profiling data loaded')"

# 2. Test compiler (may take 1-2 minutes)
python test_compiler_flickr.py

# 3. Verify compilation result was created
ls -lh output/compilation_result.json
# Expected: 3-5 KB

# 4. Test complete pipeline
cd ..
python run_pipeline.py
```

---

## Expected Output (Successful Run)

**Compiler:**
```
============================================================
Dataset: FLICKR
  Nodes: 89,250
  Edges: 899,756
============================================================

[Phase 1] Graph Partition...
[Phase 2] Generating candidate PEPs...
[Phase 3] Pipeline-aware global optimization...
[Phase 4] Model code generation...
[Phase 5] Generating execution plan...

✓ Best Configuration: k=8
✓ Estimated Makespan: 449.78 ms
✓ Compilation result saved to: compiler/output/compilation_result.json
```

**Executor:**
```
[Phase 1] Preparing execution environment...
[Phase 2] Loading graph data and partitions...
[Phase 3] Processing clusters...
[Phase 4] Exporting models to ONNX...
[Phase 5] Executing pipeline...
[Phase 6] Verifying results...

✓ Pipeline execution completed successfully!
✓ Final embedding shape: [89250, 256]
```

---

## Still Having Issues?

### Enable Verbose Logging

Edit `compiler/utils/config.py`:
```python
verbose = True  # Shows detailed progress
```

### Check Log Files

```bash
# Compiler logs
cat logs/compiler_output.log

# Executor logs
cat logs/executor_output.log
```

### Report an Issue

If problems persist, please report with:
1. Full error message and stack trace
2. Output of `pip list | grep torch`
3. Python version: `python --version`
4. Operating system
5. Contents of `profiling/results/lookup_table.json` (first 20 lines)

---

## Summary of Fixed Issue

**Your specific error was caused by:**
1. Empty `lookup_table.json` file (56 bytes with dummy data)
2. Complete profiling data was in `lookup_table copy.json`

**Fix applied:**
```bash
cp profiling/results/"lookup_table copy.json" profiling/results/lookup_table.json
```

**Verification:**
```bash
ls -lh profiling/results/lookup_table.json
# Now shows: 18K (correct size)
```

The compiler should now work correctly! 🎉
