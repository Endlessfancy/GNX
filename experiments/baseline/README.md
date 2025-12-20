# GNX Baseline Experiments

This directory contains baseline performance tests for GraphSAGE on the Flickr dataset, running on single devices without any graph partitioning or parallelism.

## Purpose

These baselines provide reference performance metrics for comparison with the pipeline and data parallel execution approaches implemented in GNX.

## Directory Structure

```
baseline/
├── CPU_only/
│   └── run_cpu_baseline.py       # CPU-only baseline test
└── GPU_only/
    └── run_gpu_baseline.py       # GPU-only baseline test
```

## Baseline Tests

### CPU-only Baseline

**Location**: `CPU_only/run_cpu_baseline.py`

**What it does**:
- Loads complete Flickr dataset (89,250 nodes, 899,756 edges)
- Runs GraphSAGE model (2 layers: 500→256→256) entirely on CPU
- No graph partitioning or parallelism
- Executes 10 timed inference runs + 2 warmup runs
- Records timing statistics (mean, std, min, max)

**How to run** (Linux):
```bash
cd /home/haoyang/private/GNX_final/experiments/baseline/CPU_only
python run_cpu_baseline.py
```

**How to run** (Windows):
```bash
cd experiments\baseline\CPU_only
python run_cpu_baseline.py
```

**Output**:
- Console: Real-time progress and results
- File: `cpu_baseline_results.txt` (saved in same directory)

---

### GPU-only Baseline

**Location**: `GPU_only/run_gpu_baseline.py`

**What it does**:
- Loads complete Flickr dataset
- Runs GraphSAGE model entirely on GPU (CUDA)
- No graph partitioning or parallelism
- Uses `torch.cuda.synchronize()` for accurate timing
- Executes 10 timed inference runs + 2 warmup runs
- Records timing statistics and GPU memory usage

**Requirements**:
- NVIDIA GPU with CUDA support
- PyTorch with CUDA enabled
- CUDA toolkit installed

**How to run** (Linux):
```bash
cd /home/haoyang/private/GNX_final/experiments/baseline/GPU_only
python run_gpu_baseline.py
```

**How to run** (Windows):
```bash
cd experiments\baseline\GPU_only
python run_gpu_baseline.py
```

**Output**:
- Console: Real-time progress, GPU info, and results
- File: `gpu_baseline_results.txt` (saved in same directory)

---

## Model Architecture

Both baselines use the same GraphSAGE architecture:

```
GraphSAGE (2 layers)
├── Layer 1: SAGEConv(500, 256)
├── ReLU activation
└── Layer 2: SAGEConv(256, 256)
```

**Total parameters**: ~513k

## Dataset: Flickr

- **Nodes**: 89,250
- **Edges**: 899,756 (undirected)
- **Node features**: 500 dimensions
- **Classes**: 7
- **Type**: Social network (image classification)

## Expected Output Format

### Console Output Example:
```
================================================================================
CPU-only Baseline Test
================================================================================

Loading Flickr dataset...
  Nodes: 89,250
  Edges: 899,756
  Features: 500

Initializing GraphSAGE model...
  Model parameters: 513,536
  Device: cpu

Running 2 warmup iterations...
  Warmup 1/2 completed
  Warmup 2/2 completed

Running 10 timed iterations...
  Run 1/10: 1234.56ms
  Run 2/10: 1245.67ms
  ...

================================================================================
Results
================================================================================
Output shape: torch.Size([89250, 256])

Timing Statistics (10 runs):
  Mean:   1240.50 ms
  Std:    15.30 ms
  Min:    1220.10 ms
  Max:    1260.80 ms

Results saved to: cpu_baseline_results.txt
```

### Results File Example:
```
================================================================================
CPU-only Baseline Results
================================================================================

Dataset:
  Name: Flickr
  Nodes: 89,250
  Edges: 899,756
  Features: 500

Model:
  Architecture: GraphSAGE (2 layers)
  Hidden dim: 256
  Output dim: 256

Performance:
  Device: CPU
  Number of runs: 10
  Mean time: 1240.50 ms
  Std time: 15.30 ms
  Min time: 1220.10 ms
  Max time: 1260.80 ms

Individual Run Times (ms):
  Run 1: 1234.56
  Run 2: 1245.67
  ...
```

## Comparison with GNX Pipeline Execution

After running the baselines, you can compare them with GNX's pipeline execution results:

**GNX Pipeline Execution** (from test_pipeline_execution.py):
- Uses graph partitioning (16 subgraphs)
- Employs pipeline parallelism (multi-block execution)
- Supports data parallelism (ratio-based node splitting)
- Enables heterogeneous device execution (CPU+GPU+NPU)

**Example Windows Result**:
- Total time: ~29,800 ms (with 2 clusters, 16 subgraphs)
- Output: [89,250, 256] embeddings ✓

**Performance Factors**:
- Partitioning overhead
- Model export/loading time
- Inter-device communication
- Parallel execution gains
- Device-specific optimizations

## Notes

1. **First Run**: May be slower due to dataset download and caching
2. **Warmup Runs**: Not included in timing statistics to avoid cold-start effects
3. **Reproducibility**: Times may vary based on system load and hardware
4. **GPU Timing**: Uses `torch.cuda.synchronize()` to ensure accurate measurements
5. **Memory**: GPU version reports GPU memory usage; CPU version runs entirely in system RAM

## Troubleshooting

### GPU Baseline Issues

**Error**: "CUDA is not available"
- Verify NVIDIA GPU is present: `nvidia-smi`
- Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with CUDA: Visit https://pytorch.org/get-started/locally/

**Error**: Out of GPU memory
- The Flickr dataset should fit in most modern GPUs (4GB+)
- Close other GPU-using applications
- Try reducing batch size (though this is full-graph inference)

### CPU Baseline Issues

**Error**: "No module named torch_geometric"
- Install PyTorch Geometric: `pip install torch_geometric`

**Slow Performance**:
- CPU baseline is expected to be slower than GPU
- Performance depends heavily on CPU cores and memory bandwidth
- First run may be slower due to dataset download

## Contact

For issues or questions about the baseline experiments, please refer to the main GNX documentation or open an issue on the GitHub repository.
