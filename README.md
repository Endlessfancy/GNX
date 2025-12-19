# GNN Heterogeneous Compiler & Executor

A complete compilation and execution framework for Graph Neural Networks (GNNs) on heterogeneous hardware (CPU, GPU, NPU).

## Overview

This project provides an end-to-end system for optimizing GNN inference across heterogeneous devices:

1. **Compiler**: Automatically partitions graphs, generates optimized execution plans (PEPs), and produces ONNX models
2. **Executor**: Executes the compiled plans with automatic model export, ghost node handling, and result aggregation
3. **Automation**: Ready-to-use scripts for complete workflow execution

**Key Achievements**:
- 🚀 Automated heterogeneous device optimization
- 📊 Pipeline and data parallelism support
- ⚡ METIS-based graph partitioning
- 🔧 Built-in profiling and cost estimation
- 📦 Standalone execution (no external dependencies)

---

## Quick Start

### Prerequisites

```bash
# Create environment
conda create -n gnn_hetero python=3.9
conda activate gnn_hetero

# Install dependencies
pip install torch torch-geometric torch-scatter onnxruntime numpy scipy networkx
conda install -c conda-forge metis
```

### Run Complete Pipeline

```bash
cd /home/haoyang/private/GNX_final

# Option 1: Python script (cross-platform, recommended)
python run_pipeline.py

# Option 2: Shell script (Linux/macOS)
bash run_full_pipeline.sh

# Option 3: Batch script (Windows)
run_full_pipeline.bat
```

**Expected Runtime**: ~20-30 seconds (first run)

**Expected Output**:
```
=== GNN Complete Pipeline ===

[Phase 1/3] Compiler...
  ✓ Partitioned into 8 subgraphs
  ✓ Estimated makespan: 449.78ms

[Phase 2/3] Model Export...
  ✓ Exported 2 models

[Phase 3/3] Executor...
  ✓ Actual latency: 412.53ms
  ✓ Estimation error: -8.3%

Pipeline completed successfully!
Summary saved to: pipeline_summary.txt
```

---

## Project Structure

```
GNX_final/
├── compiler/                    # GNN Compiler
│   ├── core/                    # Core algorithms
│   │   ├── graph_partitioner.py # METIS graph partitioning
│   │   ├── pep_generator.py     # Execution plan generation
│   │   ├── cost_estimator.py    # Latency estimation
│   │   └── global_optimizer.py  # Makespan minimization
│   │
│   ├── utils/                   # Utilities
│   │   ├── graph_loader.py      # Dataset loading
│   │   ├── profiling_loader.py  # Profiling data
│   │   └── interpolator.py      # Latency interpolation
│   │
│   ├── output/                  # Compilation outputs
│   │   ├── compilation_result.json
│   │   └── models/              # ONNX models
│   │
│   ├── compiler.py              # Main compiler class
│   ├── test_compiler_flickr.py  # Test script
│   └── README.md                # Compiler documentation
│
├── executer/                    # Pipeline Executor
│   ├── executor.py              # Main orchestrator
│   ├── subgraph_executor.py     # Subgraph execution
│   ├── data_loader.py           # Graph data loading
│   ├── model_manager.py         # Model management
│   ├── model_export_utils.py    # Standalone ONNX export
│   ├── ghost_node_handler.py    # Ghost node features
│   ├── test_executor.py         # Test script
│   └── README.md                # Executor documentation
│
├── run_pipeline.py              # Automated workflow (Python)
├── run_full_pipeline.sh         # Automated workflow (Linux)
├── run_full_pipeline.bat        # Automated workflow (Windows)
├── quick_run.sh                 # Quick test (minimal output)
│
├── README.md                    # This file
├── PIPELINE_GUIDE.md            # Complete usage guide
└── WINDOWS_DEPLOYMENT.md        # Windows deployment guide
```

---

## Components

### 1. Compiler

**Purpose**: Optimize GNN execution plans for heterogeneous hardware

**Key Features**:
- METIS k-way graph partitioning (balanced, minimal edge-cut)
- PEP generation (pipeline + data parallelism)
- Cost estimation with profiling-based interpolation
- Global makespan minimization with bubble reduction

**Input**: Graph dataset (PyG format)
**Output**: `compilation_result.json` with partition config, execution plan, ONNX model references

**Documentation**: See `compiler/README.md`

---

### 2. Executor

**Purpose**: Execute compiled GNN models on actual hardware

**Key Features**:
- Automatic graph data partitioning
- Ghost node feature collection (pre-collection strategy)
- Standalone ONNX model export (no external dependencies)
- Sequential execution with performance measurement
- Result validation and aggregation

**Input**: `compilation_result.json` from compiler
**Output**: Node embeddings `[num_nodes, output_dim]`, actual latency, performance comparison

**Documentation**: See `executer/README.md`

---

### 3. Automation Scripts

**Purpose**: One-command execution of complete pipeline

**Available Scripts**:
1. **`run_pipeline.py`** ⭐ (Recommended)
   - Cross-platform (Linux/Windows/macOS)
   - Colored output
   - Detailed logging
   - Automatic summary generation

2. **`run_full_pipeline.sh`** (Linux/macOS)
   - Native shell script
   - Fast execution
   - Full features

3. **`run_full_pipeline.bat`** (Windows)
   - Native batch file
   - Double-click to run

4. **`quick_run.sh`** (Quick test)
   - Minimal output
   - Fast validation

**Documentation**: See `PIPELINE_GUIDE.md`

---

## Workflow

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. COMPILER                                                  │
│    Input: Flickr graph (89,250 nodes, 899,756 edges)       │
│    ├─ Graph Partitioning (METIS k=8)                       │
│    ├─ PEP Generation (45 candidates)                       │
│    ├─ Cost Estimation (profiling-based)                    │
│    └─ Global Optimization (makespan minimization)          │
│    Output: compilation_result.json                          │
│            Estimated makespan: 449.78ms                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. MODEL EXPORT (Automatic)                                 │
│    ├─ Check for placeholder models (<200 bytes)            │
│    ├─ Export real ONNX models using built-in utils         │
│    └─ Verify model correctness                             │
│    Output: ONNX models (~2.3 MB each)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. EXECUTOR                                                  │
│    ├─ Load and partition graph data                        │
│    ├─ Collect ghost node features (~212% overhead)         │
│    ├─ Load and compile ONNX models                         │
│    ├─ Execute 8 subgraphs sequentially                     │
│    └─ Aggregate results and measure latency                │
│    Output: Embeddings [89250, 256]                          │
│            Actual latency: 412.53ms                         │
│            Estimation error: -8.3%                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Python Dependencies

### Core Dependencies

```
Python >= 3.8
torch >= 2.0.0
torch-geometric >= 2.3.0
torch-scatter >= 2.1.0
onnxruntime >= 1.15.0
numpy >= 1.21.0
scipy >= 1.7.0
networkx >= 2.6.0
metis >= 0.2
```

### Installation

```bash
# Create environment
conda create -n gnn_hetero python=3.9
conda activate gnn_hetero

# PyTorch (CPU)
pip install torch torchvision torchaudio

# PyTorch Geometric
pip install torch-geometric
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Other dependencies
pip install onnxruntime numpy scipy networkx
conda install -c conda-forge metis
```

**For GPU support**:
```bash
# PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

## Configuration

### Compiler Settings

Edit `compiler/utils/config.py`:

```python
# Graph partitioning
NUM_PARTITIONS = 8                  # Number of subgraphs
METIS_OBJECTIVE = 'cut'             # Minimize edge cuts

# PEP generation
NUM_STAGES = 7                      # GraphSAGE stages
DEVICES = ['CPU', 'GPU', 'NPU']     # Available devices
DATA_PARALLEL_RATIOS = [            # Node split ratios
    [0.5, 0.5],
    [0.6, 0.4],
    [0.3, 0.7]
]

# Global optimization
MAX_ITERATIONS = 20                 # Max iterations
PATIENCE = 5                        # Early stopping
```

### Executor Settings

Edit `executer/executor.py`:

```python
class PipelineExecutor:
    def __init__(self, ...):
        self.sequential = True      # Sequential vs pipeline parallel
        self.verbose = True         # Detailed logging
        self.warmup_runs = 0        # Warmup iterations
```

---

## Advanced Usage

### Custom Dataset

```python
from compiler import GNNCompiler

# Compile for custom dataset
compiler = GNNCompiler(dataset_name='reddit')
result = compiler.compile()

# Execute
from executer import PipelineExecutor
executor = PipelineExecutor(
    compilation_result_path='compiler/output/compilation_result.json',
    dataset_name='reddit'
)
executor.prepare()
output = executor.execute()
```

### Manual Compilation Steps

```python
# Step 1: Partition
from compiler.core.graph_partitioner import GraphPartitioner
partitioner = GraphPartitioner(data, num_parts=8)
partition_config = partitioner.partition()

# Step 2: Generate PEPs
from compiler.core.pep_generator import PEPGenerator
generator = PEPGenerator(num_stages=7)
peps = generator.generate_all_peps()

# Step 3: Estimate costs
from compiler.core.cost_estimator import CostEstimator
estimator = CostEstimator(profiling_dir='profiling')
costs = [estimator.estimate_pep_latency(pep, ...) for pep in peps]

# Step 4: Optimize
from compiler.core.global_optimizer import GlobalOptimizer
optimizer = GlobalOptimizer(peps, costs, partition_config)
best_config = optimizer.optimize()
```

### Export Models Manually

```python
from executer.model_export_utils import SimpleModelExporter

exporter = SimpleModelExporter()
exporter.export_combined_model(
    device='CPU',
    stages=[1, 2, 3, 4, 5, 6, 7],
    output_path='my_model.onnx',
    num_nodes=10000,
    num_edges=50000,
    num_features=500,
    dynamic=True
)
```

---

## Performance

### Flickr Dataset Benchmarks

| Metric | Value |
|--------|-------|
| Nodes | 89,250 |
| Edges | 899,756 |
| Features | 500 |
| Subgraphs | 8 |
| Unique Models | 2 |
| **Compiler Time** | **~12s** |
| **Model Export** | **~5s** |
| **Executor Time** | **~8s** |
| **Total Pipeline** | **~25s** |
| **Estimated Makespan** | **449.78ms** |
| **Actual Latency** | **412.53ms** |
| **Estimation Error** | **-8.3%** |

### Optimization Techniques

1. **Graph Partitioning**: METIS balanced k-way with edge-cut minimization
2. **Pipeline Parallelism**: Different stages on different devices concurrently
3. **Data Parallelism**: Split nodes across devices (50/50, 60/40, etc.)
4. **Bubble Reduction**: Minimize device idle time through intelligent scheduling
5. **Model Caching**: Load models once, reuse across subgraphs

---

## Troubleshooting

### Common Issues

#### 1. METIS Not Found

```bash
conda install -c conda-forge metis
# or
pip install metis-python
```

#### 2. torch-scatter Installation Fails

```bash
# Use pre-built wheels
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Or conda
conda install pytorch-scatter -c pyg
```

#### 3. ONNX Runtime Error

```bash
pip uninstall onnxruntime
pip install onnxruntime==1.15.1
```

#### 4. Out of Memory

- Reduce number of subgraphs: `NUM_PARTITIONS = 4`
- Use CPU-only execution
- Clear GPU cache: `torch.cuda.empty_cache()`

### Debug Mode

```bash
# Enable verbose logging
export GNN_DEBUG=1
python run_pipeline.py

# Check detailed logs
cat logs/compiler_output.log
cat logs/executor_output.log
```

---

## Testing

### Verification Suite

```bash
# Test compiler
cd compiler
python test_compiler_flickr.py

# Test executor
cd executer
python test_executor.py

# Test complete pipeline
cd ..
python run_pipeline.py
```

### Unit Tests

```bash
# Compiler components
python -c "from compiler.core.graph_partitioner import GraphPartitioner; print('✓')"
python -c "from compiler.core.pep_generator import PEPGenerator; print('✓')"
python -c "from compiler.core.cost_estimator import CostEstimator; print('✓')"
python -c "from compiler.core.global_optimizer import GlobalOptimizer; print('✓')"

# Executor components
python -c "from executer.data_loader import GraphDataLoader; print('✓')"
python -c "from executer.model_export_utils import SimpleModelExporter; print('✓')"
python -c "from executer.executor import PipelineExecutor; print('✓')"
```

---

## Documentation

- **`README.md`** (this file): Project overview and quick start
- **`compiler/README.md`**: Detailed compiler documentation
- **`executer/README.md`**: Detailed executor documentation
- **`PIPELINE_GUIDE.md`**: Complete workflow guide
- **`WINDOWS_DEPLOYMENT.md`**: Windows deployment instructions
- **`executer/STANDALONE_MIGRATION.md`**: Migration from old executor

---

## Project Timeline

### Phase 1: Core Compiler ✅
- ✅ METIS graph partitioning
- ✅ PEP generation with pipeline/data parallelism
- ✅ Profiling-based cost estimation
- ✅ Global makespan optimization
- ✅ ONNX placeholder generation

### Phase 2: Executor Implementation ✅
- ✅ Graph data loading and partitioning
- ✅ Ghost node feature handling
- ✅ Standalone model export utilities
- ✅ Sequential subgraph execution
- ✅ Result aggregation and validation

### Phase 3: Automation ✅
- ✅ Cross-platform Python script
- ✅ Linux/macOS shell script
- ✅ Windows batch script
- ✅ Comprehensive documentation

### Phase 4: Future Enhancements
- [ ] Pipeline parallelism execution
- [ ] Multi-GPU support
- [ ] NPU with OpenVINO IR
- [ ] Adaptive load balancing
- [ ] More GNN models (GCN, GAT, etc.)
- [ ] More datasets (Reddit, Yelp, Amazon, etc.)

---

## System Requirements

### Minimum

- **OS**: Linux, macOS, Windows 10+
- **Python**: 3.8+
- **RAM**: 8 GB
- **Storage**: 5 GB

### Recommended

- **OS**: Ubuntu 20.04+ or Windows 11
- **Python**: 3.9
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with CUDA 11.8 (optional)
- **Storage**: 10 GB

---

## Windows Deployment

For Windows users, we provide detailed deployment instructions:

1. **Minimal package**: Only `compiler/` and `executer/` directories needed
2. **No external dependencies**: Fully standalone executor
3. **Simple installation**: Anaconda + pip
4. **One-click execution**: `run_full_pipeline.bat` or `python run_pipeline.py`

See `WINDOWS_DEPLOYMENT.md` for complete guide.

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## Citation

If you use this project in your research, please cite:

```bibtex
@inproceedings{gnn-hetero-compiler-2024,
  title={Efficient GNN Compilation and Execution on Heterogeneous Hardware},
  author={Your Name},
  booktitle={Conference Name},
  year={2024}
}
```

---

## License

[Add your license here]

---

## Acknowledgments

- **METIS**: Graph partitioning library
- **PyTorch Geometric**: GNN framework
- **ONNX Runtime**: Cross-platform inference engine

---

## Contact

For questions, issues, or collaboration:
- Email: [your email]
- GitHub Issues: [repository URL]
- Documentation: See README files in each directory

---

## Quick Reference

### Essential Commands

```bash
# Setup
conda create -n gnn_hetero python=3.9
conda activate gnn_hetero
pip install torch torch-geometric torch-scatter onnxruntime numpy scipy networkx
conda install -c conda-forge metis

# Run complete pipeline
python run_pipeline.py

# Run compiler only
cd compiler && python test_compiler_flickr.py

# Run executor only (after compiler)
cd executer && python test_executor.py

# Check results
cat pipeline_summary.txt
ls -lh compiler/output/models/*.onnx
```

### File Locations

- **Compilation result**: `compiler/output/compilation_result.json`
- **ONNX models**: `compiler/output/models/*.onnx`
- **Logs**: `logs/compiler_output.log`, `logs/executor_output.log`
- **Summary**: `pipeline_summary.txt`

---

**Last Updated**: 2024
**Version**: 1.0.0
