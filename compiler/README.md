# GNN Compiler

A compiler for optimizing Graph Neural Network (GNN) execution across heterogeneous devices (CPU, GPU, NPU).

## Overview

The GNN Compiler automatically partitions graphs, generates optimized execution plans (PEPs), and produces ONNX models for efficient inference on heterogeneous hardware.

**Key Features**:
- METIS-based graph partitioning
- Pipeline and data parallelism optimization
- Cost estimation with profiling data
- Automatic model code generation
- Global makespan minimization

---

## Directory Structure

```
compiler/
├── core/                       # Core compiler algorithms
│   ├── graph_partitioner.py    # METIS k-way graph partitioning
│   ├── pep_generator.py        # Parallel Execution Plan generation
│   ├── cost_estimator.py       # Latency estimation with profiling
│   ├── global_optimizer.py     # Pipeline-aware makespan optimization
│   └── model_codegen.py        # ONNX placeholder generation
│
├── utils/                      # Utility modules
│   ├── graph_loader.py         # PyG dataset loading
│   ├── profiling_loader.py     # Profiling data management
│   ├── interpolator.py         # Latency interpolation
│   └── config.py               # Configuration settings
│
├── output/                     # Compilation outputs (auto-created)
│   ├── compilation_result.json # Complete compilation result
│   └── models/                 # Generated ONNX models
│
├── compiler.py                 # Main compiler class
├── test_compiler_flickr.py     # Flickr dataset test script
└── README.md                   # This file
```

---

## Quick Start

### Prerequisites

```bash
conda activate hybridKGRAG  # or your environment name

# Required packages
pip install torch torch-geometric networkx metis scipy numpy
```

### Running the Compiler

```bash
cd /home/haoyang/private/GNX_final/compiler
python test_compiler_flickr.py
```

**Expected Output**:
```
=== GNN Compiler Test (Flickr Dataset) ===
Dataset: Flickr (89,250 nodes, 899,756 edges, 500 features)

[Phase 1/4] Graph Partitioning...
  Using METIS k-way partitioning
  ✓ Partitioned into 8 subgraphs

[Phase 2/4] PEP Generation...
  ✓ Generated 45 candidate PEPs

[Phase 3/4] Cost Estimation...
  ✓ Estimated latencies for all PEPs

[Phase 4/4] Global Optimization...
  ✓ Best configuration found

Results:
  Estimated makespan: 449.78ms
  Subgraphs: 8
  Unique models: 2

Outputs saved to: output/compilation_result.json
```

---

## Compilation Pipeline

### Phase 1: Graph Partitioning

**Module**: `core/graph_partitioner.py`

Uses METIS to partition the input graph into k balanced subgraphs.

```python
from compiler.core.graph_partitioner import GraphPartitioner

partitioner = GraphPartitioner(data, num_parts=8)
partition_config = partitioner.partition()
```

**Output**:
- `partition_map`: node_id → subgraph_id mapping
- `subgraph_sizes`: number of nodes in each subgraph
- `cut_edges`: number of edges crossing partitions

**Algorithm**: METIS k-way partitioning with edge-cut minimization

---

### Phase 2: PEP Generation

**Module**: `core/pep_generator.py`

Generates candidate Parallel Execution Plans (PEPs) combining pipeline and data parallelism.

```python
from compiler.core.pep_generator import PEPGenerator

generator = PEPGenerator(num_stages=7)
peps = generator.generate_all_peps()
```

**PEP Format**:
```python
[
    [['CPU', 'GPU'], [1, 2, 3], [0.5, 0.5]],  # Stages 1-3 on CPU+GPU (50/50)
    [['NPU'], [4, 5, 6, 7]]                    # Stages 4-7 on NPU
]
```

**Generation Strategy**:
- Enumerate stage split points
- For each split, try all device combinations
- Add data parallelism with ratio variations
- Filter invalid PEPs (e.g., reused devices, gaps)

---

### Phase 3: Cost Estimation

**Module**: `core/cost_estimator.py`

Estimates latency for each PEP using profiling data and interpolation.

```python
from compiler.core.cost_estimator import CostEstimator

estimator = CostEstimator(profiling_dir='../profiling')
latency = estimator.estimate_pep_latency(pep, num_nodes, num_edges)
```

**Estimation Process**:
1. Load profiling data (stage latencies at different graph sizes)
2. For each PEP block:
   - Interpolate latency based on subgraph size
   - Handle data parallelism (max latency among devices)
3. Sum all block latencies

**Profiling Data Format**:
```json
{
    "CPU_stage_1": {
        "nodes": [1000, 5000, 10000, ...],
        "edges": [5000, 25000, 50000, ...],
        "latencies_ms": [12.3, 45.6, 89.1, ...]
    }
}
```

---

### Phase 4: Global Optimization

**Module**: `core/global_optimizer.py`

Optimizes subgraph-to-PEP assignment to minimize total makespan with pipeline parallelism.

```python
from compiler.core.global_optimizer import GlobalOptimizer

optimizer = GlobalOptimizer(peps, costs, partition_config)
best_config = optimizer.optimize()
```

**Optimization Algorithm**:

1. **Clustering**: Group subgraphs with same PEP (share models)
2. **Bubble Analysis**: Simulate pipeline execution, track device idle time
3. **Iterative Improvement**:
   - Try swapping PEPs for subgraphs with worst bubbles
   - Accept if makespan improves
   - Early stopping (patience=5)

**Key Concepts**:
- **Makespan**: Max completion time across all devices
- **Bubble Time**: Device idle waiting for dependencies
- **Pipeline Parallelism**: Different stages execute concurrently on different devices

**Example**:
```
Timeline with bubbles:
Device   0     10    20    30    40    50 (ms)
CPU:     [S0       ][S1   ]~~[S2   ]      (~ = bubble)
GPU:     ~~[S0   ][S1       ]~~[S2     ]
NPU:     ~~~~[S0       ][S1       ][S2  ]
Makespan: 50ms (NPU finishes last)
```

---

## Output Format

### compilation_result.json

```json
{
    "partition_config": {
        "num_subgraphs": 8,
        "partition_map": [0, 0, 1, 2, ...],
        "subgraph_sizes": [11156, 11157, ...],
        "cut_edges": 159543
    },

    "execution_plan": {
        "clusters": [
            {
                "cluster_id": 0,
                "subgraph_ids": [0, 1, 2, 3, 4, 5, 6, 7],
                "pep": [
                    [["CPU", "GPU"], [1, 2, 3, 4, 5, 6, 7], [0.5, 0.5]]
                ],
                "model_refs": {
                    "block_0_CPU": "output/models/CPU_stages_1_2_3_4_5_6_7.onnx",
                    "block_0_GPU": "output/models/GPU_stages_1_2_3_4_5_6_7.onnx"
                }
            }
        ]
    },

    "statistics": {
        "num_unique_models": 2,
        "makespan": 449.78,
        "device_utilization": {
            "CPU": 0.85,
            "GPU": 0.82
        }
    }
}
```

---

## Configuration

### config.py

Key parameters:

```python
# Graph partitioning
NUM_PARTITIONS = 8              # Number of subgraphs
METIS_OBJECTIVE = 'cut'         # Minimize edge cuts

# PEP generation
NUM_STAGES = 7                  # GraphSAGE stages
DEVICES = ['CPU', 'GPU', 'NPU'] # Available devices
DATA_PARALLEL_RATIOS = [        # Node split ratios
    [0.5, 0.5],
    [0.6, 0.4],
    [0.3, 0.7]
]

# Cost estimation
PROFILING_DIR = '../profiling'  # Profiling data directory
INTERPOLATION_METHOD = 'linear' # linear, cubic, log

# Global optimization
MAX_ITERATIONS = 20             # Max optimization iterations
PATIENCE = 5                    # Early stopping patience
```

---

## Python Dependencies

### Required

```
torch >= 2.0.0
torch-geometric >= 2.3.0
numpy >= 1.21.0
scipy >= 1.7.0
networkx >= 2.6.0
metis >= 0.2
```

### Optional

```
pandas >= 1.3.0          # For profiling data analysis
matplotlib >= 3.4.0      # For visualization
```

### Installation

```bash
# Using pip
pip install torch torch-geometric numpy scipy networkx metis

# Using conda
conda install pytorch pytorch-geometric numpy scipy networkx -c pytorch -c pyg
conda install -c conda-forge metis
```

---

## Advanced Usage

### Custom Graph Partitioning

```python
from compiler import GNNCompiler

compiler = GNNCompiler(dataset_name='flickr')

# Custom partition strategy
compiler.partition_config = {
    'num_subgraphs': 16,          # More subgraphs
    'balance_constraint': 1.05,   # Tighter balance (default: 1.1)
    'objective': 'vol'            # Volume instead of edge-cut
}

result = compiler.compile()
```

### Custom PEP Candidates

```python
from compiler.core.pep_generator import PEPGenerator

generator = PEPGenerator(num_stages=7)

# Only CPU+GPU, no NPU
generator.devices = ['CPU', 'GPU']

# Custom stage splits
generator.split_points = [3, 5]  # Try [1-3|4-7] and [1-5|6-7]

peps = generator.generate_all_peps()
```

### Profiling Data Format

Create custom profiling data:

```python
import json

profiling_data = {
    "CPU_stage_1": {
        "nodes": [1000, 5000, 10000, 50000],
        "edges": [5000, 25000, 50000, 250000],
        "latencies_ms": [10.2, 45.3, 89.1, 412.5]
    },
    # ... more stages and devices
}

with open('profiling/custom_profiling.json', 'w') as f:
    json.dump(profiling_data, f, indent=2)
```

---

## Performance Tips

### 1. Partition Count

- **More subgraphs** (k=16, 32): Better load balancing, more overhead
- **Fewer subgraphs** (k=4, 8): Less overhead, potential imbalance
- **Recommendation**: Start with k=8, tune based on dataset size

### 2. PEP Search Space

```python
# Reduce search space for faster compilation
generator = PEPGenerator(num_stages=7)
generator.max_split_points = 2       # Limit to 2 splits (3 groups)
generator.data_parallel_ratios = [[0.5, 0.5]]  # Only 50/50 split
```

### 3. Optimization

```python
optimizer = GlobalOptimizer(...)
optimizer.max_iterations = 10        # Faster, less optimal
optimizer.patience = 3               # Early stop sooner
```

---

## Troubleshooting

### METIS Not Found

**Error**: `ModuleNotFoundError: No module named 'metis'`

**Solution**:
```bash
# Try conda
conda install -c conda-forge metis

# Or pip
pip install metis-python
```

### Profiling Data Missing

**Error**: `FileNotFoundError: profiling/CPU_stage_1.json not found`

**Solution**:
1. Run profiling first: `cd ../profiling && python profile_all_stages.py`
2. Or use mock data: `compiler.use_mock_profiling = True`

### Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce partition count: `num_partitions=4`
- Use CPU-only: `devices=['CPU']`
- Increase system memory

---

## Testing

### Unit Tests

```bash
# Test individual components
python -c "from compiler.core.graph_partitioner import GraphPartitioner; print('OK')"
python -c "from compiler.core.pep_generator import PEPGenerator; print('OK')"
```

### Integration Test

```bash
# Full compilation pipeline
python test_compiler_flickr.py

# Check output
ls -lh output/compilation_result.json
cat output/compilation_result.json | python -m json.tool
```

### Validation

```python
import json

with open('output/compilation_result.json') as f:
    result = json.load(f)

# Verify structure
assert 'partition_config' in result
assert 'execution_plan' in result
assert 'statistics' in result

# Verify PEP validity
pep = result['execution_plan']['clusters'][0]['pep']
all_stages = sum([block[1] for block in pep], [])
assert sorted(all_stages) == list(range(1, 8))  # All stages 1-7 covered

print("✓ Validation passed")
```

---

## API Reference

### GNNCompiler

Main compiler class.

```python
class GNNCompiler:
    def __init__(self, dataset_name: str = 'flickr'):
        """Initialize compiler with dataset"""

    def compile(self) -> Dict:
        """Run full compilation pipeline

        Returns:
            dict: Compilation result with partition_config,
                  execution_plan, statistics
        """
```

### GraphPartitioner

```python
class GraphPartitioner:
    def partition(self) -> Dict:
        """Partition graph using METIS

        Returns:
            dict: partition_map, subgraph_sizes, cut_edges
        """
```

### PEPGenerator

```python
class PEPGenerator:
    def generate_all_peps(self) -> List[List]:
        """Generate all valid PEP candidates

        Returns:
            list: List of PEPs in format [[[devices], [stages], [ratios]], ...]
        """
```

### CostEstimator

```python
class CostEstimator:
    def estimate_pep_latency(self, pep: List, num_nodes: int,
                           num_edges: int) -> float:
        """Estimate PEP latency

        Args:
            pep: Parallel Execution Plan
            num_nodes: Number of nodes in subgraph
            num_edges: Number of edges in subgraph

        Returns:
            float: Estimated latency in milliseconds
        """
```

### GlobalOptimizer

```python
class GlobalOptimizer:
    def optimize(self) -> Dict:
        """Optimize subgraph-to-PEP assignment

        Returns:
            dict: Best configuration with clusters, makespan
        """
```

---

## Citation

If you use this compiler in your research, please cite:

```bibtex
@inproceedings{gnn-compiler-2024,
  title={Efficient GNN Compilation for Heterogeneous Hardware},
  author={Your Name},
  booktitle={Conference},
  year={2024}
}
```

---

## License

[Add your license here]

---

## Contact

For questions or issues, please contact [your email] or open an issue on GitHub.
