# GNN Pipeline Executor

A complete pipeline executor for running compiled GNN models across heterogeneous devices.

## Overview

The Pipeline Executor takes the compiler's output (`compilation_result.json`) and executes the complete GNN inference pipeline, including:
- Graph data loading and partitioning
- Ghost node feature collection
- Automatic ONNX model export
- Sequential subgraph execution
- Result aggregation and validation

**Key Features**:
- Standalone implementation (no external executor dependencies)
- Automatic model export with built-in utilities
- Ghost node handling with pre-collection strategy
- Support for data parallelism (CPU+GPU splits)
- Performance measurement and validation

---

## Directory Structure

```
executer/
├── executor.py                # Main pipeline orchestrator
├── subgraph_executor.py       # Single subgraph execution engine
├── data_loader.py             # Graph loading and partitioning
├── model_manager.py           # Model export, load, and caching
├── model_export_utils.py      # Standalone ONNX export utilities
├── ghost_node_handler.py      # Ghost node feature management
├── test_executor.py           # End-to-end test script
├── README.md                  # This file
└── STANDALONE_MIGRATION.md    # Migration documentation
```

---

## Quick Start

### Prerequisites

```bash
conda activate hybridKGRAG  # or your environment name

# Required packages
pip install torch torch-geometric torch-scatter onnxruntime numpy
```

### Running the Executor

```bash
# 1. Ensure compiler has run
cd /home/haoyang/private/GNX_final/compiler
python test_compiler_flickr.py

# 2. Run executor
cd ../executer
python test_executor.py
```

**Expected Output**:
```
=== GNN Pipeline Executor Test ===

[1/5] Loading compilation result...
  ✓ Loaded: 8 subgraphs, 1 cluster

[2/5] Loading graph data...
  ✓ Loaded Flickr: 89,250 nodes, 899,756 edges

[3/5] Collecting ghost node features...
  ✓ Collected features for all subgraphs

[4/5] Preparing models...
  Checking model files...
    ✓ Model exists: block_0_CPU (2.3 MB)
    ✓ Model exists: block_0_GPU (2.3 MB)
  ✓ All models loaded: 2

[5/5] Executing pipeline...
  Subgraph 0: 412.3ms
  Subgraph 1: 405.1ms
  ...
  ✓ Pipeline completed in 3.24s

Results:
  Output shape: [89250, 256]
  Actual latency: 412.5ms (avg per subgraph)
  Compiler estimate: 449.78ms
  Estimation error: -8.3%

✓ All tests passed!
```

---

## Execution Pipeline

### Phase 1: Initialization

**Module**: `executor.py` - `PipelineExecutor.__init__()`

Loads the compilation result and validates structure.

```python
from executer import PipelineExecutor

executor = PipelineExecutor(
    compilation_result_path='../compiler/output/compilation_result.json',
    dataset_name='flickr'
)
```

**Loaded Data**:
- Partition configuration (subgraph assignments)
- Execution plan (PEPs and model references)
- Statistics (estimated makespan)

---

### Phase 2: Graph Data Loading

**Module**: `data_loader.py` - `GraphDataLoader`

Loads the full graph and partitions it according to compiler's configuration.

```python
from executer.data_loader import GraphDataLoader

loader = GraphDataLoader(
    dataset_name='flickr',
    partition_config=partition_config
)

# Get data for a specific subgraph
subgraph_data = loader.get_subgraph_data(subgraph_id=0)
```

**Process**:
1. Load full PyG dataset (e.g., Flickr)
2. Build partition mapping (node_id → subgraph_id)
3. For each subgraph:
   - Extract owned nodes
   - Build local edge_index
   - Identify ghost nodes (neighbors from other subgraphs)
   - Create node mapping (global → local indices)

**Output Structure**:
```python
{
    'edge_index': torch.Tensor,      # Local edge indices [2, num_local_edges]
    'num_nodes': int,                # Total nodes (owned + ghost)
    'owned_nodes': list,             # Global IDs of owned nodes
    'ghost_nodes': list,             # Global IDs of ghost nodes
    'node_mapping': dict             # global_id → local_id
}
```

---

### Phase 3: Ghost Node Handling

**Module**: `ghost_node_handler.py` - `GhostNodeHandler`

Pre-collects features for all ghost nodes to avoid runtime communication.

```python
from executer.ghost_node_handler import GhostNodeHandler

handler = GhostNodeHandler(data_loader)

# Get combined features for a subgraph
combined_features = handler.get_combined_features(
    subgraph_id=0,
    owned_features=owned_x
)
```

**Strategy**: Pre-collection
- Collect all ghost features at initialization (one-time cost)
- Store in memory for fast access during execution
- Avoids inter-subgraph communication overhead

**Typical Overhead**: ~212% (normal for graph partitioning with ~18% cut edges)

---

### Phase 4: Model Management

**Module**: `model_manager.py` - `ModelManager`

Handles model export, loading, and caching.

```python
from executer.model_manager import ModelManager

manager = ModelManager(execution_plan)

# Check and export models if needed
manager.ensure_models_exist()

# Load and compile models
manager.load_models()

# Get a specific model
model = manager.get_model(block_id=0, device='CPU')
```

**Model Export** (`model_export_utils.py`):

The executor includes a **standalone model export utility** that:
1. Defines all 7 GraphSAGE stages
2. Combines stages into a single model
3. Exports to ONNX format
4. Verifies correctness

**No external dependencies required!**

```python
from executer.model_export_utils import SimpleModelExporter

exporter = SimpleModelExporter()
exporter.export_combined_model(
    device='CPU',
    stages=[1, 2, 3, 4, 5, 6, 7],
    output_path='model.onnx',
    num_nodes=10000,
    num_edges=50000,
    num_features=500,
    dynamic=True
)
```

**Placeholder Detection**:
- Files < 200 bytes are considered placeholders
- Automatically re-exported with real models

---

### Phase 5: Subgraph Execution

**Module**: `subgraph_executor.py` - `SubgraphExecutor`

Executes a single subgraph according to its assigned PEP.

```python
from executer.subgraph_executor import SubgraphExecutor

executor = SubgraphExecutor(
    subgraph_id=0,
    pep=pep,
    models=models
)

output, exec_time = executor.execute(edge_index, x, owned_nodes)
```

**Execution Flow**:

For each block in the PEP:
1. **Single Device**: Direct execution
   ```python
   # Example: [['CPU'], [1, 2, 3]]
   output = model(x, edge_index)
   ```

2. **Data Parallelism**: Split nodes, execute on each device, merge
   ```python
   # Example: [['CPU', 'GPU'], [1, 2, 3], [0.5, 0.5]]
   cpu_nodes = owned_nodes[:split_point]
   gpu_nodes = owned_nodes[split_point:]

   cpu_output = cpu_model(x, edge_index, cpu_nodes)
   gpu_output = gpu_model(x, edge_index, gpu_nodes)

   output = merge([cpu_output, gpu_output])
   ```

**Supported Configurations**:
- 1-3 blocks per PEP
- Single device or data parallelism
- Flexible stage combinations

---

### Phase 6: Pipeline Orchestration

**Module**: `executor.py` - `PipelineExecutor.execute()`

Coordinates execution across all subgraphs.

```python
result = executor.execute()
```

**Sequential Execution** (Current Implementation):
```python
for subgraph_id in range(num_subgraphs):
    # Get subgraph data
    data = loader.get_subgraph_data(subgraph_id)

    # Get combined features (owned + ghost)
    x = handler.get_combined_features(subgraph_id, owned_x)

    # Execute
    output, time = subgraph_executor.execute(data['edge_index'], x, ...)

    # Store results
    embeddings[owned_nodes] = output
    total_time += time
```

**Future**: Pipeline parallelism (concurrent execution of different stages)

**Output**:
```python
{
    'embeddings': torch.Tensor,     # [num_nodes, output_dim]
    'total_time': float,            # Total execution time (seconds)
    'subgraph_times': list,         # Per-subgraph times
    'avg_subgraph_time': float      # Average time per subgraph
}
```

---

## Model Export Utilities

### Built-in 7-Stage GraphSAGE

**Module**: `model_export_utils.py`

Complete standalone implementation:

```python
# Stage definitions
SAGEStage1_Gather       # Neighbor gathering
SAGEStage2_Message      # Message computation
SAGEStage3_ReduceSum    # Sum aggregation
SAGEStage4_ReduceCount  # Count neighbors
SAGEStage5_Normalize    # Mean normalization
SAGEStage6_Transform    # Linear transformation
SAGEStage7_Activate     # ReLU activation
```

### CombinedStagesModel

Combines multiple stages into a single executable model:

```python
from executer.model_export_utils import CombinedStagesModel

# Initialize stages
stages = [
    SAGEStage1_Gather(),
    SAGEStage2_Message(),
    # ... up to Stage7
]

# Combine
model = CombinedStagesModel(stages, stage_indices=[1,2,3,4,5,6,7])

# Use like any PyTorch model
output = model(x, edge_index)
```

### SimpleModelExporter

High-level ONNX export:

```python
from executer.model_export_utils import SimpleModelExporter

exporter = SimpleModelExporter()
exporter.export_combined_model(
    device='CPU',
    stages=[1, 2, 3, 4, 5, 6, 7],
    output_path='output/model.onnx',
    num_nodes=89250,
    num_edges=899756,
    num_features=500,
    dynamic=True  # Dynamic batch size
)
```

**Features**:
- Automatic dummy input generation
- Dynamic axes configuration
- Built-in verification (PyTorch vs ONNX)
- File size validation (>200 bytes)

---

## Python Dependencies

### Required

```
torch >= 2.0.0
torch-geometric >= 2.3.0
torch-scatter >= 2.1.0       # For scatter operations
onnxruntime >= 1.15.0        # For ONNX inference
numpy >= 1.21.0
```

### Optional

```
onnx >= 1.14.0              # For ONNX model inspection
openvino >= 2023.0          # For NPU/IR conversion (future)
```

### Installation

```bash
# Using pip
pip install torch torch-geometric torch-scatter onnxruntime numpy

# torch-scatter with pre-built wheels
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Or using conda
conda install pytorch pytorch-geometric -c pytorch -c pyg
conda install pytorch-scatter -c pyg
pip install onnxruntime
```

---

## Configuration

### Execution Settings

Modify in `executor.py`:

```python
class PipelineExecutor:
    def __init__(self, ...):
        # Execution mode
        self.sequential = True          # Sequential vs pipeline parallel
        self.verbose = True             # Print detailed logs

        # Performance
        self.warmup_runs = 0            # Warmup iterations
        self.measure_runs = 1           # Measurement iterations
```

### Model Export Settings

Modify in `model_manager.py`:

```python
class ModelManager:
    def _export_model(self, ...):
        # Dataset-specific parameters
        num_nodes = 89250               # Flickr default
        num_edges = 899756
        num_features = 500

        # Export settings
        dynamic = True                  # Dynamic shapes
```

---

## Advanced Usage

### Custom Dataset

```python
executor = PipelineExecutor(
    compilation_result_path='result.json',
    dataset_name='reddit'  # or 'yelp', 'amazon', etc.
)

# Update model export parameters
executor.model_manager.export_params = {
    'num_nodes': 232965,
    'num_edges': 11606919,
    'num_features': 602
}
```

### Manual Model Export

```python
from executer.model_export_utils import export_model_for_pep_block

pep_block = {
    'devices': ['CPU', 'GPU'],
    'stages': [1, 2, 3, 4, 5, 6, 7],
    'ratios': [0.5, 0.5]
}

export_model_for_pep_block(
    pep_block=pep_block,
    output_path='my_model.onnx',
    num_nodes=10000,
    num_edges=50000,
    num_features=128
)
```

### Verify Exported Models

```python
import onnxruntime as ort

session = ort.InferenceSession('model.onnx')

# Check inputs
for inp in session.get_inputs():
    print(f"Input: {inp.name}, shape: {inp.shape}, type: {inp.type}")

# Check outputs
for out in session.get_outputs():
    print(f"Output: {out.name}, shape: {out.shape}, type: {out.type}")
```

---

## Performance Optimization

### 1. Ghost Node Strategy

Current: **Pre-collection**
- Pros: No runtime communication, simple
- Cons: Memory overhead (~212%)

Future: On-demand collection
```python
handler = GhostNodeHandler(data_loader, strategy='on_demand')
```

### 2. Model Caching

Models are automatically cached after first load:
```python
manager.load_models()  # First load: ~1s
model = manager.get_model(0, 'CPU')  # Cached: <1ms
```

### 3. ONNX Runtime Optimization

```python
import onnxruntime as ort

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 4  # Parallel ops

session = ort.InferenceSession(
    model_path,
    sess_options=session_options
)
```

### 4. GPU Execution

Ensure CUDA provider is available:
```python
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)

# Verify GPU usage
assert 'CUDAExecutionProvider' in session.get_providers()
```

---

## Troubleshooting

### Model Export Fails

**Error**: `RuntimeError: Exported model is too small (126 bytes)`

**Cause**: Model export failed, produced placeholder only

**Solution**:
```python
# Check dependencies
python -c "import torch, torch_scatter; print('OK')"

# Manually export
from executer.model_export_utils import SimpleModelExporter
exporter = SimpleModelExporter()
exporter.export_combined_model(...)
```

### Ghost Node Features Mismatch

**Error**: `RuntimeError: Ghost features shape mismatch`

**Cause**: Partition map doesn't match actual graph

**Solution**:
1. Re-run compiler to regenerate partition
2. Verify partition_config matches dataset
3. Check for dataset version mismatch

### ONNX Runtime Error

**Error**: `Segmentation fault` or `Invalid model`

**Solution**:
```bash
# Reinstall ONNX Runtime
pip uninstall onnxruntime
pip install onnxruntime

# Or use specific version
pip install onnxruntime==1.15.1
```

### Out of Memory

**Error**: `CUDA out of memory` or `RuntimeError: out of memory`

**Solution**:
- Reduce batch size (fewer subgraphs)
- Use CPU-only execution
- Increase system memory
- Clear cache: `torch.cuda.empty_cache()`

---

## Testing

### Unit Tests

```bash
# Test individual components
python -c "from executer.data_loader import GraphDataLoader; print('OK')"
python -c "from executer.model_export_utils import SimpleModelExporter; print('OK')"
```

### Integration Test

```bash
# Full pipeline
python test_executor.py

# Check outputs
ls -lh ../compiler/output/models/*.onnx
```

### Validation Script

```python
import torch
from executer import PipelineExecutor

executor = PipelineExecutor(
    compilation_result_path='../compiler/output/compilation_result.json',
    dataset_name='flickr'
)

executor.prepare()
result = executor.execute()

# Validate output shape
assert result['embeddings'].shape == (89250, 256), "Shape mismatch"

# Validate no NaN/Inf
assert not torch.isnan(result['embeddings']).any(), "NaN in output"
assert not torch.isinf(result['embeddings']).any(), "Inf in output"

# Validate timing
assert result['total_time'] > 0, "Invalid timing"

print("✓ All validations passed")
```

---

## API Reference

### PipelineExecutor

Main executor class.

```python
class PipelineExecutor:
    def __init__(self, compilation_result_path: str, dataset_name: str):
        """Initialize executor

        Args:
            compilation_result_path: Path to compilation_result.json
            dataset_name: Dataset name (flickr, reddit, yelp, etc.)
        """

    def prepare(self):
        """Prepare for execution

        Steps:
            1. Load graph data
            2. Collect ghost features
            3. Export and load models
            4. Create subgraph executors
        """

    def execute(self) -> dict:
        """Execute complete pipeline

        Returns:
            dict: {
                'embeddings': torch.Tensor [num_nodes, output_dim],
                'total_time': float,
                'subgraph_times': list,
                'avg_subgraph_time': float
            }
        """
```

### SubgraphExecutor

Single subgraph executor.

```python
class SubgraphExecutor:
    def execute(self, edge_index: torch.Tensor, x: torch.Tensor,
                owned_nodes: list) -> Tuple[torch.Tensor, float]:
        """Execute subgraph inference

        Args:
            edge_index: Edge indices [2, num_edges]
            x: Node features [num_nodes, feat_dim]
            owned_nodes: List of owned node IDs

        Returns:
            tuple: (output embeddings, execution time)
        """
```

### GraphDataLoader

Graph data loading and partitioning.

```python
class GraphDataLoader:
    def get_subgraph_data(self, subgraph_id: int) -> dict:
        """Get data for a specific subgraph

        Args:
            subgraph_id: Subgraph ID

        Returns:
            dict: {
                'edge_index': torch.Tensor,
                'num_nodes': int,
                'owned_nodes': list,
                'ghost_nodes': list,
                'node_mapping': dict
            }
        """
```

### ModelManager

Model export, load, and caching.

```python
class ModelManager:
    def ensure_models_exist(self):
        """Check and export models if needed"""

    def load_models(self):
        """Load and compile all models"""

    def get_model(self, block_id: int, device: str):
        """Get compiled model

        Args:
            block_id: Block ID in PEP
            device: 'CPU', 'GPU', or 'NPU'

        Returns:
            onnxruntime.InferenceSession
        """
```

---

## Migration from executor copy/

The executer is now **fully standalone** and does not require the `executor copy/` directory.

**What changed**:
- ✅ Created `model_export_utils.py` with all necessary functions
- ✅ Removed dependency on `pep_model_exporter.py`
- ✅ Self-contained 7-stage model definitions
- ✅ Simplified ONNX export pipeline

See `STANDALONE_MIGRATION.md` for details.

---

## Future Enhancements

### Phase 2: Pipeline Parallelism

Currently: Sequential execution
```python
for sg in subgraphs:
    execute(sg)  # One at a time
```

Future: Concurrent execution
```python
# Different blocks execute in parallel on different devices
Thread1 (CPU): [Block0, SG0] -> [Block0, SG1] -> ...
Thread2 (GPU): [Block1, SG0] -> [Block1, SG1] -> ...
Thread3 (NPU): [Block2, SG0] -> [Block2, SG1] -> ...
```

### Advanced Features

- [ ] Asynchronous model execution
- [ ] Dynamic load balancing
- [ ] Adaptive ghost node collection
- [ ] Multi-GPU support
- [ ] NPU with OpenVINO IR
- [ ] Profiling and performance analysis

---

## Citation

If you use this executor in your research, please cite:

```bibtex
@inproceedings{gnn-executor-2024,
  title={Efficient GNN Execution on Heterogeneous Hardware},
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
