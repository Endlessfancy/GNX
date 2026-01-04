(GNX) PS D:\Research\GNX\executer> python .\test_pipeline_execution.py
================================================================================
Pipeline Execution Test
================================================================================

[1/4] Creating custom execution plan...
  Created 2 clusters:
    Cluster 0: custom_pep_0
      PEP: [[['CPU', 'GPU'], [1, 2, 3, 4, 5], [0.5, 0.5]], [['NPU'], [6, 7]]]
      Subgraphs: [0, 1, 2, 3, 4, 5, 6, 7]
    Cluster 1: custom_pep_1
      PEP: [[['CPU'], [1, 2, 3, 4]], [['GPU', 'NPU'], [5, 6, 7], [0.7, 0.3]]]
      Subgraphs: [8, 9, 10, 11, 12, 13, 14, 15]

[2/4] Initializing executor...
  ✓ Executor initialized

[3/4] Preparing executor (loading data and models)...
  Note: This may take a while on first run (model export)

Preparing executor...

[Step 1/4] Loading graph data...
  Loaded Flickr dataset:
  Nodes: 89,250
  Edges: 899,756
  Features: 500
  ✓ Graph loaded: 89,250 nodes, 899,756 edges
  ✓ Partitioned into 16 subgraphs

[Step 2/4] Collecting ghost node features...
  ✓ Ghost features collected: 440,592 total ghost nodes

[Step 3/4] Exporting and loading models...
  ✓ Model manager initialized
    Unique models needed: 6
    Models directory: D:\Research\GNX\executer\models

  Checking model files...
    ✓ Model exists: cluster_0_block_0_CPU (507.3 KB)
    ✓ Model exists: cluster_0_block_0_GPU (507.3 KB)
    ✓ Model exists: cluster_0_block_1_NPU (757.8 KB)
    ✓ Model exists: cluster_1_block_0_CPU (507.0 KB)
    ✓ Model exists: cluster_1_block_1_GPU (758.5 KB)
    ✓ Model exists: cluster_1_block_1_NPU (758.5 KB)

  Loading and compiling models...
    Loading cluster_0_block_0_CPU...
      ✓ Compiled for CPUExecutionProvider
    Loading cluster_0_block_0_GPU...
      ✓ Compiled for CPUExecutionProvider
    Loading cluster_0_block_1_NPU...
      ✓ Compiled for CPUExecutionProvider
    Loading cluster_1_block_0_CPU...
      ✓ Compiled for CPUExecutionProvider
    Loading cluster_1_block_1_GPU...
      ✓ Compiled for CPUExecutionProvider
    Loading cluster_1_block_1_NPU...
      ✓ Compiled for CPUExecutionProvider
  ✓ All models loaded: 6

[Step 4/4] Creating subgraph executors...
  ✓ Created 16 subgraph executors

✓ Preparation complete!

  ✓ Preparation complete

[4/4] Executing pipeline...

======================================================================
Cluster 0: custom_pep_0
  PEP: [[['CPU', 'GPU'], [1, 2, 3, 4, 5], [0.5, 0.5]], [['NPU'], [6, 7]]]
  Subgraphs: [0, 1, 2, 3, 4, 5, 6, 7]
======================================================================

  Subgraph 0... 1221.24ms
  Subgraph 1... 792.62ms
  Subgraph 2... 611.35ms
  Subgraph 3... 470.92ms
  Subgraph 4... 531.50ms
  Subgraph 5... 508.97ms
  Subgraph 6... 483.73ms
  Subgraph 7... 349.20ms

✓ Cluster 0 completed in 16835.13ms


======================================================================
Cluster 1: custom_pep_1
  PEP: [[['CPU'], [1, 2, 3, 4]], [['GPU', 'NPU'], [5, 6, 7], [0.7, 0.3]]]
  Subgraphs: [8, 9, 10, 11, 12, 13, 14, 15]
======================================================================

  Subgraph 8... 538.05ms
  Subgraph 9... 329.12ms
  Subgraph 10... 311.61ms
  Subgraph 11... 222.02ms
  Subgraph 12... 535.56ms
  Subgraph 13... 340.50ms
  Subgraph 14... 326.88ms
  Subgraph 15... 349.03ms

✓ Cluster 1 completed in 11784.55ms


✓ All clusters executed
  Total time: 29838.77ms
  ✓ Execution complete

================================================================================
Execution Results
================================================================================
Total time: 29838.77ms
Output embeddings shape: torch.Size([89250, 256])

Per-cluster times:
  Cluster 0: 16835.13ms
  Cluster 1: 11784.55ms

Per-subgraph times (first 5):
  Subgraph 0: 1221.24ms
  Subgraph 1: 792.62ms
  Subgraph 2: 611.35ms
  Subgraph 3: 470.92ms
  Subgraph 4: 531.50ms

✓ Pipeline execution test completed successfully!
(GNX) PS D:\Research\GNX\executer>