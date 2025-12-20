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
    Unique models needed: 0
    Models directory: D:\Research\GNX\executer\models

  Checking model files...

  Loading and compiling models...
  ✓ All models loaded: 0

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

  Subgraph 0...   ✗ Execution failed: (0, 'CPU')
Traceback (most recent call last):
  File "D:\Research\GNX\executer\test_pipeline_execution.py", line 63, in main
    result = executor.execute()
  File "D:\Research\GNX\executer\executor.py", line 196, in execute
    embeddings, sg_time = executor.execute(edge_index, x, owned_nodes)
  File "D:\Research\GNX\executer\subgraph_executor.py", line 72, in execute
    current_data = self._execute_block(
  File "D:\Research\GNX\executer\subgraph_executor.py", line 122, in _execute_block
    output_data = self._execute_data_parallel(
  File "D:\Research\GNX\executer\subgraph_executor.py", line 224, in _execute_data_parallel
    model = self.models[(block_id, device)]
KeyError: (0, 'CPU')