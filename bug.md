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
    Unique models needed: 4
    Models directory: D:\Research\GNX\executer\models

  Checking model files...
    Model missing: block_0_CPU
      Exporting CPU model for stages [1, 2, 3, 4, 5]...
  Exporting CPU model (stages [1, 2, 3, 4, 5]) to D:\Research\GNX\executer\models\CPU_stages_1_2_3_4.onnx
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\onnx\symbolic_opset9.py:6075: UserWarning: Warning: ONNX export does not support duplicated values in 'index' field, this will cause the ONNX model to be incorrect.
  warnings.warn(
  ✓ Model exported successfully (0.01 MB)
  ⚠ Verification failed: Required inputs (['x', 'edge_index']) are missing from input feed (['input_0', 'input_1']).
      ✓ Exported: 7.0 KB
    Model missing: block_0_GPU
      Exporting GPU model for stages [1, 2, 3, 4, 5]...
  Exporting GPU model (stages [1, 2, 3, 4, 5]) to D:\Research\GNX\executer\models\GPU_stages_1_2_3_4_5.onnx
  ✓ Model exported successfully (0.01 MB)
  ⚠ Verification failed: Required inputs (['x', 'edge_index']) are missing from input feed (['input_0', 'input_1']).
      ✓ Exported: 7.0 KB
    Model missing: block_1_NPU
      Exporting NPU model for stages [6, 7]...
  Exporting NPU model (stages [6, 7]) to D:\Research\GNX\executer\models\NPU_stages_5_6_7.onnx
  ✓ Model exported successfully (0.74 MB)
  ⚠ Verification failed: local variable 'dummy_edge_index' referenced before assignment
      ✓ Exported: 757.8 KB
    Model missing: block_1_GPU
      Exporting GPU model for stages [5, 6, 7]...
  Exporting GPU model (stages [5, 6, 7]) to D:\Research\GNX\executer\models\GPU_stages_5_6_7.onnx
  ✓ Model exported successfully (0.74 MB)
  ⚠ Verification failed: local variable 'dummy_edge_index' referenced before assignment
      ✓ Exported: 758.5 KB

  Loading and compiling models...
    Loading block_0_CPU...
      ✓ Compiled for CPUExecutionProvider
    Loading block_0_GPU...
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:123: UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
  warnings.warn(
      ✓ Compiled for CUDAExecutionProvider
    Loading block_1_NPU...
      ✓ Compiled for CPUExecutionProvider
    Loading block_1_GPU...
      ✓ Compiled for CUDAExecutionProvider
  ✓ All models loaded: 4

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

  Subgraph 0...   ✗ Execution failed: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: mean_agg for the following indices
 index: 1 Got: 500 Expected: 256
 Please fix either the inputs/outputs or the model.
Traceback (most recent call last):
  File "D:\Research\GNX\executer\test_pipeline_execution.py", line 63, in main
    result = executor.execute()
  File "D:\Research\GNX\executer\executor.py", line 196, in execute
    embeddings, sg_time = executor.execute(edge_index, x, owned_nodes)
  File "D:\Research\GNX\executer\subgraph_executor.py", line 72, in execute
    current_data = self._execute_block(
  File "D:\Research\GNX\executer\subgraph_executor.py", line 114, in _execute_block
    output_data = self._execute_single_device(
  File "D:\Research\GNX\executer\subgraph_executor.py", line 168, in _execute_single_device
    outputs = model.run(None, input_dict)
  File "C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 287, in run
    return self._sess.run(output_names, input_feed, run_options)
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: mean_agg for the following indices
 index: 1 Got: 500 Expected: 256
 Please fix either the inputs/outputs or the model.