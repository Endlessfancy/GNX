(MIX) PS C:\Private\Research\GNX_final\GNX\latency> python .\test_sequential_latency.py --pep pep2 --num-iterations 3
======================================================================
Sequential Profiling for Cost Modeling
======================================================================
PEP: pep2
Subgraphs: 8
Iterations: 3
Warmup: 1
Output: cost_model.csv
Available devices: ['CPU', 'GPU', 'NPU']

Loading Flickr dataset...
Loading Flickr dataset...
  Loaded Flickr dataset:
  Nodes: 89,250
  Edges: 899,756
  Features: 500
Partitioned into 8 subgraphs:
  Max nodes per subgraph: 89518
  Max edges per subgraph: 307822
Max subgraph size: 89518 nodes, 307822 edges

======================================================================
Sequential Profiling: PEP2
======================================================================

PEP2:
------------------------------------------------------------
  Block 0: [CPU+GPU] Stages [1,2,3,4] (CPU:30%, GPU:70%)
  Block 1: [NPU] Stages [5,6,7]
------------------------------------------------------------

Exporting models...

============================================================
Exporting models for PEP configuration
============================================================

Block 0: stages [1, 2, 3, 4]

Exporting stages [1, 2, 3, 4] for CPU:
  ONNX exists: stages_1_2_3_4_CPU.onnx (6.7 KB)
  IR exists: stages_1_2_3_4_CPU_dynamic.xml

Exporting stages [1, 2, 3, 4] for GPU:
  ONNX exists: stages_1_2_3_4_GPU.onnx (6.7 KB)
  IR exists: stages_1_2_3_4_GPU_dynamic.xml

Block 1: stages [5, 6, 7]

Exporting stages [5, 6, 7] for NPU:
  ONNX exists: stages_5_6_7_NPU.onnx (1002.1 KB)
  IR exists: stages_5_6_7_NPU_n89518_e307822.xml

============================================================
Exported 3 models
============================================================

Creating stage executors...
  Loading Block0_S1_2_3_4_CPU on CPU (stream 0)...
    Compiled successfully on CPU
  Created executor: Block0_S1_2_3_4_CPU
  Loading Block0_S1_2_3_4_GPU on GPU (stream 1)...
    Compiled successfully on GPU
  Created executor: Block0_S1_2_3_4_GPU
  Created stage: Block0_S1_2_3_4 with devices ['CPU', 'GPU']
  Loading Block1_S5_6_7_NPU on NPU (stream 0)...
    Compiled successfully on NPU
    NPU static shape from filename: n=89518, e=307822
  Created executor: Block1_S5_6_7_NPU
  Created stage: Block1_S5_6_7 with devices ['NPU']

Profiling 8 subgraphs x 3 iterations
Warmup: 1 iterations
Traceback (most recent call last):
  File "C:\Private\Research\GNX_final\GNX\latency\test_sequential_latency.py", line 609, in <module>
    main()
  File "C:\Private\Research\GNX_final\GNX\latency\test_sequential_latency.py", line 579, in main
    df = profiler.run_profiling(
  File "C:\Private\Research\GNX_final\GNX\latency\test_sequential_latency.py", line 446, in run_profiling
    self.profile_subgraph(sg_id, x, edge_index, iter_id=-1)
  File "C:\Private\Research\GNX_final\GNX\latency\test_sequential_latency.py", line 278, in profile_subgraph
    outputs, profiling = executor.run(inputs, batch_id=sg_id)
  File "C:\Private\Research\GNX_final\GNX\latency\stage_executor.py", line 255, in run
    inputs, padding_info = prepare_npu_inputs(
  File "C:\Private\Research\GNX_final\GNX\latency\npu_utils.py", line 265, in prepare_npu_inputs
    original_size = inputs[first_key].shape[0]
AttributeError: 'NoneType' object has no attribute 'shape'