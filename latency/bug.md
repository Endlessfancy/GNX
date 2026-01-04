(MIX) PS C:\Private\Research\GNX_final\GNX\latency> python .\test_sequential_latency.py --pep pep2
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
  Block 0: [CPU] Stages [1,2,3,4]
  Block 1: [GPU+NPU] Stages [5,6,7] (GPU:70%, NPU:30%)
------------------------------------------------------------

Exporting models...

============================================================
Exporting models for PEP configuration
============================================================

Block 0: stages [1, 2, 3, 4]

Exporting stages [1, 2, 3, 4] for CPU:
  Exporting CPU model (stages [1, 2, 3, 4]) to C:\Private\Research\GNX_final\GNX\latency\models\stages_1_2_3_4_CPU.onnx
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\utils.py:1547: OnnxExporterWarning: Exporting to ONNX opset version 18 is not supported. by 'torch.onnx.export()'. The highest opset version supported is 17. To use a newer opset version, consider 'torch.onnx.dynamo_export()'. Note that dynamo_export() is in preview. Please report errors with dynamo_export() as Github issues to https://github.com/pytorch/pytorch/issues.
  warnings.warn(
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\symbolic_opset9.py:6628: UserWarning: Warning: ONNX export does not support duplicated values in 'index' field, this will cause the ONNX model to be incorrect.
  warnings.warn(
  ✓ Model exported successfully (0.01 MB)
  Converting to IR: stages_1_2_3_4_CPU_dynamic.xml
  IR saved: stages_1_2_3_4_CPU_dynamic.xml

Block 1: stages [5, 6, 7]

Exporting stages [5, 6, 7] for GPU:
  Exporting GPU model (stages [5, 6, 7]) to C:\Private\Research\GNX_final\GNX\latency\models\stages_5_6_7_GPU.onnx
  ✓ Model exported successfully (0.98 MB)
  Converting to IR: stages_5_6_7_GPU_dynamic.xml
  IR saved: stages_5_6_7_GPU_dynamic.xml

Exporting stages [5, 6, 7] for NPU:
  Exporting NPU model (stages [5, 6, 7]) to C:\Private\Research\GNX_final\GNX\latency\models\stages_5_6_7_NPU.onnx
  ✓ Model exported successfully (0.98 MB)
  Converting to IR: stages_5_6_7_NPU_n89518_e307822.xml
  IR saved: stages_5_6_7_NPU_n89518_e307822.xml

============================================================
Exported 3 models
============================================================

Creating stage executors...
  Loading Block0_S1_2_3_4_CPU on CPU (stream 0)...
    Compiled successfully on CPU
  Created executor: Block0_S1_2_3_4_CPU
  Created stage: Block0_S1_2_3_4 with devices ['CPU']
  Loading Block1_S5_6_7_GPU on GPU (stream 0)...
    Compiled successfully on GPU
  Created executor: Block1_S5_6_7_GPU
  Loading Block1_S5_6_7_NPU on NPU (stream 1)...
    Compiled successfully on NPU
    NPU static shape from filename: n=89518, e=307822
  Created executor: Block1_S5_6_7_NPU
  Created stage: Block1_S5_6_7 with devices ['GPU', 'NPU']

Profiling 8 subgraphs x 3 iterations
Warmup: 1 iterations
Traceback (most recent call last):
  File "C:\Private\Research\GNX_final\GNX\latency\test_sequential_latency.py", line 577, in <module>
    main()
  File "C:\Private\Research\GNX_final\GNX\latency\test_sequential_latency.py", line 547, in main
    df = profiler.run_profiling(
  File "C:\Private\Research\GNX_final\GNX\latency\test_sequential_latency.py", line 414, in run_profiling
    self.profile_subgraph(sg_id, x, edge_index, iter_id=-1)
  File "C:\Private\Research\GNX_final\GNX\latency\test_sequential_latency.py", line 299, in profile_subgraph
    outputs, profiling = executors[device].run(inputs, batch_id=sg_id)
  File "C:\Private\Research\GNX_final\GNX\latency\stage_executor.py", line 265, in run
    self.request.infer()
  File "C:\Env\Anaconda\envs\MIX\lib\site-packages\openvino\runtime\ie_api.py", line 132, in infer
    return OVDict(super().infer(_data_dispatch(
RuntimeError: Exception from src\inference\src\cpp\infer_request.cpp:223:
Check 'pshape.compatible(ov::PartialShape(user_tensor->get_shape())) || is_batched_input(port)' failed at src\plugins\intel_gpu\src\plugin\sync_infer_request.cpp:803:
[GPU] The input tensor size is not equal to model port shape, can't handle input tensor with name: parameter:sum_agg, because model input (shape=[?,500]) and tensor (shape=[81380,501]) are incompatible