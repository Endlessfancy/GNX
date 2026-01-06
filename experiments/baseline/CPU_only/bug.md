(MIX) PS C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only> python .\cpu_batch_4096_baseline.py --device NPU
================================================================================
NPU Batch-4096 Baseline Test (1 Layer, 1-hop) - OpenVINO
================================================================================

Loading Flickr dataset...
  Nodes: 89,250
  Edges: 899,756
  Features: 500

Partitioning 89,250 nodes into batches of 4096...
  Expected batches: 22
  Batch 1/22: target=4,096, halo=60,663, total=64,759, edges=618,876
  Batch 2/22: target=4,096, halo=43,870, total=47,966, edges=426,540
  Batch 3/22: target=4,096, halo=35,279, total=39,375, edges=332,844
  Batch 4/22: target=4,096, halo=29,835, total=33,931, edges=277,624
  Batch 5/22: target=4,096, halo=27,672, total=31,768, edges=255,618
  Batch 6/22: target=4,096, halo=24,685, total=28,781, edges=228,462
  Batch 7/22: target=4,096, halo=23,327, total=27,423, edges=214,524
  Batch 8/22: target=4,096, halo=22,178, total=26,274, edges=204,478
  Batch 9/22: target=4,096, halo=20,688, total=24,784, edges=191,408
  Batch 10/22: target=4,096, halo=20,069, total=24,165, edges=185,456
  Batch 11/22: target=4,096, halo=18,812, total=22,908, edges=174,022
  Batch 12/22: target=4,096, halo=18,589, total=22,685, edges=172,422
  Batch 13/22: target=4,096, halo=17,455, total=21,551, edges=161,304
  Batch 14/22: target=4,096, halo=16,966, total=21,062, edges=157,548
  Batch 15/22: target=4,096, halo=16,021, total=20,117, edges=150,008
  Batch 16/22: target=4,096, halo=15,907, total=20,003, edges=150,044
  Batch 17/22: target=4,096, halo=15,479, total=19,575, edges=145,752
  Batch 18/22: target=4,096, halo=14,965, total=19,061, edges=142,904
  Batch 19/22: target=4,096, halo=14,459, total=18,555, edges=137,708
  Batch 20/22: target=4,096, halo=13,952, total=18,048, edges=134,254
  Batch 21/22: target=4,096, halo=13,465, total=17,561, edges=130,658
  Batch 22/22: target=3,234, halo=10,535, total=13,769, edges=96,832

Exporting 22 static ONNX models for NPU...
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\utils.py:1547: OnnxExporterWarning: Exporting to ONNX opset version 18 is not supported. by 'torch.onnx.export()'. The highest opset version supported is 17. To use a newer opset version, consider 'torch.onnx.dynamo_export()'. Note that dynamo_export() is in preview. Please report errors with dynamo_export() as Github issues to https://github.com/pytorch/pytorch/issues.
  warnings.warn(
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\symbolic_opset9.py:6628: UserWarning: Warning: ONNX export does not support duplicated values in 'index' field, this will cause the ONNX model to be incorrect.
  warnings.warn(
Traceback (most recent call last):
  File "C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\cpu_batch_4096_baseline.py", line 557, in <module>
    results = run_batch_baseline(
  File "C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\cpu_batch_4096_baseline.py", line 341, in run_batch_baseline
    export_onnx_model(
  File "C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\cpu_batch_4096_baseline.py", line 179, in export_onnx_model
    torch.onnx.export(
  File "C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\utils.py", line 516, in export
    _export(
  File "C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\utils.py", line 1612, in _export
    graph, params_dict, torch_out = _model_to_graph(
  File "C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\utils.py", line 1138, in _model_to_graph
    graph = _optimize_graph(
  File "C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\utils.py", line 677, in _optimize_graph
    graph = _C._jit_pass_onnx(graph, operator_export_type)
  File "C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\utils.py", line 1956, in _run_symbolic_function
    return symbolic_fn(graph_context, *inputs, **attrs)
  File "C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\symbolic_opset9.py", line 6668, in index_add
    raise errors.SymbolicValueError(
torch.onnx.errors.SymbolicValueError: ONNX export does not support exporting 'index_add_()' function with duplicated values in 'index' parameter yet.  [Caused by the value '32 defined in (%32 : Float(64759, 500, strides=[500, 1], requires_grad=0, device=cpu) = onnx::Constant[value=<Tensor>](), scope: __main__.GraphSAGEFullModel::/__main__.SAGEStage3_ReduceSum::stage3 # C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\cpu_batch_4096_baseline.py:57:0
)' (type 'Tensor') in the TorchScript graph. The containing node has kind 'onnx::Constant'.]
    (node defined in C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\cpu_batch_4096_baseline.py(57): forward
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\nn\modules\module.py(1522): _slow_forward
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\nn\modules\module.py(1541): _call_impl
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\nn\modules\module.py(1532): _wrapped_call_impl
C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\cpu_batch_4096_baseline.py(138): forward
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\nn\modules\module.py(1522): _slow_forward
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\nn\modules\module.py(1541): _call_impl
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\nn\modules\module.py(1532): _wrapped_call_impl
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\jit\_trace.py(129): wrapper
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\jit\_trace.py(138): forward
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\nn\modules\module.py(1541): _call_impl
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\nn\modules\module.py(1532): _wrapped_call_impl
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\jit\_trace.py(1310): _get_trace_graph
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\utils.py(914): _trace_and_get_graph_from_model
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\utils.py(1010): _create_jit_graph
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\utils.py(1134): _model_to_graph
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\utils.py(1612): _export
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\utils.py(516): export
C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\cpu_batch_4096_baseline.py(179): export_onnx_model
C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\cpu_batch_4096_baseline.py(341): run_batch_baseline
C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\cpu_batch_4096_baseline.py(557): <module>
)

    Inputs:
        Empty
    Outputs:
        #0: 32 defined in (%32 : Float(64759, 500, strides=[500, 1], requires_grad=0, device=cpu) = onnx::Constant[value=<Tensor>](), scope: __main__.GraphSAGEFullModel::/__main__.SAGEStage3_ReduceSum::stage3 # C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\cpu_batch_4096_baseline.py:57:0
    )  (type 'Tensor')