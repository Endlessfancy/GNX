(GNX) PS D:\Research\GNX\executer> python .\test_gpu_simple.py
======================================================================
Simple GPU Test
======================================================================

======================================================================
TEST 1: Linear Model (Stage 6 only)
======================================================================

============================================================
Testing: linear_100
  Data size: 100 nodes, 0 edges
============================================================
  ✓ ONNX exported: 501.2 KB
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\openvino\runtime\__init__.py:10: DeprecationWarning: The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release. Please replace `openvino.runtime` with `openvino`.
  warnings.warn(
  Available devices: ['CPU', 'GPU', 'NPU']
  ✓ IR converted

  Testing CPU inference...
  ✓ CPU inference OK, output shape: (100, 256)

  Testing GPU inference...
75 warnings generated.
  ✓ GPU compilation OK
  ✓ GPU inference OK, output shape: (100, 256)
  ✓ CPU vs GPU max diff: 0.000000

============================================================
Testing: linear_1000
  Data size: 1000 nodes, 0 edges
============================================================
  ✓ ONNX exported: 501.2 KB
  Available devices: ['CPU', 'GPU', 'NPU']
  ✓ IR converted

  Testing CPU inference...
  ✓ CPU inference OK, output shape: (1000, 256)

  Testing GPU inference...
75 warnings generated.
  ✓ GPU compilation OK
  ✓ GPU inference OK, output shape: (1000, 256)
  ✓ CPU vs GPU max diff: 0.000000

============================================================
Testing: linear_5000
  Data size: 5000 nodes, 0 edges
============================================================
  ✓ ONNX exported: 501.2 KB
  Available devices: ['CPU', 'GPU', 'NPU']
  ✓ IR converted

  Testing CPU inference...
  ✓ CPU inference OK, output shape: (5000, 256)

  Testing GPU inference...
75 warnings generated.
  ✓ GPU compilation OK
  ✓ GPU inference OK, output shape: (5000, 256)
  ✓ CPU vs GPU max diff: 0.000000

============================================================
Testing: linear_10000
  Data size: 10000 nodes, 0 edges
============================================================
  ✓ ONNX exported: 501.2 KB
  Available devices: ['CPU', 'GPU', 'NPU']
  ✓ IR converted

  Testing CPU inference...
  ✓ CPU inference OK, output shape: (10000, 256)

  Testing GPU inference...
75 warnings generated.
  ✓ GPU compilation OK
  ✓ GPU inference OK, output shape: (10000, 256)
  ✓ CPU vs GPU max diff: 0.000000

============================================================
Testing: linear_50000
  Data size: 50000 nodes, 0 edges
============================================================
  ✓ ONNX exported: 501.2 KB
  Available devices: ['CPU', 'GPU', 'NPU']
  ✓ IR converted

  Testing CPU inference...
  ✓ CPU inference OK, output shape: (50000, 256)

  Testing GPU inference...
75 warnings generated.
  ✓ GPU compilation OK
  ✓ GPU inference OK, output shape: (50000, 256)
  ✓ CPU vs GPU max diff: 0.000000

======================================================================
TEST 2: Gather Model (Stage 1-2)
======================================================================

============================================================
Testing: gather_100_500
  Data size: 100 nodes, 500 edges
============================================================
  ✓ ONNX exported: 0.3 KB
  Available devices: ['CPU', 'GPU', 'NPU']
  ✓ IR converted

  Testing CPU inference...
  ✓ CPU inference OK, output shape: (500, 500)

  Testing GPU inference...
75 warnings generated.
  ✓ GPU compilation OK
  ✓ GPU inference OK, output shape: (500, 500)
  ✓ CPU vs GPU max diff: 0.000000

============================================================
Testing: gather_1000_5000
  Data size: 1000 nodes, 5000 edges
============================================================
  ✓ ONNX exported: 0.3 KB
  Available devices: ['CPU', 'GPU', 'NPU']
  ✓ IR converted

  Testing CPU inference...
  ✓ CPU inference OK, output shape: (5000, 500)

  Testing GPU inference...
75 warnings generated.
  ✓ GPU compilation OK
  ✓ GPU inference OK, output shape: (5000, 500)
  ✓ CPU vs GPU max diff: 0.000000

============================================================
Testing: gather_5000_25000
  Data size: 5000 nodes, 25000 edges
============================================================
  ✓ ONNX exported: 0.3 KB
  Available devices: ['CPU', 'GPU', 'NPU']
  ✓ IR converted

  Testing CPU inference...
  ✓ CPU inference OK, output shape: (25000, 500)

  Testing GPU inference...
75 warnings generated.
  ✓ GPU compilation OK
  ✓ GPU inference OK, output shape: (25000, 500)
  ✓ CPU vs GPU max diff: 0.000000

============================================================
Testing: gather_10000_50000
  Data size: 10000 nodes, 50000 edges
============================================================
  ✓ ONNX exported: 0.3 KB
  Available devices: ['CPU', 'GPU', 'NPU']
  ✓ IR converted

  Testing CPU inference...
  ✓ CPU inference OK, output shape: (50000, 500)

  Testing GPU inference...
75 warnings generated.
  ✓ GPU compilation OK
  ✓ GPU inference OK, output shape: (50000, 500)
  ✓ CPU vs GPU max diff: 0.000000

============================================================
Testing: gather_50000_200000
  Data size: 50000 nodes, 200000 edges
============================================================
  ✓ ONNX exported: 0.3 KB
  Available devices: ['CPU', 'GPU', 'NPU']
  ✓ IR converted

  Testing CPU inference...
  ✓ CPU inference OK, output shape: (200000, 500)

  Testing GPU inference...
75 warnings generated.
  ✓ GPU compilation OK
  ✓ GPU inference OK, output shape: (200000, 500)
  ✓ CPU vs GPU max diff: 0.000000

======================================================================
TEST 3: Scatter Model (Stage 3)
======================================================================

============================================================
Testing: scatter_100_500
  Data size: 100 nodes, 500 edges
============================================================
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\onnx\utils.py:616: UserWarning: ONNX Preprocess - Removing mutation from node aten::index_add_ on block input: '0'. This changes graph semantics. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\torch\csrc\jit\passes\onnx\remove_inplace_ops_for_onnx.cpp:354.)
  _C._jit_pass_onnx_remove_inplace_ops_for_onnx(graph, module)
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\onnx\symbolic_opset9.py:6075: UserWarning: Warning: ONNX export does not support duplicated values in 'index' field, this will cause the ONNX model to be incorrect.
  warnings.warn(
  ✗ ONNX export failed: ONNX export does not support exporting 'index_add_()' function with duplicated values in 'index' parameter yet.  [Caused by the value '18 defined in (%18 : Float(100, 500, strides=[500, 1], requires_grad=0, device=cpu) = onnx::Constant[value=<Tensor>](), scope: __main__.SimpleScatterModel:: # D:\Research\GNX\executer\test_gpu_simple.py:49:0
)' (type 'Tensor') in the TorchScript graph. The containing node has kind 'onnx::Constant'.]
    (node defined in D:\Research\GNX\executer\test_gpu_simple.py(49): forward
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\nn\modules\module.py(1726): _slow_forward
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\nn\modules\module.py(1747): _call_impl
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\nn\modules\module.py(1736): _wrapped_call_impl
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\jit\_trace.py(130): wrapper
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\jit\_trace.py(139): forward
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\nn\modules\module.py(1747): _call_impl
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\nn\modules\module.py(1736): _wrapped_call_impl
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\jit\_trace.py(1500): _get_trace_graph
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\onnx\utils.py(904): _trace_and_get_graph_from_model
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\onnx\utils.py(997): _create_jit_graph
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\onnx\utils.py(1113): _model_to_graph
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\onnx\utils.py(1564): _export
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\onnx\utils.py(502): export
C:\Users\29067\miniconda3\envs\GNX\lib\site-packages\torch\onnx\__init__.py(375): export
D:\Research\GNX\executer\test_gpu_simple.py(82): export_and_test_model
D:\Research\GNX\executer\test_gpu_simple.py(193): test_scatter_model
D:\Research\GNX\executer\test_gpu_simple.py(254): main
D:\Research\GNX\executer\test_gpu_simple.py(278): <module>
)

    Inputs:
        Empty
    Outputs:
        #0: 18 defined in (%18 : Float(100, 500, strides=[500, 1], requires_grad=0, device=cpu) = onnx::Constant[value=<Tensor>](), scope: __main__.SimpleScatterModel:: # D:\Research\GNX\executer\test_gpu_simple.py:49:0
    )  (type 'Tensor')
  ⚠ Scatter model failed at 100 nodes, 500 edges

======================================================================
SUMMARY
======================================================================
  Linear     @    100 nodes: ✓ PASS
  Linear     @   1000 nodes: ✓ PASS
  Linear     @   5000 nodes: ✓ PASS
  Linear     @  10000 nodes: ✓ PASS
  Linear     @  50000 nodes: ✓ PASS
  Gather     @    100 nodes: ✓ PASS
  Gather     @   1000 nodes: ✓ PASS
  Gather     @   5000 nodes: ✓ PASS
  Gather     @  10000 nodes: ✓ PASS
  Gather     @  50000 nodes: ✓ PASS
  Scatter    @    100 nodes: ✗ FAIL

⚠ 1 test(s) failed
  First failure: ('Scatter', 100, False)