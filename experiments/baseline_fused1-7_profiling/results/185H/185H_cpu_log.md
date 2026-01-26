========================================================================
Fused Block 0-7 Baseline - CPU + GPU
========================================================================

Exporting models...
======================================================================
Exporting FusedBlock0_7 Models (Stage 1-7 Combined)
======================================================================

Exporting to ONNX: fused_block0_7.onnx
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\symbolic_opset9.py:6628: UserWarning: Warning: ONNX export does not support duplicated values in 'index' field, this will cause the ONNX model to be incorrect.
  warnings.warn(
  ONNX export: OK
Converting to CPU IR: fused_block0_7_cpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  CPU IR: OK
Converting to GPU IR: fused_block0_7_gpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  GPU IR: OK

Models exported: 2 files

Measuring CPU and GPU latencies...

======================================================================
Measuring CPU/GPU Latencies (FusedBlock0_7)
======================================================================

[CPU]
  [5000n, 50000e]... 59.20ms
  [5000n, 125000e]... 111.44ms
  [5000n, 200000e]... 389.46ms
  [5000n, 250000e]... 405.21ms
  [5000n, 300000e]... 770.30ms
  [5000n, 375000e]... 476.31ms
  [5000n, 500000e]... 844.63ms
  [10000n, 100000e]... 106.21ms
  [10000n, 250000e]... 460.91ms
  [10000n, 400000e]... 547.18ms
  [10000n, 500000e]... 686.55ms
  [10000n, 600000e]... 847.18ms
  [10000n, 750000e]... 1214.26ms
  [10000n, 1000000e]... 1287.86ms
  [20000n, 200000e]... 286.03ms
  [20000n, 500000e]... 725.26ms
  [20000n, 800000e]... 1128.51ms
  [20000n, 1000000e]... 1471.03ms
  [20000n, 1200000e]... 1947.48ms
  [20000n, 1500000e]... 2161.41ms
  [20000n, 2000000e]... 2561.94ms
  [50000n, 500000e]... 816.17ms
  [50000n, 1250000e]... 2129.88ms
  [50000n, 2000000e]... 2840.28ms
  [50000n, 2500000e]... 2992.09ms
  [50000n, 3000000e]... 4555.16ms
  [50000n, 3750000e]... 5387.27ms
  [50000n, 5000000e]... 6015.67ms
  [80000n, 800000e]... 1279.50ms
  [80000n, 2000000e]... 3126.54ms
  [80000n, 3200000e]... 3730.72ms
  [80000n, 4000000e]... 6274.10ms
  [80000n, 4800000e]... 6235.15ms
  [80000n, 6000000e]... 7102.58ms
  [80000n, 8000000e]... 9430.59ms
  [100000n, 1000000e]... 1781.67ms
  [100000n, 2500000e]... 3233.63ms
  [100000n, 4000000e]... 4715.21ms
  [100000n, 5000000e]... 6526.45ms
  [100000n, 6000000e]... 7771.45ms
  [100000n, 7500000e]... 8502.92ms
  [100000n, 10000000e]... 
