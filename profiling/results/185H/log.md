========================================================================
GNN Stage Profiling - Intel AI PC
========================================================================

Activating MIX environment...

Environment activated. Running profiling...

This will take approximately 17-20 hours to complete.
The script will:
  1. Export 14 dynamic models (CPU/GPU) + 275 static models (NPU, skip Stage 3/4)
  2. Measure 1045 latencies (55 sizes x 3 PUs x 7 stages, NPU skip Stage 3/4)
  3. Estimate bandwidth and separate compute time
  4. Generate lookup_table.json and bandwidth_table.json

Test cases: 1k-150k nodes (55 combinations)
  - Small:  1k, 5k, 10k nodes
  - Medium: 20k, 30k, 40k, 50k, 60k nodes
  - Large:  80k, 100k, 150k nodes (covers actual subgraph sizes)
  - Edge ratios: 2x, 3x, 5x, 7x, 10x per node size

======================================================================
GNX Stage Profiling - Incremental Pipeline
======================================================================
Test cases: 55
Expected CPU/GPU measurements: 770
Expected NPU measurements: 385
Total measurements: 1155

Execution order:
  Phase 1: CPU/GPU (export → measure → save checkpoint)
  Phase 2: NPU (export → measure → merge results)


======================================================================
PHASE 1: CPU/GPU Processing
======================================================================

[Step 1/6] Exporting CPU/GPU dynamic models...
======================================================================
=== Exporting Dynamic Models (CPU/GPU) ===
======================================================================

Stage 1:
  Exporting ONNX: stage1_dynamic.onnx
  Converting to CPU IR: stage1_cpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  Converting to GPU IR: stage1_gpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  ✓ Stage 1 dynamic models exported

Stage 2:
  Exporting ONNX: stage2_dynamic.onnx
  Converting to CPU IR: stage2_cpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  Converting to GPU IR: stage2_gpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  ✓ Stage 2 dynamic models exported

Stage 3:
  Exporting ONNX: stage3_dynamic.onnx
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\utils.py:631: UserWarning: ONNX Preprocess - Removing mutation from node aten::index_add_ on block input: '0'. This changes graph semantics. (Triggered internally at ..\torch\csrc\jit\passes\onnx\remove_inplace_ops_for_onnx.cpp:355.)
  _C._jit_pass_onnx_remove_inplace_ops_for_onnx(graph, module)
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\symbolic_opset9.py:6628: UserWarning: Warning: ONNX export does not support duplicated values in 'index' field, this will cause the ONNX model to be incorrect.
  warnings.warn(
  Converting to CPU IR: stage3_cpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  Converting to GPU IR: stage3_gpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  ✓ Stage 3 dynamic models exported

Stage 4:
  Exporting ONNX: stage4_dynamic.onnx
  Converting to CPU IR: stage4_cpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  Converting to GPU IR: stage4_gpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  ✓ Stage 4 dynamic models exported

Stage 5:
  Exporting ONNX: stage5_dynamic.onnx
  Converting to CPU IR: stage5_cpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  Converting to GPU IR: stage5_gpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  ✓ Stage 5 dynamic models exported

Stage 6:
  Exporting ONNX: stage6_dynamic.onnx
  Converting to CPU IR: stage6_cpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  Converting to GPU IR: stage6_gpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  ✓ Stage 6 dynamic models exported

Stage 7:
  Exporting ONNX: stage7_dynamic.onnx
  Converting to CPU IR: stage7_cpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  Converting to GPU IR: stage7_gpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  ✓ Stage 7 dynamic models exported

✓ All dynamic models exported (14 files)

[Step 2/6] Measuring CPU/GPU latencies...

======================================================================
=== Measuring Latencies (CPU+GPU) ===
======================================================================
[1/770] Stage 1 on CPU - 1000n 2000e... 4.33ms ±6.40
[2/770] Stage 1 on CPU - 1000n 3000e... 6.30ms ±7.03
[3/770] Stage 1 on CPU - 1000n 5000e... 5.72ms ±3.58
[4/770] Stage 1 on CPU - 1000n 7000e... 8.03ms ±5.42
[5/770] Stage 1 on CPU - 1000n 10000e... 7.92ms ±1.91
[6/770] Stage 1 on CPU - 5000n 10000e... 15.57ms ±13.68
[7/770] Stage 1 on CPU - 5000n 15000e... 14.82ms ±5.89
[8/770] Stage 1 on CPU - 5000n 25000e... 20.78ms ±17.80
[9/770] Stage 1 on CPU - 5000n 35000e... 28.63ms ±11.28
[10/770] Stage 1 on CPU - 5000n 50000e... 41.60ms ±12.96
[11/770] Stage 1 on CPU - 10000n 20000e... 16.60ms ±12.72
[12/770] Stage 1 on CPU - 10000n 30000e... 25.57ms ±11.53
[13/770] Stage 1 on CPU - 10000n 50000e... 43.50ms ±18.78
[14/770] Stage 1 on CPU - 10000n 70000e... 55.58ms ±19.96
[15/770] Stage 1 on CPU - 10000n 100000e... 60.56ms ±9.39
[16/770] Stage 1 on CPU - 20000n 40000e... 22.35ms ±4.39
[17/770] Stage 1 on CPU - 20000n 60000e... 43.10ms ±6.22
[18/770] Stage 1 on CPU - 20000n 100000e... 75.31ms ±8.85
[19/770] Stage 1 on CPU - 20000n 140000e... 107.56ms ±13.58
[20/770] Stage 1 on CPU - 20000n 200000e... 167.17ms ±17.24
[21/770] Stage 1 on CPU - 30000n 60000e... 53.57ms ±4.70
[22/770] Stage 1 on CPU - 30000n 90000e... 82.26ms ±10.07
[23/770] Stage 1 on CPU - 30000n 150000e... 139.48ms ±13.52
[24/770] Stage 1 on CPU - 30000n 210000e... 176.44ms ±19.24
[25/770] Stage 1 on CPU - 30000n 300000e... 214.24ms ±13.35
[26/770] Stage 1 on CPU - 40000n 80000e... 55.29ms ±6.46
[27/770] Stage 1 on CPU - 40000n 120000e... 86.37ms ±11.41
[28/770] Stage 1 on CPU - 40000n 200000e... 169.65ms ±7.95
[29/770] Stage 1 on CPU - 40000n 280000e... 245.77ms ±9.28
[30/770] Stage 1 on CPU - 40000n 400000e... 293.48ms ±23.31
[31/770] Stage 1 on CPU - 50000n 100000e... 75.77ms ±8.28
[32/770] Stage 1 on CPU - 50000n 150000e... 114.72ms ±12.19
[33/770] Stage 1 on CPU - 50000n 250000e... 272.66ms ±23.93
[34/770] Stage 1 on CPU - 50000n 350000e... 332.05ms ±55.63
[35/770] Stage 1 on CPU - 50000n 500000e... 389.82ms ±41.93
[36/770] Stage 1 on CPU - 60000n 120000e... 141.31ms ±16.68
[37/770] Stage 1 on CPU - 60000n 180000e... 187.87ms ±26.01
[38/770] Stage 1 on CPU - 60000n 300000e... 246.10ms ±30.82
[39/770] Stage 1 on CPU - 60000n 420000e... 321.31ms ±26.07
[40/770] Stage 1 on CPU - 60000n 600000e... 491.25ms ±55.47
[41/770] Stage 1 on CPU - 80000n 160000e... 129.00ms ±12.12
[42/770] Stage 1 on CPU - 80000n 240000e... 191.03ms ±15.11
[43/770] Stage 1 on CPU - 80000n 400000e... 409.35ms ±21.49
[44/770] Stage 1 on CPU - 80000n 560000e... 432.55ms ±26.20
[45/770] Stage 1 on CPU - 80000n 800000e... 782.67ms ±144.42
[46/770] Stage 1 on CPU - 100000n 200000e... 166.58ms ±13.69
[47/770] Stage 1 on CPU - 100000n 300000e... 273.23ms ±39.94
[48/770] Stage 1 on CPU - 100000n 500000e... 460.05ms ±78.35
[49/770] Stage 1 on CPU - 100000n 700000e... 653.25ms ±119.54
[50/770] Stage 1 on CPU - 100000n 1000000e... 836.97ms ±155.22
[51/770] Stage 1 on CPU - 150000n 300000e... 269.57ms ±44.09
[52/770] Stage 1 on CPU - 150000n 450000e... 347.53ms ±18.20
[53/770] Stage 1 on CPU - 150000n 750000e... 684.74ms ±133.17
[54/770] Stage 1 on CPU - 150000n 1050000e... 938.12ms ±181.92
[55/770] Stage 1 on CPU - 150000n 1500000e... 1285.57ms ±270.46
[56/770] Stage 1 on GPU - 1000n 2000e... 2.83ms ±1.24
[57/770] Stage 1 on GPU - 1000n 3000e... 3.88ms ±0.79
[58/770] Stage 1 on GPU - 1000n 5000e... 4.44ms ±0.67
[59/770] Stage 1 on GPU - 1000n 7000e... 6.19ms ±1.47
[60/770] Stage 1 on GPU - 1000n 10000e... 8.70ms ±1.00
[61/770] Stage 1 on GPU - 5000n 10000e... 10.08ms ±1.57
[62/770] Stage 1 on GPU - 5000n 15000e... 13.32ms ±1.46
[63/770] Stage 1 on GPU - 5000n 25000e... 21.28ms ±2.96
[64/770] Stage 1 on GPU - 5000n 35000e... 27.62ms ±2.09
[65/770] Stage 1 on GPU - 5000n 50000e... 40.32ms ±2.27
[66/770] Stage 1 on GPU - 10000n 20000e... 19.71ms ±2.76
[67/770] Stage 1 on GPU - 10000n 30000e... 28.13ms ±4.40
[68/770] Stage 1 on GPU - 10000n 50000e... 41.09ms ±2.18
[69/770] Stage 1 on GPU - 10000n 70000e... 63.10ms ±10.21
[70/770] Stage 1 on GPU - 10000n 100000e... 82.79ms ±12.22
[71/770] Stage 1 on GPU - 20000n 40000e... 35.94ms ±2.45
[72/770] Stage 1 on GPU - 20000n 60000e... 50.93ms ±2.84
[73/770] Stage 1 on GPU - 20000n 100000e... 86.66ms ±11.56
[74/770] Stage 1 on GPU - 20000n 140000e... 109.59ms ±7.79
[75/770] Stage 1 on GPU - 20000n 200000e... 154.40ms ±14.03
[76/770] Stage 1 on GPU - 30000n 60000e... 55.61ms ±4.27
[77/770] Stage 1 on GPU - 30000n 90000e... 91.26ms ±11.40
[78/770] Stage 1 on GPU - 30000n 150000e... 129.88ms ±11.10
[79/770] Stage 1 on GPU - 30000n 210000e... 169.55ms ±8.42
[80/770] Stage 1 on GPU - 30000n 300000e... 226.61ms ±10.60
[81/770] Stage 1 on GPU - 40000n 80000e... 73.97ms ±8.42
[82/770] Stage 1 on GPU - 40000n 120000e... 105.63ms ±13.67
[83/770] Stage 1 on GPU - 40000n 200000e... 162.14ms ±13.66
[84/770] Stage 1 on GPU - 40000n 280000e... 220.47ms ±14.15
[85/770] Stage 1 on GPU - 40000n 400000e... 329.12ms ±17.79
[86/770] Stage 1 on GPU - 50000n 100000e... 104.94ms ±13.39
[87/770] Stage 1 on GPU - 50000n 150000e... 138.58ms ±12.73
[88/770] Stage 1 on GPU - 50000n 250000e... 211.13ms ±12.27
[89/770] Stage 1 on GPU - 50000n 350000e... 300.95ms ±13.27
[90/770] Stage 1 on GPU - 50000n 500000e... 413.09ms ±18.41
[91/770] Stage 1 on GPU - 60000n 120000e... 117.55ms ±13.68
[92/770] Stage 1 on GPU - 60000n 180000e... 151.91ms ±9.70
[93/770] Stage 1 on GPU - 60000n 300000e... 235.86ms ±11.56
[94/770] Stage 1 on GPU - 60000n 420000e... 345.71ms ±15.46
[95/770] Stage 1 on GPU - 60000n 600000e... 484.87ms ±22.65
[96/770] Stage 1 on GPU - 80000n 160000e... 148.73ms ±13.23
[97/770] Stage 1 on GPU - 80000n 240000e... 203.32ms ±11.95
[98/770] Stage 1 on GPU - 80000n 400000e... 345.72ms ±18.07
[99/770] Stage 1 on GPU - 80000n 560000e... 467.95ms ±15.17
[100/770] Stage 1 on GPU - 80000n 800000e... 628.57ms ±19.06
[101/770] Stage 1 on GPU - 100000n 200000e... 184.01ms ±13.07
[102/770] Stage 1 on GPU - 100000n 300000e... 258.10ms ±15.55
[103/770] Stage 1 on GPU - 100000n 500000e... 413.28ms ±15.98
[104/770] Stage 1 on GPU - 100000n 700000e... 567.26ms ±18.98
[105/770] Stage 1 on GPU - 100000n 1000000e... 786.81ms ±21.56
[106/770] Stage 1 on GPU - 150000n 300000e... 290.44ms ±10.71
[107/770] Stage 1 on GPU - 150000n 450000e... 412.59ms ±19.97
[108/770] Stage 1 on GPU - 150000n 750000e... 634.23ms ±20.68
[109/770] Stage 1 on GPU - 150000n 1050000e... 860.39ms ±21.38
[110/770] Stage 1 on GPU - 150000n 1500000e... 1184.02ms ±29.59
[111/770] Stage 2 on CPU - 1000n 2000e... 1.42ms ±2.40
[112/770] Stage 2 on CPU - 1000n 3000e... 1.62ms ±0.21
[113/770] Stage 2 on CPU - 1000n 5000e... 3.83ms ±0.70
[114/770] Stage 2 on CPU - 1000n 7000e... 5.18ms ±0.79
[115/770] Stage 2 on CPU - 1000n 10000e... 7.92ms ±1.45
[116/770] Stage 2 on CPU - 5000n 10000e... 7.74ms ±0.84
[117/770] Stage 2 on CPU - 5000n 15000e... 9.55ms ±0.68
[118/770] Stage 2 on CPU - 5000n 25000e... 15.58ms ±1.24
[119/770] Stage 2 on CPU - 5000n 35000e... 21.83ms ±1.22
[120/770] Stage 2 on CPU - 5000n 50000e... 32.35ms ±2.40
[121/770] Stage 2 on CPU - 10000n 20000e... 12.27ms ±1.14
[122/770] Stage 2 on CPU - 10000n 30000e... 20.52ms ±8.48
[123/770] Stage 2 on CPU - 10000n 50000e... 33.20ms ±3.31
[124/770] Stage 2 on CPU - 10000n 70000e... 45.24ms ±3.64
[125/770] Stage 2 on CPU - 10000n 100000e... 65.41ms ±4.80
[126/770] Stage 2 on CPU - 20000n 40000e... 25.64ms ±2.08
[127/770] Stage 2 on CPU - 20000n 60000e... 39.15ms ±2.90
[128/770] Stage 2 on CPU - 20000n 100000e... 63.62ms ±5.07
[129/770] Stage 2 on CPU - 20000n 140000e... 81.95ms ±3.98
[130/770] Stage 2 on CPU - 20000n 200000e... 122.57ms ±7.98
[131/770] Stage 2 on CPU - 30000n 60000e... 35.16ms ±3.06
[132/770] Stage 2 on CPU - 30000n 90000e... 54.06ms ±3.51
[133/770] Stage 2 on CPU - 30000n 150000e... 90.64ms ±4.98
[134/770] Stage 2 on CPU - 30000n 210000e... 136.33ms ±10.52
[135/770] Stage 2 on CPU - 30000n 300000e... 197.70ms ±8.72
[136/770] Stage 2 on CPU - 40000n 80000e... 58.48ms ±18.29
[137/770] Stage 2 on CPU - 40000n 120000e... 76.07ms ±7.22
[138/770] Stage 2 on CPU - 40000n 200000e... 117.35ms ±6.47
[139/770] Stage 2 on CPU - 40000n 280000e... 164.87ms ±9.05
[140/770] Stage 2 on CPU - 40000n 400000e... 241.93ms ±10.01
[141/770] Stage 2 on CPU - 50000n 100000e... 60.74ms ±3.62
[142/770] Stage 2 on CPU - 50000n 150000e... 91.52ms ±4.75
[143/770] Stage 2 on CPU - 50000n 250000e... 154.61ms ±7.40
[144/770] Stage 2 on CPU - 50000n 350000e... 208.83ms ±11.07
[145/770] Stage 2 on CPU - 50000n 500000e... 306.23ms ±13.85
[146/770] Stage 2 on CPU - 60000n 120000e... 80.51ms ±3.86
[147/770] Stage 2 on CPU - 60000n 180000e... 119.88ms ±5.49
[148/770] Stage 2 on CPU - 60000n 300000e... 188.22ms ±10.31
[149/770] Stage 2 on CPU - 60000n 420000e... 249.64ms ±11.44
[150/770] Stage 2 on CPU - 60000n 600000e... 367.37ms ±17.82
[151/770] Stage 2 on CPU - 80000n 160000e... 103.41ms ±5.24
[152/770] Stage 2 on CPU - 80000n 240000e... 142.44ms ±13.37
[153/770] Stage 2 on CPU - 80000n 400000e... 240.24ms ±11.33
[154/770] Stage 2 on CPU - 80000n 560000e... 346.60ms ±9.42
[155/770] Stage 2 on CPU - 80000n 800000e... 455.40ms ±17.14
[156/770] Stage 2 on CPU - 100000n 200000e... 113.42ms ±6.78
[157/770] Stage 2 on CPU - 100000n 300000e... 186.95ms ±9.16
[158/770] Stage 2 on CPU - 100000n 500000e... 302.16ms ±20.71
[159/770] Stage 2 on CPU - 100000n 700000e... 410.99ms ±16.54
[160/770] Stage 2 on CPU - 100000n 1000000e... 607.81ms ±32.21
[161/770] Stage 2 on CPU - 150000n 300000e... 178.34ms ±9.74
[162/770] Stage 2 on CPU - 150000n 450000e... 273.95ms ±13.00
[163/770] Stage 2 on CPU - 150000n 750000e... 446.24ms ±15.98
[164/770] Stage 2 on CPU - 150000n 1050000e... 649.50ms ±30.13
[165/770] Stage 2 on CPU - 150000n 1500000e... 893.75ms ±25.61
[166/770] Stage 2 on GPU - 1000n 2000e... 2.50ms ±0.51
[167/770] Stage 2 on GPU - 1000n 3000e... 3.38ms ±0.68
[168/770] Stage 2 on GPU - 1000n 5000e... 5.49ms ±1.87
[169/770] Stage 2 on GPU - 1000n 7000e... 7.12ms ±1.77
[170/770] Stage 2 on GPU - 1000n 10000e... 9.42ms ±0.92
[171/770] Stage 2 on GPU - 5000n 10000e... 9.88ms ±1.12
[172/770] Stage 2 on GPU - 5000n 15000e... 14.46ms ±1.78
[173/770] Stage 2 on GPU - 5000n 25000e... 23.64ms ±1.83
[174/770] Stage 2 on GPU - 5000n 35000e... 32.54ms ±2.37
[175/770] Stage 2 on GPU - 5000n 50000e... 47.91ms ±3.66
[176/770] Stage 2 on GPU - 10000n 20000e... 18.67ms ±1.63
[177/770] Stage 2 on GPU - 10000n 30000e... 28.89ms ±1.93
[178/770] Stage 2 on GPU - 10000n 50000e... 50.15ms ±3.54
[179/770] Stage 2 on GPU - 10000n 70000e... 75.59ms ±12.09
[180/770] Stage 2 on GPU - 10000n 100000e... 101.38ms ±10.67
[181/770] Stage 2 on GPU - 20000n 40000e... 37.97ms ±2.16
[182/770] Stage 2 on GPU - 20000n 60000e... 58.33ms ±5.78
[183/770] Stage 2 on GPU - 20000n 100000e... 100.39ms ±12.40
[184/770] Stage 2 on GPU - 20000n 140000e... 135.34ms ±12.60
[185/770] Stage 2 on GPU - 20000n 200000e... 193.22ms ±16.86
[186/770] Stage 2 on GPU - 30000n 60000e... 60.65ms ±11.98
[187/770] Stage 2 on GPU - 30000n 90000e... 97.04ms ±12.57
[188/770] Stage 2 on GPU - 30000n 150000e... 149.57ms ±14.27
[189/770] Stage 2 on GPU - 30000n 210000e... 203.17ms ±14.47
[190/770] Stage 2 on GPU - 30000n 300000e... 298.75ms ±14.12
[191/770] Stage 2 on GPU - 40000n 80000e... 85.09ms ±9.78
[192/770] Stage 2 on GPU - 40000n 120000e... 117.73ms ±11.48
[193/770] Stage 2 on GPU - 40000n 200000e... 192.72ms ±16.03
[194/770] Stage 2 on GPU - 40000n 280000e... 277.74ms ±12.67
[195/770] Stage 2 on GPU - 40000n 400000e... 388.75ms ±11.09
[196/770] Stage 2 on GPU - 50000n 100000e... 101.32ms ±11.28
[197/770] Stage 2 on GPU - 50000n 150000e... 152.61ms ±17.70
[198/770] Stage 2 on GPU - 50000n 250000e... 249.67ms ±14.77
[199/770] Stage 2 on GPU - 50000n 350000e... 354.37ms ±16.87
[200/770] Stage 2 on GPU - 50000n 500000e... 470.21ms ±14.12
[201/770] Stage 2 on GPU - 60000n 120000e... 121.65ms ±13.38
[202/770] Stage 2 on GPU - 60000n 180000e... 171.80ms ±16.28
[203/770] Stage 2 on GPU - 60000n 300000e... 304.99ms ±15.33
[204/770] Stage 2 on GPU - 60000n 420000e... 398.93ms ±17.44
[205/770] Stage 2 on GPU - 60000n 600000e... 565.27ms ±29.08
[206/770] Stage 2 on GPU - 80000n 160000e... 152.36ms ±13.62
[207/770] Stage 2 on GPU - 80000n 240000e... 233.35ms ±13.51
[208/770] Stage 2 on GPU - 80000n 400000e... 388.09ms ±14.56
[209/770] Stage 2 on GPU - 80000n 560000e... 522.18ms ±13.30
[210/770] Stage 2 on GPU - 80000n 800000e... 733.71ms ±22.33
[211/770] Stage 2 on GPU - 100000n 200000e... 189.87ms ±14.06
[212/770] Stage 2 on GPU - 100000n 300000e... 298.91ms ±12.05
[213/770] Stage 2 on GPU - 100000n 500000e... 477.17ms ±15.99
[214/770] Stage 2 on GPU - 100000n 700000e... 653.65ms ±19.64
[215/770] Stage 2 on GPU - 100000n 1000000e... 908.92ms ±22.92
[216/770] Stage 2 on GPU - 150000n 300000e... 299.71ms ±14.14
[217/770] Stage 2 on GPU - 150000n 450000e... 432.96ms ±13.70
[218/770] Stage 2 on GPU - 150000n 750000e... 702.87ms ±19.06
[219/770] Stage 2 on GPU - 150000n 1050000e... 981.80ms ±29.72
[220/770] Stage 2 on GPU - 150000n 1500000e... 1348.42ms ±29.53
[221/770] Stage 3 on CPU - 1000n 2000e... 3.08ms ±0.88
[222/770] Stage 3 on CPU - 1000n 3000e... 3.89ms ±0.45
[223/770] Stage 3 on CPU - 1000n 5000e... 6.01ms ±0.67
[224/770] Stage 3 on CPU - 1000n 7000e... 8.52ms ±0.57
[225/770] Stage 3 on CPU - 1000n 10000e... 13.03ms ±0.68
[226/770] Stage 3 on CPU - 5000n 10000e... 17.69ms ±0.73
[227/770] Stage 3 on CPU - 5000n 15000e... 25.58ms ±1.41
[228/770] Stage 3 on CPU - 5000n 25000e... 32.47ms ±1.58
[229/770] Stage 3 on CPU - 5000n 35000e... 38.43ms ±2.45
[230/770] Stage 3 on CPU - 5000n 50000e... 50.09ms ±2.56
[231/770] Stage 3 on CPU - 10000n 20000e... 58.87ms ±3.47
[232/770] Stage 3 on CPU - 10000n 30000e... 82.84ms ±12.85
[233/770] Stage 3 on CPU - 10000n 50000e... 122.31ms ±25.38
[234/770] Stage 3 on CPU - 10000n 70000e... 186.48ms ±7.02
[235/770] Stage 3 on CPU - 10000n 100000e... 84.89ms ±9.83
[236/770] Stage 3 on CPU - 20000n 40000e... 62.72ms ±3.87
[237/770] Stage 3 on CPU - 20000n 60000e... 89.56ms ±7.61
[238/770] Stage 3 on CPU - 20000n 100000e... 133.61ms ±14.54
[239/770] Stage 3 on CPU - 20000n 140000e... 564.71ms ±230.93
[240/770] Stage 3 on CPU - 20000n 200000e... 256.18ms ±36.56
[241/770] Stage 3 on CPU - 30000n 60000e... 99.98ms ±12.90
[242/770] Stage 3 on CPU - 30000n 90000e... 124.01ms ±9.21
[243/770] Stage 3 on CPU - 30000n 150000e... 427.97ms ±30.99
[244/770] Stage 3 on CPU - 30000n 210000e... 303.42ms ±52.54
[245/770] Stage 3 on CPU - 30000n 300000e... 531.82ms ±166.40
[246/770] Stage 3 on CPU - 40000n 80000e... 236.91ms ±30.86
[247/770] Stage 3 on CPU - 40000n 120000e... 204.71ms ±43.59
[248/770] Stage 3 on CPU - 40000n 200000e... 441.01ms ±129.06
[249/770] Stage 3 on CPU - 40000n 280000e... 503.92ms ±163.97
[250/770] Stage 3 on CPU - 40000n 400000e... 845.86ms ±222.31
[251/770] Stage 3 on CPU - 50000n 100000e... 236.15ms ±33.10
[252/770] Stage 3 on CPU - 50000n 150000e... 376.91ms ±94.78
[253/770] Stage 3 on CPU - 50000n 250000e... 500.46ms ±116.07
[254/770] Stage 3 on CPU - 50000n 350000e... 776.29ms ±195.98
[255/770] Stage 3 on CPU - 50000n 500000e... 962.03ms ±339.02
[256/770] Stage 3 on CPU - 60000n 120000e... 288.44ms ±36.27
[257/770] Stage 3 on CPU - 60000n 180000e... 426.28ms ±84.48
[258/770] Stage 3 on CPU - 60000n 300000e... 556.85ms ±117.42
[259/770] Stage 3 on CPU - 60000n 420000e... 1070.54ms ±650.08
[260/770] Stage 3 on CPU - 60000n 600000e... 1072.29ms ±319.33
[261/770] Stage 3 on CPU - 80000n 160000e... 591.44ms ±249.66
[262/770] Stage 3 on CPU - 80000n 240000e... 548.25ms ±117.15
[263/770] Stage 3 on CPU - 80000n 400000e... 699.95ms ±131.26
[264/770] Stage 3 on CPU - 80000n 560000e... 1082.00ms ±298.03
[265/770] Stage 3 on CPU - 80000n 800000e... 1419.79ms ±423.69
[266/770] Stage 3 on CPU - 100000n 200000e... 487.53ms ±101.92
[267/770] Stage 3 on CPU - 100000n 300000e... 599.87ms ±119.56
[268/770] Stage 3 on CPU - 100000n 500000e... 1002.25ms ±271.11
[269/770] Stage 3 on CPU - 100000n 700000e... 1525.13ms ±944.26
[270/770] Stage 3 on CPU - 100000n 1000000e... 2059.76ms ±1176.57
[271/770] Stage 3 on CPU - 150000n 300000e... 817.23ms ±153.53
[272/770] Stage 3 on CPU - 150000n 450000e... 1038.72ms ±230.08
[273/770] Stage 3 on CPU - 150000n 750000e... 1742.67ms ±957.37
[274/770] Stage 3 on CPU - 150000n 1050000e... 2143.50ms ±1220.51
[275/770] Stage 3 on CPU - 150000n 1500000e... 2654.14ms ±1609.45
[276/770] Stage 3 on GPU - 1000n 2000e... 2.79ms ±0.68
[277/770] Stage 3 on GPU - 1000n 3000e... 3.08ms ±0.93
[278/770] Stage 3 on GPU - 1000n 5000e... 3.82ms ±0.56
[279/770] Stage 3 on GPU - 1000n 7000e... 4.83ms ±0.79
[280/770] Stage 3 on GPU - 1000n 10000e... 6.18ms ±0.89
[281/770] Stage 3 on GPU - 5000n 10000e... 9.51ms ±1.16
[282/770] Stage 3 on GPU - 5000n 15000e... 12.17ms ±1.92
[283/770] Stage 3 on GPU - 5000n 25000e... 16.04ms ±1.24
[284/770] Stage 3 on GPU - 5000n 35000e... 21.06ms ±1.03
[285/770] Stage 3 on GPU - 5000n 50000e... 28.58ms ±1.26
[286/770] Stage 3 on GPU - 10000n 20000e... 17.50ms ±0.84
[287/770] Stage 3 on GPU - 10000n 30000e... 23.66ms ±2.92
[288/770] Stage 3 on GPU - 10000n 50000e... 32.76ms ±1.33
[289/770] Stage 3 on GPU - 10000n 70000e... 43.28ms ±3.93
[290/770] Stage 3 on GPU - 10000n 100000e... 53.78ms ±2.13
[291/770] Stage 3 on GPU - 20000n 40000e... 35.73ms ±2.38
[292/770] Stage 3 on GPU - 20000n 60000e... 43.57ms ±2.03
[293/770] Stage 3 on GPU - 20000n 100000e... 64.13ms ±7.96
[294/770] Stage 3 on GPU - 20000n 140000e... 82.55ms ±3.91
[295/770] Stage 3 on GPU - 20000n 200000e... 110.24ms ±4.02
[296/770] Stage 3 on GPU - 30000n 60000e... 52.90ms ±2.95
[297/770] Stage 3 on GPU - 30000n 90000e... 70.13ms ±2.61
[298/770] Stage 3 on GPU - 30000n 150000e... 101.30ms ±4.56
[299/770] Stage 3 on GPU - 30000n 210000e... 128.35ms ±5.28
[300/770] Stage 3 on GPU - 30000n 300000e... 167.27ms ±6.05
[301/770] Stage 3 on GPU - 40000n 80000e... 69.73ms ±3.88
[302/770] Stage 3 on GPU - 40000n 120000e... 88.31ms ±4.16
[303/770] Stage 3 on GPU - 40000n 200000e... 127.94ms ±11.10
[304/770] Stage 3 on GPU - 40000n 280000e... 163.27ms ±6.19
[305/770] Stage 3 on GPU - 40000n 400000e... 221.07ms ±5.72
[306/770] Stage 3 on GPU - 50000n 100000e... 88.49ms ±3.83
[307/770] Stage 3 on GPU - 50000n 150000e... 115.57ms ±4.57
[308/770] Stage 3 on GPU - 50000n 250000e... 164.14ms ±5.06
[309/770] Stage 3 on GPU - 50000n 350000e... 216.38ms ±10.16
[310/770] Stage 3 on GPU - 50000n 500000e... 273.53ms ±5.23
[311/770] Stage 3 on GPU - 60000n 120000e... 103.95ms ±4.17
[312/770] Stage 3 on GPU - 60000n 180000e... 130.31ms ±4.70
[313/770] Stage 3 on GPU - 60000n 300000e... 188.53ms ±5.97
[314/770] Stage 3 on GPU - 60000n 420000e... 243.31ms ±5.91
[315/770] Stage 3 on GPU - 60000n 600000e... 324.95ms ±7.13
[316/770] Stage 3 on GPU - 80000n 160000e... 142.06ms ±7.12
[317/770] Stage 3 on GPU - 80000n 240000e... 177.02ms ±7.74
[318/770] Stage 3 on GPU - 80000n 400000e... 253.23ms ±7.84
[319/770] Stage 3 on GPU - 80000n 560000e... 324.28ms ±6.00
[320/770] Stage 3 on GPU - 80000n 800000e... 434.87ms ±12.60
[321/770] Stage 3 on GPU - 100000n 200000e... 176.03ms ±11.88
[322/770] Stage 3 on GPU - 100000n 300000e... 220.38ms ±8.11
[323/770] Stage 3 on GPU - 100000n 500000e... 312.23ms ±7.30
[324/770] Stage 3 on GPU - 100000n 700000e... 405.16ms ±11.11
[325/770] Stage 3 on GPU - 100000n 1000000e... 539.12ms ±5.85
[326/770] Stage 3 on GPU - 150000n 300000e... 261.96ms ±8.29
[327/770] Stage 3 on GPU - 150000n 450000e... 349.05ms ±15.91
[328/770] Stage 3 on GPU - 150000n 750000e... 497.93ms ±10.99
[329/770] Stage 3 on GPU - 150000n 1050000e... 644.94ms ±18.98
[330/770] Stage 3 on GPU - 150000n 1500000e... 824.67ms ±12.08
[331/770] Stage 4 on CPU - 1000n 2000e... 0.19ms ±0.12
[332/770] Stage 4 on CPU - 1000n 3000e... 0.25ms ±0.36
[333/770] Stage 4 on CPU - 1000n 5000e... 0.20ms ±0.05
[334/770] Stage 4 on CPU - 1000n 7000e... 0.26ms ±0.09
[335/770] Stage 4 on CPU - 1000n 10000e... 0.29ms ±0.09
[336/770] Stage 4 on CPU - 5000n 10000e... 0.50ms ±0.42
[337/770] Stage 4 on CPU - 5000n 15000e... 0.30ms ±0.06
[338/770] Stage 4 on CPU - 5000n 25000e... 0.41ms ±0.15
[339/770] Stage 4 on CPU - 5000n 35000e... 0.48ms ±0.10
[340/770] Stage 4 on CPU - 5000n 50000e... 0.56ms ±0.09
[341/770] Stage 4 on CPU - 10000n 20000e... 0.48ms ±0.19
[342/770] Stage 4 on CPU - 10000n 30000e... 0.41ms ±0.12
[343/770] Stage 4 on CPU - 10000n 50000e... 0.57ms ±0.13
[344/770] Stage 4 on CPU - 10000n 70000e... 0.60ms ±0.07
[345/770] Stage 4 on CPU - 10000n 100000e... 0.77ms ±0.11
[346/770] Stage 4 on CPU - 20000n 40000e... 0.51ms ±0.13
[347/770] Stage 4 on CPU - 20000n 60000e... 0.65ms ±0.20
[348/770] Stage 4 on CPU - 20000n 100000e... 0.76ms ±0.11
[349/770] Stage 4 on CPU - 20000n 140000e... 1.77ms ±0.19
[350/770] Stage 4 on CPU - 20000n 200000e... 2.54ms ±0.24
[351/770] Stage 4 on CPU - 30000n 60000e... 0.61ms ±0.07
[352/770] Stage 4 on CPU - 30000n 90000e... 0.80ms ±0.11
[353/770] Stage 4 on CPU - 30000n 150000e... 1.97ms ±0.19
[354/770] Stage 4 on CPU - 30000n 210000e... 2.78ms ±0.26
[355/770] Stage 4 on CPU - 30000n 300000e... 3.76ms ±0.36
[356/770] Stage 4 on CPU - 40000n 80000e... 0.70ms ±0.08
[357/770] Stage 4 on CPU - 40000n 120000e... 1.39ms ±0.22
[358/770] Stage 4 on CPU - 40000n 200000e... 2.58ms ±0.29
[359/770] Stage 4 on CPU - 40000n 280000e... 3.52ms ±0.29
[360/770] Stage 4 on CPU - 40000n 400000e... 5.24ms ±0.75
[361/770] Stage 4 on CPU - 50000n 100000e... 0.86ms ±0.17
[362/770] Stage 4 on CPU - 50000n 150000e... 1.96ms ±0.25
[363/770] Stage 4 on CPU - 50000n 250000e... 3.16ms ±0.24
[364/770] Stage 4 on CPU - 50000n 350000e... 4.45ms ±0.30
[365/770] Stage 4 on CPU - 50000n 500000e... 6.27ms ±0.39
[366/770] Stage 4 on CPU - 60000n 120000e... 1.34ms ±0.19
[367/770] Stage 4 on CPU - 60000n 180000e... 2.23ms ±0.13
[368/770] Stage 4 on CPU - 60000n 300000e... 3.84ms ±0.44
[369/770] Stage 4 on CPU - 60000n 420000e... 5.45ms ±0.43
[370/770] Stage 4 on CPU - 60000n 600000e... 7.73ms ±0.61
[371/770] Stage 4 on CPU - 80000n 160000e... 2.17ms ±0.26
[372/770] Stage 4 on CPU - 80000n 240000e... 3.14ms ±0.43
[373/770] Stage 4 on CPU - 80000n 400000e... 5.32ms ±0.54
[374/770] Stage 4 on CPU - 80000n 560000e... 7.43ms ±0.57
[375/770] Stage 4 on CPU - 80000n 800000e... 10.48ms ±0.98
[376/770] Stage 4 on CPU - 100000n 200000e... 2.57ms ±0.25
[377/770] Stage 4 on CPU - 100000n 300000e... 4.02ms ±0.46
[378/770] Stage 4 on CPU - 100000n 500000e... 6.69ms ±0.71
[379/770] Stage 4 on CPU - 100000n 700000e... 9.21ms ±0.86
[380/770] Stage 4 on CPU - 100000n 1000000e... 12.72ms ±0.86
[381/770] Stage 4 on CPU - 150000n 300000e... 3.92ms ±0.36
[382/770] Stage 4 on CPU - 150000n 450000e... 6.13ms ±0.70
[383/770] Stage 4 on CPU - 150000n 750000e... 10.39ms ±1.14
[384/770] Stage 4 on CPU - 150000n 1050000e... 14.18ms ±0.91
[385/770] Stage 4 on CPU - 150000n 1500000e... 19.93ms ±1.27
[386/770] Stage 4 on GPU - 1000n 2000e... 0.76ms ±0.25
[387/770] Stage 4 on GPU - 1000n 3000e... 0.76ms ±0.13
[388/770] Stage 4 on GPU - 1000n 5000e... 0.72ms ±0.22
[389/770] Stage 4 on GPU - 1000n 7000e... 0.82ms ±0.22
[390/770] Stage 4 on GPU - 1000n 10000e... 0.85ms ±0.20
[391/770] Stage 4 on GPU - 5000n 10000e... 0.70ms ±0.18
[392/770] Stage 4 on GPU - 5000n 15000e... 0.80ms ±0.22
[393/770] Stage 4 on GPU - 5000n 25000e... 0.82ms ±0.20
[394/770] Stage 4 on GPU - 5000n 35000e... 0.73ms ±0.22
[395/770] Stage 4 on GPU - 5000n 50000e... 0.75ms ±0.09
[396/770] Stage 4 on GPU - 10000n 20000e... 0.83ms ±0.29
[397/770] Stage 4 on GPU - 10000n 30000e... 0.72ms ±0.17
[398/770] Stage 4 on GPU - 10000n 50000e... 0.71ms ±0.27
[399/770] Stage 4 on GPU - 10000n 70000e... 0.88ms ±0.15
[400/770] Stage 4 on GPU - 10000n 100000e... 0.87ms ±0.28
[401/770] Stage 4 on GPU - 20000n 40000e... 0.95ms ±0.68
[402/770] Stage 4 on GPU - 20000n 60000e... 0.94ms ±0.25
[403/770] Stage 4 on GPU - 20000n 100000e... 0.86ms ±0.17
[404/770] Stage 4 on GPU - 20000n 140000e... 0.89ms ±0.10
[405/770] Stage 4 on GPU - 20000n 200000e... 1.00ms ±0.17
[406/770] Stage 4 on GPU - 30000n 60000e... 0.93ms ±0.35
[407/770] Stage 4 on GPU - 30000n 90000e... 1.02ms ±0.24
[408/770] Stage 4 on GPU - 30000n 150000e... 1.14ms ±0.23
[409/770] Stage 4 on GPU - 30000n 210000e... 1.15ms ±0.17
[410/770] Stage 4 on GPU - 30000n 300000e... 1.37ms ±0.19
[411/770] Stage 4 on GPU - 40000n 80000e... 0.90ms ±0.21
[412/770] Stage 4 on GPU - 40000n 120000e... 1.02ms ±0.15
[413/770] Stage 4 on GPU - 40000n 200000e... 1.21ms ±0.14
[414/770] Stage 4 on GPU - 40000n 280000e... 1.32ms ±0.32
[415/770] Stage 4 on GPU - 40000n 400000e... 1.77ms ±0.58
[416/770] Stage 4 on GPU - 50000n 100000e... 0.87ms ±0.20
[417/770] Stage 4 on GPU - 50000n 150000e... 1.20ms ±0.53
[418/770] Stage 4 on GPU - 50000n 250000e... 1.36ms ±0.09
[419/770] Stage 4 on GPU - 50000n 350000e... 1.50ms ±0.19
[420/770] Stage 4 on GPU - 50000n 500000e... 1.94ms ±0.41
[421/770] Stage 4 on GPU - 60000n 120000e... 0.92ms ±0.19
[422/770] Stage 4 on GPU - 60000n 180000e... 1.19ms ±0.25
[423/770] Stage 4 on GPU - 60000n 300000e... 1.42ms ±0.16
[424/770] Stage 4 on GPU - 60000n 420000e... 1.58ms ±0.14
[425/770] Stage 4 on GPU - 60000n 600000e... 2.15ms ±0.41
[426/770] Stage 4 on GPU - 80000n 160000e... 1.13ms ±0.23
[427/770] Stage 4 on GPU - 80000n 240000e... 1.35ms ±0.53
[428/770] Stage 4 on GPU - 80000n 400000e... 1.63ms ±0.20
[429/770] Stage 4 on GPU - 80000n 560000e... 2.27ms ±0.80
[430/770] Stage 4 on GPU - 80000n 800000e... 2.81ms ±0.53
[431/770] Stage 4 on GPU - 100000n 200000e... 1.53ms ±0.54
[432/770] Stage 4 on GPU - 100000n 300000e... 1.33ms ±0.30
[433/770] Stage 4 on GPU - 100000n 500000e... 2.05ms ±0.63
[434/770] Stage 4 on GPU - 100000n 700000e... 2.46ms ±0.32
[435/770] Stage 4 on GPU - 100000n 1000000e... 3.28ms ±0.79
[436/770] Stage 4 on GPU - 150000n 300000e... 1.60ms ±0.98
[437/770] Stage 4 on GPU - 150000n 450000e... 2.00ms ±0.62
[438/770] Stage 4 on GPU - 150000n 750000e... 2.77ms ±0.93
[439/770] Stage 4 on GPU - 150000n 1050000e... 3.49ms ±0.53
[440/770] Stage 4 on GPU - 150000n 1500000e... 4.77ms ±0.98
[441/770] Stage 5 on CPU - 1000n 2000e... 0.76ms ±0.15
[442/770] Stage 5 on CPU - 1000n 3000e... 0.68ms ±0.10
[443/770] Stage 5 on CPU - 1000n 5000e... 0.65ms ±0.13
[444/770] Stage 5 on CPU - 1000n 7000e... 0.68ms ±0.18
[445/770] Stage 5 on CPU - 1000n 10000e... 0.64ms ±0.09
[446/770] Stage 5 on CPU - 5000n 10000e... 2.74ms ±0.35
[447/770] Stage 5 on CPU - 5000n 15000e... 2.98ms ±0.72
[448/770] Stage 5 on CPU - 5000n 25000e... 3.20ms ±0.95
[449/770] Stage 5 on CPU - 5000n 35000e... 2.84ms ±0.58
[450/770] Stage 5 on CPU - 5000n 50000e... 2.83ms ±0.60
[451/770] Stage 5 on CPU - 10000n 20000e... 5.76ms ±0.84
[452/770] Stage 5 on CPU - 10000n 30000e... 5.71ms ±0.59
[453/770] Stage 5 on CPU - 10000n 50000e... 5.98ms ±1.19
[454/770] Stage 5 on CPU - 10000n 70000e... 5.58ms ±0.48
[455/770] Stage 5 on CPU - 10000n 100000e... 5.68ms ±0.44
[456/770] Stage 5 on CPU - 20000n 40000e... 12.52ms ±1.58
[457/770] Stage 5 on CPU - 20000n 60000e... 11.71ms ±0.89
[458/770] Stage 5 on CPU - 20000n 100000e... 12.12ms ±1.19
[459/770] Stage 5 on CPU - 20000n 140000e... 12.04ms ±0.91
[460/770] Stage 5 on CPU - 20000n 200000e... 12.36ms ±1.48
[461/770] Stage 5 on CPU - 30000n 60000e... 18.62ms ±2.95
[462/770] Stage 5 on CPU - 30000n 90000e... 18.30ms ±1.87
[463/770] Stage 5 on CPU - 30000n 150000e... 19.07ms ±2.14
[464/770] Stage 5 on CPU - 30000n 210000e... 18.11ms ±1.24
[465/770] Stage 5 on CPU - 30000n 300000e... 18.58ms ±1.65
[466/770] Stage 5 on CPU - 40000n 80000e... 25.37ms ±2.81
[467/770] Stage 5 on CPU - 40000n 120000e... 24.16ms ±3.37
[468/770] Stage 5 on CPU - 40000n 200000e... 24.23ms ±1.69
[469/770] Stage 5 on CPU - 40000n 280000e... 26.42ms ±2.45
[470/770] Stage 5 on CPU - 40000n 400000e... 26.70ms ±2.42
[471/770] Stage 5 on CPU - 50000n 100000e... 31.33ms ±3.15
[472/770] Stage 5 on CPU - 50000n 150000e... 36.07ms ±2.94
[473/770] Stage 5 on CPU - 50000n 250000e... 35.29ms ±2.91
[474/770] Stage 5 on CPU - 50000n 350000e... 35.27ms ±3.04
[475/770] Stage 5 on CPU - 50000n 500000e... 35.37ms ±3.46
[476/770] Stage 5 on CPU - 60000n 120000e... 41.69ms ±2.77
[477/770] Stage 5 on CPU - 60000n 180000e... 42.04ms ±3.02
[478/770] Stage 5 on CPU - 60000n 300000e... 42.53ms ±3.82
[479/770] Stage 5 on CPU - 60000n 420000e... 42.58ms ±3.32
[480/770] Stage 5 on CPU - 60000n 600000e... 42.00ms ±2.90
[481/770] Stage 5 on CPU - 80000n 160000e... 56.34ms ±5.80
[482/770] Stage 5 on CPU - 80000n 240000e... 51.99ms ±4.67
[483/770] Stage 5 on CPU - 80000n 400000e... 52.73ms ±9.95
[484/770] Stage 5 on CPU - 80000n 560000e... 51.56ms ±3.39
[485/770] Stage 5 on CPU - 80000n 800000e... 51.23ms ±4.22
[486/770] Stage 5 on CPU - 100000n 200000e... 63.92ms ±3.71
[487/770] Stage 5 on CPU - 100000n 300000e... 66.05ms ±6.47
[488/770] Stage 5 on CPU - 100000n 500000e... 63.64ms ±4.17
[489/770] Stage 5 on CPU - 100000n 700000e... 72.11ms ±7.35
[490/770] Stage 5 on CPU - 100000n 1000000e... 79.21ms ±4.96
[491/770] Stage 5 on CPU - 150000n 300000e... 116.98ms ±5.51
[492/770] Stage 5 on CPU - 150000n 450000e... 118.25ms ±7.95
[493/770] Stage 5 on CPU - 150000n 750000e... 112.49ms ±12.43
[494/770] Stage 5 on CPU - 150000n 1050000e... 97.49ms ±6.14
[495/770] Stage 5 on CPU - 150000n 1500000e... 95.06ms ±6.46
[496/770] Stage 5 on GPU - 1000n 2000e... 1.44ms ±0.16
[497/770] Stage 5 on GPU - 1000n 3000e... 1.62ms ±0.42
[498/770] Stage 5 on GPU - 1000n 5000e... 1.92ms ±1.09
[499/770] Stage 5 on GPU - 1000n 7000e... 1.77ms ±0.37
[500/770] Stage 5 on GPU - 1000n 10000e... 2.01ms ±0.91
[501/770] Stage 5 on GPU - 5000n 10000e... 5.79ms ±0.64
[502/770] Stage 5 on GPU - 5000n 15000e... 6.33ms ±1.14
[503/770] Stage 5 on GPU - 5000n 25000e... 6.03ms ±1.04
[504/770] Stage 5 on GPU - 5000n 35000e... 6.67ms ±0.89
[505/770] Stage 5 on GPU - 5000n 50000e... 5.92ms ±0.73
[506/770] Stage 5 on GPU - 10000n 20000e... 12.21ms ±2.33
[507/770] Stage 5 on GPU - 10000n 30000e... 11.94ms ±1.42
[508/770] Stage 5 on GPU - 10000n 50000e... 11.99ms ±1.22
[509/770] Stage 5 on GPU - 10000n 70000e... 11.36ms ±0.57
[510/770] Stage 5 on GPU - 10000n 100000e... 11.91ms ±1.39
[511/770] Stage 5 on GPU - 20000n 40000e... 22.16ms ±1.95
[512/770] Stage 5 on GPU - 20000n 60000e... 23.19ms ±3.31
[513/770] Stage 5 on GPU - 20000n 100000e... 22.01ms ±1.66
[514/770] Stage 5 on GPU - 20000n 140000e... 22.28ms ±1.96
[515/770] Stage 5 on GPU - 20000n 200000e... 22.65ms ±1.80
[516/770] Stage 5 on GPU - 30000n 60000e... 34.68ms ±2.78
[517/770] Stage 5 on GPU - 30000n 90000e... 35.64ms ±3.66
[518/770] Stage 5 on GPU - 30000n 150000e... 34.24ms ±2.23
[519/770] Stage 5 on GPU - 30000n 210000e... 34.09ms ±2.33
[520/770] Stage 5 on GPU - 30000n 300000e... 34.96ms ±7.16
[521/770] Stage 5 on GPU - 40000n 80000e... 42.92ms ±1.69
[522/770] Stage 5 on GPU - 40000n 120000e... 43.17ms ±2.57
[523/770] Stage 5 on GPU - 40000n 200000e... 43.81ms ±2.46
[524/770] Stage 5 on GPU - 40000n 280000e... 42.74ms ±2.24
[525/770] Stage 5 on GPU - 40000n 400000e... 43.00ms ±2.16
[526/770] Stage 5 on GPU - 50000n 100000e... 55.84ms ±2.02
[527/770] Stage 5 on GPU - 50000n 150000e... 55.91ms ±2.61
[528/770] Stage 5 on GPU - 50000n 250000e... 56.00ms ±2.07
[529/770] Stage 5 on GPU - 50000n 350000e... 56.83ms ±4.00
[530/770] Stage 5 on GPU - 50000n 500000e... 57.14ms ±2.73
[531/770] Stage 5 on GPU - 60000n 120000e... 64.24ms ±3.26
[532/770] Stage 5 on GPU - 60000n 180000e... 63.53ms ±2.23
[533/770] Stage 5 on GPU - 60000n 300000e... 65.38ms ±3.55
[534/770] Stage 5 on GPU - 60000n 420000e... 64.03ms ±3.03
[535/770] Stage 5 on GPU - 60000n 600000e... 63.85ms ±3.64
[536/770] Stage 5 on GPU - 80000n 160000e... 92.05ms ±9.20
[537/770] Stage 5 on GPU - 80000n 240000e... 89.62ms ±7.13
[538/770] Stage 5 on GPU - 80000n 400000e... 90.15ms ±7.03
[539/770] Stage 5 on GPU - 80000n 560000e... 88.72ms ±11.33
[540/770] Stage 5 on GPU - 80000n 800000e... 87.79ms ±8.29
[541/770] Stage 5 on GPU - 100000n 200000e... 127.57ms ±13.39
[542/770] Stage 5 on GPU - 100000n 300000e... 127.09ms ±10.56
[543/770] Stage 5 on GPU - 100000n 500000e... 126.28ms ±9.83
[544/770] Stage 5 on GPU - 100000n 700000e... 127.82ms ±10.77
[545/770] Stage 5 on GPU - 100000n 1000000e... 125.06ms ±9.39
[546/770] Stage 5 on GPU - 150000n 300000e... 195.35ms ±7.74
[547/770] Stage 5 on GPU - 150000n 450000e... 195.12ms ±10.64
[548/770] Stage 5 on GPU - 150000n 750000e... 195.52ms ±8.66
[549/770] Stage 5 on GPU - 150000n 1050000e... 196.18ms ±12.32
[550/770] Stage 5 on GPU - 150000n 1500000e... 195.28ms ±10.49
[551/770] Stage 6 on CPU - 1000n 2000e... 3.64ms ±0.63
[552/770] Stage 6 on CPU - 1000n 3000e... 3.48ms ±0.70
[553/770] Stage 6 on CPU - 1000n 5000e... 2.60ms ±0.23
[554/770] Stage 6 on CPU - 1000n 7000e... 2.57ms ±0.14
[555/770] Stage 6 on CPU - 1000n 10000e... 2.61ms ±0.21
[556/770] Stage 6 on CPU - 5000n 10000e... 15.58ms ±1.09
[557/770] Stage 6 on CPU - 5000n 15000e... 13.74ms ±1.73
[558/770] Stage 6 on CPU - 5000n 25000e... 12.63ms ±0.63
[559/770] Stage 6 on CPU - 5000n 35000e... 14.99ms ±1.58
[560/770] Stage 6 on CPU - 5000n 50000e... 15.31ms ±0.90
[561/770] Stage 6 on CPU - 10000n 20000e... 24.39ms ±1.80
[562/770] Stage 6 on CPU - 10000n 30000e... 23.52ms ±0.91
[563/770] Stage 6 on CPU - 10000n 50000e... 27.69ms ±2.61
[564/770] Stage 6 on CPU - 10000n 70000e... 23.65ms ±1.05
[565/770] Stage 6 on CPU - 10000n 100000e... 28.66ms ±0.92
[566/770] Stage 6 on CPU - 20000n 40000e... 42.14ms ±2.61
[567/770] Stage 6 on CPU - 20000n 60000e... 43.38ms ±1.19
[568/770] Stage 6 on CPU - 20000n 100000e... 43.95ms ±1.23
[569/770] Stage 6 on CPU - 20000n 140000e... 42.64ms ±2.91
[570/770] Stage 6 on CPU - 20000n 200000e... 47.47ms ±5.55
[571/770] Stage 6 on CPU - 30000n 60000e... 190.38ms ±56.32
[572/770] Stage 6 on CPU - 30000n 90000e... 211.13ms ±59.35
[573/770] Stage 6 on CPU - 30000n 150000e... 75.31ms ±41.49
[574/770] Stage 6 on CPU - 30000n 210000e... 63.60ms ±3.66
[575/770] Stage 6 on CPU - 30000n 300000e... 62.64ms ±4.03
[576/770] Stage 6 on CPU - 40000n 80000e... 86.36ms ±4.70
[577/770] Stage 6 on CPU - 40000n 120000e... 85.94ms ±4.69
[578/770] Stage 6 on CPU - 40000n 200000e... 84.72ms ±6.83
[579/770] Stage 6 on CPU - 40000n 280000e... 312.78ms ±119.09
[580/770] Stage 6 on CPU - 40000n 400000e... 248.12ms ±149.78
[581/770] Stage 6 on CPU - 50000n 100000e... 105.13ms ±7.86
[582/770] Stage 6 on CPU - 50000n 150000e... 107.25ms ±6.84
[583/770] Stage 6 on CPU - 50000n 250000e... 112.54ms ±13.80
[584/770] Stage 6 on CPU - 50000n 350000e... 107.51ms ±5.63
[585/770] Stage 6 on CPU - 50000n 500000e... 454.80ms ±46.18
[586/770] Stage 6 on CPU - 60000n 120000e... 144.99ms ±30.08
[587/770] Stage 6 on CPU - 60000n 180000e... 146.36ms ±34.21
[588/770] Stage 6 on CPU - 60000n 300000e... 146.87ms ±32.24
[589/770] Stage 6 on CPU - 60000n 420000e... 261.80ms ±66.28
[590/770] Stage 6 on CPU - 60000n 600000e... 261.32ms ±53.16
[591/770] Stage 6 on CPU - 80000n 160000e... 229.63ms ±58.26
[592/770] Stage 6 on CPU - 80000n 240000e... 225.81ms ±58.49
[593/770] Stage 6 on CPU - 80000n 400000e... 586.14ms ±229.59
[594/770] Stage 6 on CPU - 80000n 560000e... 237.74ms ±56.03
[595/770] Stage 6 on CPU - 80000n 800000e... 244.78ms ±80.89
[596/770] Stage 6 on CPU - 100000n 200000e... 460.95ms ±96.20
[597/770] Stage 6 on CPU - 100000n 300000e... 278.45ms ±34.02
[598/770] Stage 6 on CPU - 100000n 500000e... 490.42ms ±269.82
[599/770] Stage 6 on CPU - 100000n 700000e... 328.67ms ±119.17
[600/770] Stage 6 on CPU - 100000n 1000000e... 370.67ms ±85.53
[601/770] Stage 6 on CPU - 150000n 300000e... 546.44ms ±146.01
[602/770] Stage 6 on CPU - 150000n 450000e... 571.23ms ±136.83
[603/770] Stage 6 on CPU - 150000n 750000e... 461.06ms ±77.17
[604/770] Stage 6 on CPU - 150000n 1050000e... 674.06ms ±125.52
[605/770] Stage 6 on CPU - 150000n 1500000e... 440.07ms ±37.48
[606/770] Stage 6 on GPU - 1000n 2000e... 2.23ms ±0.75
[607/770] Stage 6 on GPU - 1000n 3000e... 2.13ms ±0.63
[608/770] Stage 6 on GPU - 1000n 5000e... 2.15ms ±0.54
[609/770] Stage 6 on GPU - 1000n 7000e... 2.30ms ±2.02
[610/770] Stage 6 on GPU - 1000n 10000e... 2.74ms ±4.61
[611/770] Stage 6 on GPU - 5000n 10000e... 7.76ms ±1.23
[612/770] Stage 6 on GPU - 5000n 15000e... 8.01ms ±1.26
[613/770] Stage 6 on GPU - 5000n 25000e... 8.27ms ±1.26
[614/770] Stage 6 on GPU - 5000n 35000e... 8.06ms ±1.22
[615/770] Stage 6 on GPU - 5000n 50000e... 7.84ms ±0.92
[616/770] Stage 6 on GPU - 10000n 20000e... 15.58ms ±1.49
[617/770] Stage 6 on GPU - 10000n 30000e... 15.55ms ±1.44
[618/770] Stage 6 on GPU - 10000n 50000e... 15.14ms ±1.24
[619/770] Stage 6 on GPU - 10000n 70000e... 14.81ms ±1.01
[620/770] Stage 6 on GPU - 10000n 100000e... 15.22ms ±2.03
[621/770] Stage 6 on GPU - 20000n 40000e... 28.52ms ±2.04
[622/770] Stage 6 on GPU - 20000n 60000e... 28.93ms ±4.38
[623/770] Stage 6 on GPU - 20000n 100000e... 28.88ms ±1.86
[624/770] Stage 6 on GPU - 20000n 140000e... 28.90ms ±2.00
[625/770] Stage 6 on GPU - 20000n 200000e... 28.36ms ±1.59
[626/770] Stage 6 on GPU - 30000n 60000e... 44.32ms ±3.33
[627/770] Stage 6 on GPU - 30000n 90000e... 44.68ms ±2.36
[628/770] Stage 6 on GPU - 30000n 150000e... 44.10ms ±2.25
[629/770] Stage 6 on GPU - 30000n 210000e... 45.02ms ±2.44
[630/770] Stage 6 on GPU - 30000n 300000e... 46.20ms ±4.07
[631/770] Stage 6 on GPU - 40000n 80000e... 58.24ms ±8.27
[632/770] Stage 6 on GPU - 40000n 120000e... 56.85ms ±2.39
[633/770] Stage 6 on GPU - 40000n 200000e... 56.44ms ±2.83
[634/770] Stage 6 on GPU - 40000n 280000e... 56.18ms ±2.83
[635/770] Stage 6 on GPU - 40000n 400000e... 57.12ms ±3.35
[636/770] Stage 6 on GPU - 50000n 100000e... 74.49ms ±3.64
[637/770] Stage 6 on GPU - 50000n 150000e... 74.96ms ±3.77
[638/770] Stage 6 on GPU - 50000n 250000e... 73.47ms ±3.26
[639/770] Stage 6 on GPU - 50000n 350000e... 73.87ms ±3.14
[640/770] Stage 6 on GPU - 50000n 500000e... 73.69ms ±3.06
[641/770] Stage 6 on GPU - 60000n 120000e... 83.99ms ±3.07
[642/770] Stage 6 on GPU - 60000n 180000e... 85.09ms ±3.07
[643/770] Stage 6 on GPU - 60000n 300000e... 84.93ms ±3.61
[644/770] Stage 6 on GPU - 60000n 420000e... 85.50ms ±9.02
[645/770] Stage 6 on GPU - 60000n 600000e... 85.59ms ±7.22
[646/770] Stage 6 on GPU - 80000n 160000e... 115.16ms ±14.55
[647/770] Stage 6 on GPU - 80000n 240000e... 113.74ms ±4.54
[648/770] Stage 6 on GPU - 80000n 400000e... 115.28ms ±5.65
[649/770] Stage 6 on GPU - 80000n 560000e... 114.00ms ±4.15
[650/770] Stage 6 on GPU - 80000n 800000e... 114.16ms ±3.85
[651/770] Stage 6 on GPU - 100000n 200000e... 141.16ms ±5.69
[652/770] Stage 6 on GPU - 100000n 300000e... 143.50ms ±12.68
[653/770] Stage 6 on GPU - 100000n 500000e... 142.89ms ±6.77
[654/770] Stage 6 on GPU - 100000n 700000e... 142.52ms ±7.02
[655/770] Stage 6 on GPU - 100000n 1000000e... 139.74ms ±4.23
[656/770] Stage 6 on GPU - 150000n 300000e... 219.72ms ±8.61
[657/770] Stage 6 on GPU - 150000n 450000e... 221.74ms ±11.80
[658/770] Stage 6 on GPU - 150000n 750000e... 222.14ms ±8.78
[659/770] Stage 6 on GPU - 150000n 1050000e... 221.39ms ±9.31
[660/770] Stage 6 on GPU - 150000n 1500000e... 221.62ms ±6.87
[661/770] Stage 7 on CPU - 1000n 2000e... 0.66ms ±0.21
[662/770] Stage 7 on CPU - 1000n 3000e... 0.64ms ±0.12
[663/770] Stage 7 on CPU - 1000n 5000e... 0.62ms ±0.11
[664/770] Stage 7 on CPU - 1000n 7000e... 0.65ms ±0.17
[665/770] Stage 7 on CPU - 1000n 10000e... 0.61ms ±0.11
[666/770] Stage 7 on CPU - 5000n 10000e... 2.84ms ±0.40
[667/770] Stage 7 on CPU - 5000n 15000e... 2.66ms ±0.22
[668/770] Stage 7 on CPU - 5000n 25000e... 2.79ms ±0.51
[669/770] Stage 7 on CPU - 5000n 35000e... 2.87ms ±0.46
[670/770] Stage 7 on CPU - 5000n 50000e... 2.96ms ±0.57
[671/770] Stage 7 on CPU - 10000n 20000e... 5.71ms ±0.55
[672/770] Stage 7 on CPU - 10000n 30000e... 5.94ms ±0.82
[673/770] Stage 7 on CPU - 10000n 50000e... 5.92ms ±0.96
[674/770] Stage 7 on CPU - 10000n 70000e... 5.72ms ±0.77
[675/770] Stage 7 on CPU - 10000n 100000e... 6.12ms ±0.99
[676/770] Stage 7 on CPU - 20000n 40000e... 14.09ms ±1.90
[677/770] Stage 7 on CPU - 20000n 60000e... 13.28ms ±1.48
[678/770] Stage 7 on CPU - 20000n 100000e... 12.93ms ±2.17
[679/770] Stage 7 on CPU - 20000n 140000e... 12.52ms ±1.13
[680/770] Stage 7 on CPU - 20000n 200000e... 12.46ms ±1.34
[681/770] Stage 7 on CPU - 30000n 60000e... 18.78ms ±1.84
[682/770] Stage 7 on CPU - 30000n 90000e... 25.26ms ±5.62
[683/770] Stage 7 on CPU - 30000n 150000e... 24.27ms ±2.22
[684/770] Stage 7 on CPU - 30000n 210000e... 23.86ms ±2.44
[685/770] Stage 7 on CPU - 30000n 300000e... 24.25ms ±2.17
[686/770] Stage 7 on CPU - 40000n 80000e... 45.80ms ±36.95
[687/770] Stage 7 on CPU - 40000n 120000e... 31.74ms ±3.18
[688/770] Stage 7 on CPU - 40000n 200000e... 31.19ms ±3.51
[689/770] Stage 7 on CPU - 40000n 280000e... 32.91ms ±2.55
[690/770] Stage 7 on CPU - 40000n 400000e... 33.46ms ±6.40
[691/770] Stage 7 on CPU - 50000n 100000e... 39.84ms ±2.64
[692/770] Stage 7 on CPU - 50000n 150000e... 39.89ms ±2.40
[693/770] Stage 7 on CPU - 50000n 250000e... 40.14ms ±2.76
[694/770] Stage 7 on CPU - 50000n 350000e... 41.32ms ±5.50
[695/770] Stage 7 on CPU - 50000n 500000e... 32.21ms ±2.60
[696/770] Stage 7 on CPU - 60000n 120000e... 37.76ms ±2.66
[697/770] Stage 7 on CPU - 60000n 180000e... 38.71ms ±3.04
[698/770] Stage 7 on CPU - 60000n 300000e... 38.03ms ±2.32
[699/770] Stage 7 on CPU - 60000n 420000e... 38.45ms ±2.91
[700/770] Stage 7 on CPU - 60000n 600000e... 37.28ms ±2.13
[701/770] Stage 7 on CPU - 80000n 160000e... 53.48ms ±10.65
[702/770] Stage 7 on CPU - 80000n 240000e... 52.28ms ±3.68
[703/770] Stage 7 on CPU - 80000n 400000e... 52.74ms ±4.26
[704/770] Stage 7 on CPU - 80000n 560000e... 53.92ms ±4.39
[705/770] Stage 7 on CPU - 80000n 800000e... 60.69ms ±5.03
[706/770] Stage 7 on CPU - 100000n 200000e... 73.83ms ±4.76
[707/770] Stage 7 on CPU - 100000n 300000e... 73.90ms ±11.62
[708/770] Stage 7 on CPU - 100000n 500000e... 72.55ms ±3.97
[709/770] Stage 7 on CPU - 100000n 700000e... 72.35ms ±3.09
[710/770] Stage 7 on CPU - 100000n 1000000e... 71.43ms ±5.20
[711/770] Stage 7 on CPU - 150000n 300000e... 97.79ms ±5.77
[712/770] Stage 7 on CPU - 150000n 450000e... 97.66ms ±5.68
[713/770] Stage 7 on CPU - 150000n 750000e... 96.59ms ±5.53
[714/770] Stage 7 on CPU - 150000n 1050000e... 98.76ms ±6.05
[715/770] Stage 7 on CPU - 150000n 1500000e... 112.05ms ±7.29
[716/770] Stage 7 on GPU - 1000n 2000e... 1.26ms ±0.20
[717/770] Stage 7 on GPU - 1000n 3000e... 1.13ms ±0.12
[718/770] Stage 7 on GPU - 1000n 5000e... 1.25ms ±0.50
[719/770] Stage 7 on GPU - 1000n 7000e... 1.52ms ±1.21
[720/770] Stage 7 on GPU - 1000n 10000e... 1.39ms ±0.34
[721/770] Stage 7 on GPU - 5000n 10000e... 5.46ms ±1.12
[722/770] Stage 7 on GPU - 5000n 15000e... 5.05ms ±0.78
[723/770] Stage 7 on GPU - 5000n 25000e... 5.50ms ±0.85
[724/770] Stage 7 on GPU - 5000n 35000e... 5.04ms ±0.71
[725/770] Stage 7 on GPU - 5000n 50000e... 5.25ms ±1.10
[726/770] Stage 7 on GPU - 10000n 20000e... 9.95ms ±0.94
[727/770] Stage 7 on GPU - 10000n 30000e... 10.16ms ±1.55
[728/770] Stage 7 on GPU - 10000n 50000e... 9.84ms ±1.22
[729/770] Stage 7 on GPU - 10000n 70000e... 9.95ms ±1.03
[730/770] Stage 7 on GPU - 10000n 100000e... 10.27ms ±1.11
[731/770] Stage 7 on GPU - 20000n 40000e... 18.57ms ±1.20
[732/770] Stage 7 on GPU - 20000n 60000e... 18.99ms ±1.11
[733/770] Stage 7 on GPU - 20000n 100000e... 18.23ms ±1.36
[734/770] Stage 7 on GPU - 20000n 140000e... 18.54ms ±1.23
[735/770] Stage 7 on GPU - 20000n 200000e... 19.36ms ±2.21
[736/770] Stage 7 on GPU - 30000n 60000e... 29.30ms ±2.68
[737/770] Stage 7 on GPU - 30000n 90000e... 28.44ms ±1.62
[738/770] Stage 7 on GPU - 30000n 150000e... 28.79ms ±2.16
[739/770] Stage 7 on GPU - 30000n 210000e... 28.76ms ±1.92
[740/770] Stage 7 on GPU - 30000n 300000e... 28.65ms ±1.51
[741/770] Stage 7 on GPU - 40000n 80000e... 37.71ms ±2.16
[742/770] Stage 7 on GPU - 40000n 120000e... 37.52ms ±1.76
[743/770] Stage 7 on GPU - 40000n 200000e... 38.37ms ±6.52
[744/770] Stage 7 on GPU - 40000n 280000e... 37.72ms ±2.43
[745/770] Stage 7 on GPU - 40000n 400000e... 37.13ms ±2.20
[746/770] Stage 7 on GPU - 50000n 100000e... 47.69ms ±2.76
[747/770] Stage 7 on GPU - 50000n 150000e... 47.38ms ±2.33
[748/770] Stage 7 on GPU - 50000n 250000e... 47.64ms ±2.37
[749/770] Stage 7 on GPU - 50000n 350000e... 47.72ms ±3.03
[750/770] Stage 7 on GPU - 50000n 500000e... 47.26ms ±2.56
[751/770] Stage 7 on GPU - 60000n 120000e... 55.68ms ±2.75
[752/770] Stage 7 on GPU - 60000n 180000e... 54.52ms ±2.00
[753/770] Stage 7 on GPU - 60000n 300000e... 54.81ms ±3.06
[754/770] Stage 7 on GPU - 60000n 420000e... 58.19ms ±6.93
[755/770] Stage 7 on GPU - 60000n 600000e... 56.08ms ±3.87
[756/770] Stage 7 on GPU - 80000n 160000e... 87.26ms ±13.13
[757/770] Stage 7 on GPU - 80000n 240000e... 84.62ms ±11.43
[758/770] Stage 7 on GPU - 80000n 400000e... 84.61ms ±10.58
[759/770] Stage 7 on GPU - 80000n 560000e... 83.46ms ±9.84
[760/770] Stage 7 on GPU - 80000n 800000e... 87.49ms ±10.44
[761/770] Stage 7 on GPU - 100000n 200000e... 102.26ms ±10.61
[762/770] Stage 7 on GPU - 100000n 300000e... 100.83ms ±9.52
[763/770] Stage 7 on GPU - 100000n 500000e... 100.94ms ±11.60
[764/770] Stage 7 on GPU - 100000n 700000e... 100.72ms ±10.23
[765/770] Stage 7 on GPU - 100000n 1000000e... 102.31ms ±12.72
[766/770] Stage 7 on GPU - 150000n 300000e... 152.10ms ±20.15
[767/770] Stage 7 on GPU - 150000n 450000e... 149.09ms ±14.89
[768/770] Stage 7 on GPU - 150000n 750000e... 147.78ms ±13.44
[769/770] Stage 7 on GPU - 150000n 1050000e... 149.97ms ±15.19
[770/770] Stage 7 on GPU - 150000n 1500000e... 153.64ms ±15.95

✓ Measured 770 configurations for CPU+GPU

[Step 3/6] Saving CPU/GPU checkpoint...
✓ Checkpoint saved: checkpoint_cpugpu.json (770 entries)

✓ Phase 1 completed! CPU/GPU data saved.
  If NPU fails later, you still have CPU/GPU results.

======================================================================
PHASE 2: NPU Processing
======================================================================

[Step 4/6] Exporting NPU static models...

======================================================================
=== Exporting NPU Static Models ===
======================================================================
Total: 5 stages × 55 sizes = 275 models (skipping Stage 3/4)

[1/275] Stage 1 - NPU n1000_e2000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[2/275] Stage 1 - NPU n1000_e3000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[3/275] Stage 1 - NPU n1000_e5000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[4/275] Stage 1 - NPU n1000_e7000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[5/275] Stage 1 - NPU n1000_e10000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[6/275] Stage 1 - NPU n5000_e10000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[7/275] Stage 1 - NPU n5000_e15000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[8/275] Stage 1 - NPU n5000_e25000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[9/275] Stage 1 - NPU n5000_e35000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[10/275] Stage 1 - NPU n5000_e50000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[11/275] Stage 1 - NPU n10000_e20000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[12/275] Stage 1 - NPU n10000_e30000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[13/275] Stage 1 - NPU n10000_e50000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[14/275] Stage 1 - NPU n10000_e70000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[15/275] Stage 1 - NPU n10000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[16/275] Stage 1 - NPU n20000_e40000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[17/275] Stage 1 - NPU n20000_e60000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[18/275] Stage 1 - NPU n20000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[19/275] Stage 1 - NPU n20000_e140000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[20/275] Stage 1 - NPU n20000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[21/275] Stage 1 - NPU n30000_e60000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[22/275] Stage 1 - NPU n30000_e90000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[23/275] Stage 1 - NPU n30000_e150000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[24/275] Stage 1 - NPU n30000_e210000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[25/275] Stage 1 - NPU n30000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[26/275] Stage 1 - NPU n40000_e80000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[27/275] Stage 1 - NPU n40000_e120000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[28/275] Stage 1 - NPU n40000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[29/275] Stage 1 - NPU n40000_e280000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[30/275] Stage 1 - NPU n40000_e400000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[31/275] Stage 1 - NPU n50000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[32/275] Stage 1 - NPU n50000_e150000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[33/275] Stage 1 - NPU n50000_e250000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[34/275] Stage 1 - NPU n50000_e350000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[35/275] Stage 1 - NPU n50000_e500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[36/275] Stage 1 - NPU n60000_e120000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[37/275] Stage 1 - NPU n60000_e180000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[38/275] Stage 1 - NPU n60000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[39/275] Stage 1 - NPU n60000_e420000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[40/275] Stage 1 - NPU n60000_e600000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[41/275] Stage 1 - NPU n80000_e160000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[42/275] Stage 1 - NPU n80000_e240000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[43/275] Stage 1 - NPU n80000_e400000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[44/275] Stage 1 - NPU n80000_e560000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[45/275] Stage 1 - NPU n80000_e800000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[46/275] Stage 1 - NPU n100000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[47/275] Stage 1 - NPU n100000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[48/275] Stage 1 - NPU n100000_e500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[49/275] Stage 1 - NPU n100000_e700000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[50/275] Stage 1 - NPU n100000_e1000000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[51/275] Stage 1 - NPU n150000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[52/275] Stage 1 - NPU n150000_e450000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[53/275] Stage 1 - NPU n150000_e750000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[54/275] Stage 1 - NPU n150000_e1050000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[55/275] Stage 1 - NPU n150000_e1500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[56/275] Stage 2 - NPU n1000_e2000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[57/275] Stage 2 - NPU n1000_e3000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[58/275] Stage 2 - NPU n1000_e5000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[59/275] Stage 2 - NPU n1000_e7000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[60/275] Stage 2 - NPU n1000_e10000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[61/275] Stage 2 - NPU n5000_e10000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[62/275] Stage 2 - NPU n5000_e15000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[63/275] Stage 2 - NPU n5000_e25000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[64/275] Stage 2 - NPU n5000_e35000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[65/275] Stage 2 - NPU n5000_e50000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[66/275] Stage 2 - NPU n10000_e20000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[67/275] Stage 2 - NPU n10000_e30000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[68/275] Stage 2 - NPU n10000_e50000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[69/275] Stage 2 - NPU n10000_e70000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[70/275] Stage 2 - NPU n10000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[71/275] Stage 2 - NPU n20000_e40000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[72/275] Stage 2 - NPU n20000_e60000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[73/275] Stage 2 - NPU n20000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[74/275] Stage 2 - NPU n20000_e140000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[75/275] Stage 2 - NPU n20000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[76/275] Stage 2 - NPU n30000_e60000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[77/275] Stage 2 - NPU n30000_e90000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[78/275] Stage 2 - NPU n30000_e150000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[79/275] Stage 2 - NPU n30000_e210000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[80/275] Stage 2 - NPU n30000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[81/275] Stage 2 - NPU n40000_e80000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[82/275] Stage 2 - NPU n40000_e120000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[83/275] Stage 2 - NPU n40000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[84/275] Stage 2 - NPU n40000_e280000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[85/275] Stage 2 - NPU n40000_e400000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[86/275] Stage 2 - NPU n50000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[87/275] Stage 2 - NPU n50000_e150000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[88/275] Stage 2 - NPU n50000_e250000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[89/275] Stage 2 - NPU n50000_e350000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[90/275] Stage 2 - NPU n50000_e500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[91/275] Stage 2 - NPU n60000_e120000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[92/275] Stage 2 - NPU n60000_e180000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[93/275] Stage 2 - NPU n60000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[94/275] Stage 2 - NPU n60000_e420000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[95/275] Stage 2 - NPU n60000_e600000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[96/275] Stage 2 - NPU n80000_e160000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[97/275] Stage 2 - NPU n80000_e240000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[98/275] Stage 2 - NPU n80000_e400000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[99/275] Stage 2 - NPU n80000_e560000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[100/275] Stage 2 - NPU n80000_e800000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[101/275] Stage 2 - NPU n100000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[102/275] Stage 2 - NPU n100000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[103/275] Stage 2 - NPU n100000_e500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[104/275] Stage 2 - NPU n100000_e700000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[105/275] Stage 2 - NPU n100000_e1000000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[106/275] Stage 2 - NPU n150000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[107/275] Stage 2 - NPU n150000_e450000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[108/275] Stage 2 - NPU n150000_e750000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[109/275] Stage 2 - NPU n150000_e1050000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[110/275] Stage 2 - NPU n150000_e1500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[111/275] Stage 5 - NPU n1000_e2000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[112/275] Stage 5 - NPU n1000_e3000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[113/275] Stage 5 - NPU n1000_e5000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[114/275] Stage 5 - NPU n1000_e7000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[115/275] Stage 5 - NPU n1000_e10000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[116/275] Stage 5 - NPU n5000_e10000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[117/275] Stage 5 - NPU n5000_e15000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[118/275] Stage 5 - NPU n5000_e25000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[119/275] Stage 5 - NPU n5000_e35000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[120/275] Stage 5 - NPU n5000_e50000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[121/275] Stage 5 - NPU n10000_e20000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[122/275] Stage 5 - NPU n10000_e30000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[123/275] Stage 5 - NPU n10000_e50000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[124/275] Stage 5 - NPU n10000_e70000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[125/275] Stage 5 - NPU n10000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[126/275] Stage 5 - NPU n20000_e40000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[127/275] Stage 5 - NPU n20000_e60000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[128/275] Stage 5 - NPU n20000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[129/275] Stage 5 - NPU n20000_e140000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[130/275] Stage 5 - NPU n20000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[131/275] Stage 5 - NPU n30000_e60000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[132/275] Stage 5 - NPU n30000_e90000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[133/275] Stage 5 - NPU n30000_e150000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[134/275] Stage 5 - NPU n30000_e210000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[135/275] Stage 5 - NPU n30000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[136/275] Stage 5 - NPU n40000_e80000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[137/275] Stage 5 - NPU n40000_e120000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[138/275] Stage 5 - NPU n40000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[139/275] Stage 5 - NPU n40000_e280000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[140/275] Stage 5 - NPU n40000_e400000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[141/275] Stage 5 - NPU n50000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[142/275] Stage 5 - NPU n50000_e150000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[143/275] Stage 5 - NPU n50000_e250000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[144/275] Stage 5 - NPU n50000_e350000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[145/275] Stage 5 - NPU n50000_e500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[146/275] Stage 5 - NPU n60000_e120000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[147/275] Stage 5 - NPU n60000_e180000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[148/275] Stage 5 - NPU n60000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[149/275] Stage 5 - NPU n60000_e420000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[150/275] Stage 5 - NPU n60000_e600000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[151/275] Stage 5 - NPU n80000_e160000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[152/275] Stage 5 - NPU n80000_e240000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[153/275] Stage 5 - NPU n80000_e400000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[154/275] Stage 5 - NPU n80000_e560000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[155/275] Stage 5 - NPU n80000_e800000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[156/275] Stage 5 - NPU n100000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[157/275] Stage 5 - NPU n100000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[158/275] Stage 5 - NPU n100000_e500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[159/275] Stage 5 - NPU n100000_e700000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[160/275] Stage 5 - NPU n100000_e1000000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[161/275] Stage 5 - NPU n150000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[162/275] Stage 5 - NPU n150000_e450000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[163/275] Stage 5 - NPU n150000_e750000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[164/275] Stage 5 - NPU n150000_e1050000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[165/275] Stage 5 - NPU n150000_e1500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[166/275] Stage 6 - NPU n1000_e2000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[167/275] Stage 6 - NPU n1000_e3000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[168/275] Stage 6 - NPU n1000_e5000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[169/275] Stage 6 - NPU n1000_e7000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[170/275] Stage 6 - NPU n1000_e10000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[171/275] Stage 6 - NPU n5000_e10000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[172/275] Stage 6 - NPU n5000_e15000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[173/275] Stage 6 - NPU n5000_e25000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[174/275] Stage 6 - NPU n5000_e35000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[175/275] Stage 6 - NPU n5000_e50000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[176/275] Stage 6 - NPU n10000_e20000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[177/275] Stage 6 - NPU n10000_e30000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[178/275] Stage 6 - NPU n10000_e50000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[179/275] Stage 6 - NPU n10000_e70000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[180/275] Stage 6 - NPU n10000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[181/275] Stage 6 - NPU n20000_e40000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[182/275] Stage 6 - NPU n20000_e60000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[183/275] Stage 6 - NPU n20000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[184/275] Stage 6 - NPU n20000_e140000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[185/275] Stage 6 - NPU n20000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[186/275] Stage 6 - NPU n30000_e60000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[187/275] Stage 6 - NPU n30000_e90000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[188/275] Stage 6 - NPU n30000_e150000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[189/275] Stage 6 - NPU n30000_e210000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[190/275] Stage 6 - NPU n30000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[191/275] Stage 6 - NPU n40000_e80000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[192/275] Stage 6 - NPU n40000_e120000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[193/275] Stage 6 - NPU n40000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[194/275] Stage 6 - NPU n40000_e280000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[195/275] Stage 6 - NPU n40000_e400000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[196/275] Stage 6 - NPU n50000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[197/275] Stage 6 - NPU n50000_e150000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[198/275] Stage 6 - NPU n50000_e250000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[199/275] Stage 6 - NPU n50000_e350000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[200/275] Stage 6 - NPU n50000_e500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[201/275] Stage 6 - NPU n60000_e120000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[202/275] Stage 6 - NPU n60000_e180000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[203/275] Stage 6 - NPU n60000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[204/275] Stage 6 - NPU n60000_e420000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[205/275] Stage 6 - NPU n60000_e600000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[206/275] Stage 6 - NPU n80000_e160000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[207/275] Stage 6 - NPU n80000_e240000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[208/275] Stage 6 - NPU n80000_e400000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[209/275] Stage 6 - NPU n80000_e560000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[210/275] Stage 6 - NPU n80000_e800000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[211/275] Stage 6 - NPU n100000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[212/275] Stage 6 - NPU n100000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[213/275] Stage 6 - NPU n100000_e500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[214/275] Stage 6 - NPU n100000_e700000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[215/275] Stage 6 - NPU n100000_e1000000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[216/275] Stage 6 - NPU n150000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[217/275] Stage 6 - NPU n150000_e450000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[218/275] Stage 6 - NPU n150000_e750000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[219/275] Stage 6 - NPU n150000_e1050000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[220/275] Stage 6 - NPU n150000_e1500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[221/275] Stage 7 - NPU n1000_e2000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[222/275] Stage 7 - NPU n1000_e3000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[223/275] Stage 7 - NPU n1000_e5000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[224/275] Stage 7 - NPU n1000_e7000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[225/275] Stage 7 - NPU n1000_e10000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[226/275] Stage 7 - NPU n5000_e10000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[227/275] Stage 7 - NPU n5000_e15000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[228/275] Stage 7 - NPU n5000_e25000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[229/275] Stage 7 - NPU n5000_e35000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[230/275] Stage 7 - NPU n5000_e50000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[231/275] Stage 7 - NPU n10000_e20000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[232/275] Stage 7 - NPU n10000_e30000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[233/275] Stage 7 - NPU n10000_e50000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[234/275] Stage 7 - NPU n10000_e70000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[235/275] Stage 7 - NPU n10000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[236/275] Stage 7 - NPU n20000_e40000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[237/275] Stage 7 - NPU n20000_e60000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[238/275] Stage 7 - NPU n20000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[239/275] Stage 7 - NPU n20000_e140000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[240/275] Stage 7 - NPU n20000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[241/275] Stage 7 - NPU n30000_e60000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[242/275] Stage 7 - NPU n30000_e90000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[243/275] Stage 7 - NPU n30000_e150000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[244/275] Stage 7 - NPU n30000_e210000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[245/275] Stage 7 - NPU n30000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[246/275] Stage 7 - NPU n40000_e80000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[247/275] Stage 7 - NPU n40000_e120000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[248/275] Stage 7 - NPU n40000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[249/275] Stage 7 - NPU n40000_e280000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[250/275] Stage 7 - NPU n40000_e400000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[251/275] Stage 7 - NPU n50000_e100000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[252/275] Stage 7 - NPU n50000_e150000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[253/275] Stage 7 - NPU n50000_e250000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[254/275] Stage 7 - NPU n50000_e350000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[255/275] Stage 7 - NPU n50000_e500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[256/275] Stage 7 - NPU n60000_e120000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[257/275] Stage 7 - NPU n60000_e180000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[258/275] Stage 7 - NPU n60000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[259/275] Stage 7 - NPU n60000_e420000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[260/275] Stage 7 - NPU n60000_e600000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[261/275] Stage 7 - NPU n80000_e160000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[262/275] Stage 7 - NPU n80000_e240000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[263/275] Stage 7 - NPU n80000_e400000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[264/275] Stage 7 - NPU n80000_e560000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[265/275] Stage 7 - NPU n80000_e800000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[266/275] Stage 7 - NPU n100000_e200000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[267/275] Stage 7 - NPU n100000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[268/275] Stage 7 - NPU n100000_e500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[269/275] Stage 7 - NPU n100000_e700000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[270/275] Stage 7 - NPU n100000_e1000000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[271/275] Stage 7 - NPU n150000_e300000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[272/275] Stage 7 - NPU n150000_e450000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[273/275] Stage 7 - NPU n150000_e750000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[274/275] Stage 7 - NPU n150000_e1050000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓
[275/275] Stage 7 - NPU n150000_e1500000 [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
✓

✓ All NPU static models exported (275 files)

[Step 5/6] Measuring NPU latencies...

======================================================================
=== Measuring Latencies (NPU) ===
======================================================================
[1/275] Stage 1 on NPU - 1000n 2000e... 4.72ms ±1.49
[2/275] Stage 1 on NPU - 1000n 3000e... 6.68ms ±1.41
[3/275] Stage 1 on NPU - 1000n 5000e... 15.66ms ±7.23
[4/275] Stage 1 on NPU - 1000n 7000e... 11.18ms ±1.76
[5/275] Stage 1 on NPU - 1000n 10000e... 19.78ms ±6.28
[6/275] Stage 1 on NPU - 5000n 10000e... 19.94ms ±2.74
[7/275] Stage 1 on NPU - 5000n 15000e... 31.09ms ±2.29
[8/275] Stage 1 on NPU - 5000n 25000e... 61.39ms ±4.13
[9/275] Stage 1 on NPU - 5000n 35000e... 96.50ms ±3.87
[10/275] Stage 1 on NPU - 5000n 50000e... 166.67ms ±9.76
[11/275] Stage 1 on NPU - 10000n 20000e... 54.51ms ±3.61
[12/275] Stage 1 on NPU - 10000n 30000e... 91.30ms ±7.51
[13/275] Stage 1 on NPU - 10000n 50000e... 190.55ms ±16.12
[14/275] Stage 1 on NPU - 10000n 70000e... 332.17ms ±14.36
[15/275] Stage 1 on NPU - 10000n 100000e... 605.17ms ±45.79
[16/275] Stage 1 on NPU - 20000n 40000e... 135.98ms ±5.41
[17/275] Stage 1 on NPU - 20000n 60000e... 258.73ms ±7.18
[18/275] Stage 1 on NPU - 20000n 100000e... 625.51ms ±10.67
[19/275] Stage 1 on NPU - 20000n 140000e... 1160.64ms ±8.79
[20/275] Stage 1 on NPU - 20000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\infer_request.cpp:223:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:156:
L0 zeCommandQueueExecuteCommandLists result: ZE_RESULT_ERROR_DEVICE_LOST, code 0x70000001 - device hung, reset, was removed, or driver update occurred

, using PyTorch
22.06ms ±9.63
[21/275] Stage 1 on NPU - 30000n 60000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
9.21ms ±1.04
[22/275] Stage 1 on NPU - 30000n 90000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
13.48ms ±2.90
[23/275] Stage 1 on NPU - 30000n 150000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
18.60ms ±4.36
[24/275] Stage 1 on NPU - 30000n 210000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
24.64ms ±5.77
[25/275] Stage 1 on NPU - 30000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
32.29ms ±6.73
[26/275] Stage 1 on NPU - 40000n 80000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
12.39ms ±2.39
[27/275] Stage 1 on NPU - 40000n 120000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
16.78ms ±3.62
[28/275] Stage 1 on NPU - 40000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
24.29ms ±5.26
[29/275] Stage 1 on NPU - 40000n 280000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
31.40ms ±5.94
[30/275] Stage 1 on NPU - 40000n 400000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
44.39ms ±8.05
[31/275] Stage 1 on NPU - 50000n 100000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
15.09ms ±2.78
[32/275] Stage 1 on NPU - 50000n 150000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
19.96ms ±4.61
[33/275] Stage 1 on NPU - 50000n 250000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
29.53ms ±5.47
[34/275] Stage 1 on NPU - 50000n 350000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
41.17ms ±7.10
[35/275] Stage 1 on NPU - 50000n 500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
58.42ms ±10.99
[36/275] Stage 1 on NPU - 60000n 120000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
17.59ms ±3.89
[37/275] Stage 1 on NPU - 60000n 180000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
23.09ms ±4.50
[38/275] Stage 1 on NPU - 60000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
36.28ms ±6.56
[39/275] Stage 1 on NPU - 60000n 420000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
50.74ms ±9.24
[40/275] Stage 1 on NPU - 60000n 600000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
91.29ms ±0.73
[41/275] Stage 1 on NPU - 80000n 160000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
22.22ms ±6.38
[42/275] Stage 1 on NPU - 80000n 240000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
30.87ms ±3.03
[43/275] Stage 1 on NPU - 80000n 400000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
46.67ms ±5.20
[44/275] Stage 1 on NPU - 80000n 560000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
86.67ms ±1.91
[45/275] Stage 1 on NPU - 80000n 800000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
123.47ms ±1.07
[46/275] Stage 1 on NPU - 100000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
25.84ms ±5.13
[47/275] Stage 1 on NPU - 100000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
36.86ms ±5.81
[48/275] Stage 1 on NPU - 100000n 500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
62.16ms ±9.94
[49/275] Stage 1 on NPU - 100000n 700000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
109.48ms ±1.03
[50/275] Stage 1 on NPU - 100000n 1000000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
156.11ms ±1.39
[51/275] Stage 1 on NPU - 150000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
37.87ms ±6.55
[52/275] Stage 1 on NPU - 150000n 450000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
56.57ms ±8.77
[53/275] Stage 1 on NPU - 150000n 750000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
108.60ms ±6.40
[54/275] Stage 1 on NPU - 150000n 1050000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
156.95ms ±11.09
[55/275] Stage 1 on NPU - 150000n 1500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
255.10ms ±41.15
[56/275] Stage 2 on NPU - 1000n 2000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[57/275] Stage 2 on NPU - 1000n 3000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[58/275] Stage 2 on NPU - 1000n 5000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[59/275] Stage 2 on NPU - 1000n 7000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[60/275] Stage 2 on NPU - 1000n 10000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[61/275] Stage 2 on NPU - 5000n 10000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[62/275] Stage 2 on NPU - 5000n 15000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[63/275] Stage 2 on NPU - 5000n 25000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[64/275] Stage 2 on NPU - 5000n 35000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[65/275] Stage 2 on NPU - 5000n 50000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[66/275] Stage 2 on NPU - 10000n 20000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[67/275] Stage 2 on NPU - 10000n 30000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[68/275] Stage 2 on NPU - 10000n 50000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[69/275] Stage 2 on NPU - 10000n 70000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[70/275] Stage 2 on NPU - 10000n 100000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[71/275] Stage 2 on NPU - 20000n 40000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[72/275] Stage 2 on NPU - 20000n 60000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[73/275] Stage 2 on NPU - 20000n 100000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[74/275] Stage 2 on NPU - 20000n 140000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[75/275] Stage 2 on NPU - 20000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[76/275] Stage 2 on NPU - 30000n 60000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[77/275] Stage 2 on NPU - 30000n 90000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[78/275] Stage 2 on NPU - 30000n 150000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[79/275] Stage 2 on NPU - 30000n 210000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[80/275] Stage 2 on NPU - 30000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[81/275] Stage 2 on NPU - 40000n 80000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[82/275] Stage 2 on NPU - 40000n 120000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[83/275] Stage 2 on NPU - 40000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[84/275] Stage 2 on NPU - 40000n 280000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[85/275] Stage 2 on NPU - 40000n 400000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[86/275] Stage 2 on NPU - 50000n 100000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[87/275] Stage 2 on NPU - 50000n 150000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[88/275] Stage 2 on NPU - 50000n 250000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[89/275] Stage 2 on NPU - 50000n 350000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[90/275] Stage 2 on NPU - 50000n 500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[91/275] Stage 2 on NPU - 60000n 120000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[92/275] Stage 2 on NPU - 60000n 180000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[93/275] Stage 2 on NPU - 60000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[94/275] Stage 2 on NPU - 60000n 420000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[95/275] Stage 2 on NPU - 60000n 600000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[96/275] Stage 2 on NPU - 80000n 160000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[97/275] Stage 2 on NPU - 80000n 240000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[98/275] Stage 2 on NPU - 80000n 400000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[99/275] Stage 2 on NPU - 80000n 560000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[100/275] Stage 2 on NPU - 80000n 800000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[101/275] Stage 2 on NPU - 100000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[102/275] Stage 2 on NPU - 100000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[103/275] Stage 2 on NPU - 100000n 500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[104/275] Stage 2 on NPU - 100000n 700000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[105/275] Stage 2 on NPU - 100000n 1000000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[106/275] Stage 2 on NPU - 150000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[107/275] Stage 2 on NPU - 150000n 450000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[108/275] Stage 2 on NPU - 150000n 750000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[109/275] Stage 2 on NPU - 150000n 1050000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[110/275] Stage 2 on NPU - 150000n 1500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.00ms ±0.00
[111/275] Stage 5 on NPU - 1000n 2000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.19ms ±0.19
[112/275] Stage 5 on NPU - 1000n 3000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.08ms ±0.01
[113/275] Stage 5 on NPU - 1000n 5000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.07ms ±0.01
[114/275] Stage 5 on NPU - 1000n 7000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.07ms ±0.04
[115/275] Stage 5 on NPU - 1000n 10000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.08ms ±0.09
[116/275] Stage 5 on NPU - 5000n 10000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.32ms ±0.22
[117/275] Stage 5 on NPU - 5000n 15000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.41ms ±0.24
[118/275] Stage 5 on NPU - 5000n 25000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.55ms ±0.37
[119/275] Stage 5 on NPU - 5000n 35000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.46ms ±0.28
[120/275] Stage 5 on NPU - 5000n 50000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.32ms ±0.19
[121/275] Stage 5 on NPU - 10000n 20000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
1.11ms ±0.35
[122/275] Stage 5 on NPU - 10000n 30000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
1.25ms ±0.47
[123/275] Stage 5 on NPU - 10000n 50000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
1.02ms ±0.24
[124/275] Stage 5 on NPU - 10000n 70000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
1.15ms ±0.29
[125/275] Stage 5 on NPU - 10000n 100000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
1.71ms ±0.62
[126/275] Stage 5 on NPU - 20000n 40000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
2.79ms ±0.49
[127/275] Stage 5 on NPU - 20000n 60000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
3.38ms ±1.19
[128/275] Stage 5 on NPU - 20000n 100000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
1.99ms ±0.34
[129/275] Stage 5 on NPU - 20000n 140000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
2.70ms ±0.35
[130/275] Stage 5 on NPU - 20000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
2.02ms ±0.27
[131/275] Stage 5 on NPU - 30000n 60000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.22ms ±0.77
[132/275] Stage 5 on NPU - 30000n 90000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.06ms ±0.40
[133/275] Stage 5 on NPU - 30000n 150000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.20ms ±0.74
[134/275] Stage 5 on NPU - 30000n 210000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.00ms ±0.91
[135/275] Stage 5 on NPU - 30000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.18ms ±0.65
[136/275] Stage 5 on NPU - 40000n 80000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
5.35ms ±0.86
[137/275] Stage 5 on NPU - 40000n 120000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
5.31ms ±0.71
[138/275] Stage 5 on NPU - 40000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
5.35ms ±0.87
[139/275] Stage 5 on NPU - 40000n 280000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
5.67ms ±0.73
[140/275] Stage 5 on NPU - 40000n 400000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
5.40ms ±0.67
[141/275] Stage 5 on NPU - 50000n 100000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.79ms ±0.79
[142/275] Stage 5 on NPU - 50000n 150000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
6.26ms ±1.25
[143/275] Stage 5 on NPU - 50000n 250000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
6.81ms ±2.01
[144/275] Stage 5 on NPU - 50000n 350000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
6.77ms ±1.51
[145/275] Stage 5 on NPU - 50000n 500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.93ms ±0.46
[146/275] Stage 5 on NPU - 60000n 120000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
7.29ms ±1.29
[147/275] Stage 5 on NPU - 60000n 180000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
8.40ms ±2.89
[148/275] Stage 5 on NPU - 60000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
7.46ms ±1.26
[149/275] Stage 5 on NPU - 60000n 420000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
7.38ms ±1.99
[150/275] Stage 5 on NPU - 60000n 600000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
7.57ms ±1.69
[151/275] Stage 5 on NPU - 80000n 160000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
9.65ms ±1.91
[152/275] Stage 5 on NPU - 80000n 240000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
9.81ms ±2.25
[153/275] Stage 5 on NPU - 80000n 400000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
9.20ms ±1.97
[154/275] Stage 5 on NPU - 80000n 560000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
9.57ms ±2.25
[155/275] Stage 5 on NPU - 80000n 800000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
9.60ms ±1.98
[156/275] Stage 5 on NPU - 100000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
10.58ms ±2.06
[157/275] Stage 5 on NPU - 100000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
12.36ms ±3.95
[158/275] Stage 5 on NPU - 100000n 500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
10.50ms ±1.88
[159/275] Stage 5 on NPU - 100000n 700000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
11.46ms ±2.04
[160/275] Stage 5 on NPU - 100000n 1000000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
11.32ms ±3.35
[161/275] Stage 5 on NPU - 150000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
17.07ms ±4.93
[162/275] Stage 5 on NPU - 150000n 450000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
16.62ms ±3.26
[163/275] Stage 5 on NPU - 150000n 750000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
16.28ms ±3.16
[164/275] Stage 5 on NPU - 150000n 1050000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
16.57ms ±3.27
[165/275] Stage 5 on NPU - 150000n 1500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
16.87ms ±3.67
[166/275] Stage 6 on NPU - 1000n 2000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.12ms ±1.05
[167/275] Stage 6 on NPU - 1000n 3000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.44ms ±0.93
[168/275] Stage 6 on NPU - 1000n 5000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.44ms ±1.10
[169/275] Stage 6 on NPU - 1000n 7000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.37ms ±0.54
[170/275] Stage 6 on NPU - 1000n 10000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.27ms ±1.17
[171/275] Stage 6 on NPU - 5000n 10000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
17.88ms ±6.00
[172/275] Stage 6 on NPU - 5000n 15000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
17.98ms ±5.32
[173/275] Stage 6 on NPU - 5000n 25000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
18.37ms ±5.67
[174/275] Stage 6 on NPU - 5000n 35000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
18.17ms ±6.04
[175/275] Stage 6 on NPU - 5000n 50000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
18.69ms ±7.24
[176/275] Stage 6 on NPU - 10000n 20000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
29.90ms ±8.81
[177/275] Stage 6 on NPU - 10000n 30000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
29.78ms ±8.73
[178/275] Stage 6 on NPU - 10000n 50000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
30.53ms ±11.96
[179/275] Stage 6 on NPU - 10000n 70000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
30.25ms ±12.07
[180/275] Stage 6 on NPU - 10000n 100000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
29.40ms ±12.32
[181/275] Stage 6 on NPU - 20000n 40000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
46.41ms ±6.51
[182/275] Stage 6 on NPU - 20000n 60000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
43.12ms ±7.65
[183/275] Stage 6 on NPU - 20000n 100000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
45.34ms ±6.38
[184/275] Stage 6 on NPU - 20000n 140000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
44.83ms ±6.41
[185/275] Stage 6 on NPU - 20000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
44.58ms ±7.19
[186/275] Stage 6 on NPU - 30000n 60000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
63.87ms ±6.54
[187/275] Stage 6 on NPU - 30000n 90000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
65.63ms ±6.92
[188/275] Stage 6 on NPU - 30000n 150000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
66.77ms ±6.57
[189/275] Stage 6 on NPU - 30000n 210000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
67.02ms ±7.13
[190/275] Stage 6 on NPU - 30000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
62.21ms ±8.71
[191/275] Stage 6 on NPU - 40000n 80000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
82.77ms ±8.35
[192/275] Stage 6 on NPU - 40000n 120000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
83.76ms ±9.36
[193/275] Stage 6 on NPU - 40000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
86.79ms ±7.06
[194/275] Stage 6 on NPU - 40000n 280000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
93.10ms ±8.62
[195/275] Stage 6 on NPU - 40000n 400000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
86.84ms ±8.39
[196/275] Stage 6 on NPU - 50000n 100000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
101.48ms ±8.74
[197/275] Stage 6 on NPU - 50000n 150000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
102.41ms ±8.56
[198/275] Stage 6 on NPU - 50000n 250000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
102.31ms ±8.06
[199/275] Stage 6 on NPU - 50000n 350000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
100.09ms ±8.39
[200/275] Stage 6 on NPU - 50000n 500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
103.28ms ±9.81
[201/275] Stage 6 on NPU - 60000n 120000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
117.87ms ±7.96
[202/275] Stage 6 on NPU - 60000n 180000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
121.71ms ±12.12
[203/275] Stage 6 on NPU - 60000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
125.36ms ±9.95
[204/275] Stage 6 on NPU - 60000n 420000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
126.25ms ±10.71
[205/275] Stage 6 on NPU - 60000n 600000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
119.50ms ±8.97
[206/275] Stage 6 on NPU - 80000n 160000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
168.49ms ±12.18
[207/275] Stage 6 on NPU - 80000n 240000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
164.74ms ±9.28
[208/275] Stage 6 on NPU - 80000n 400000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
164.54ms ±6.68
[209/275] Stage 6 on NPU - 80000n 560000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
165.16ms ±9.25
[210/275] Stage 6 on NPU - 80000n 800000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
168.12ms ±8.79
[211/275] Stage 6 on NPU - 100000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
198.63ms ±6.52
[212/275] Stage 6 on NPU - 100000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
199.11ms ±7.77
[213/275] Stage 6 on NPU - 100000n 500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
207.16ms ±16.04
[214/275] Stage 6 on NPU - 100000n 700000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
210.81ms ±17.61
[215/275] Stage 6 on NPU - 100000n 1000000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
197.65ms ±7.47
[216/275] Stage 6 on NPU - 150000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
302.75ms ±24.52
[217/275] Stage 6 on NPU - 150000n 450000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
320.86ms ±40.00
[218/275] Stage 6 on NPU - 150000n 750000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
289.43ms ±15.04
[219/275] Stage 6 on NPU - 150000n 1050000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
304.20ms ±28.81
[220/275] Stage 6 on NPU - 150000n 1500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
297.33ms ±17.51
[221/275] Stage 7 on NPU - 1000n 2000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.03ms ±0.00
[222/275] Stage 7 on NPU - 1000n 3000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.04ms ±0.02
[223/275] Stage 7 on NPU - 1000n 5000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.03ms ±0.00
[224/275] Stage 7 on NPU - 1000n 7000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.25ms ±0.24
[225/275] Stage 7 on NPU - 1000n 10000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.24ms ±0.47
[226/275] Stage 7 on NPU - 5000n 10000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
1.08ms ±0.70
[227/275] Stage 7 on NPU - 5000n 15000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.90ms ±0.31
[228/275] Stage 7 on NPU - 5000n 25000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.81ms ±0.22
[229/275] Stage 7 on NPU - 5000n 35000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.91ms ±0.57
[230/275] Stage 7 on NPU - 5000n 50000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
0.78ms ±0.18
[231/275] Stage 7 on NPU - 10000n 20000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
1.75ms ±0.46
[232/275] Stage 7 on NPU - 10000n 30000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
1.74ms ±0.44
[233/275] Stage 7 on NPU - 10000n 50000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
1.42ms ±0.29
[234/275] Stage 7 on NPU - 10000n 70000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
1.70ms ±0.37
[235/275] Stage 7 on NPU - 10000n 100000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
1.37ms ±0.29
[236/275] Stage 7 on NPU - 20000n 40000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
3.73ms ±0.84
[237/275] Stage 7 on NPU - 20000n 60000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
3.23ms ±0.63
[238/275] Stage 7 on NPU - 20000n 100000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
2.26ms ±0.39
[239/275] Stage 7 on NPU - 20000n 140000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
3.22ms ±0.41
[240/275] Stage 7 on NPU - 20000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
2.40ms ±1.69
[241/275] Stage 7 on NPU - 30000n 60000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.79ms ±0.64
[242/275] Stage 7 on NPU - 30000n 90000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.84ms ±0.82
[243/275] Stage 7 on NPU - 30000n 150000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.73ms ±0.72
[244/275] Stage 7 on NPU - 30000n 210000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.94ms ±0.91
[245/275] Stage 7 on NPU - 30000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
4.96ms ±0.78
[246/275] Stage 7 on NPU - 40000n 80000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
6.08ms ±1.19
[247/275] Stage 7 on NPU - 40000n 120000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
6.51ms ±1.37
[248/275] Stage 7 on NPU - 40000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
5.84ms ±1.21
[249/275] Stage 7 on NPU - 40000n 280000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
6.13ms ±1.39
[250/275] Stage 7 on NPU - 40000n 400000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
5.78ms ±0.91
[251/275] Stage 7 on NPU - 50000n 100000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
7.85ms ±1.17
[252/275] Stage 7 on NPU - 50000n 150000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
6.57ms ±1.01
[253/275] Stage 7 on NPU - 50000n 250000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
6.62ms ±1.08
[254/275] Stage 7 on NPU - 50000n 350000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
6.73ms ±1.24
[255/275] Stage 7 on NPU - 50000n 500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
5.57ms ±2.46
[256/275] Stage 7 on NPU - 60000n 120000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
7.93ms ±1.61
[257/275] Stage 7 on NPU - 60000n 180000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
7.85ms ±1.50
[258/275] Stage 7 on NPU - 60000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
7.70ms ±2.24
[259/275] Stage 7 on NPU - 60000n 420000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
7.80ms ±1.54
[260/275] Stage 7 on NPU - 60000n 600000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
7.87ms ±1.46
[261/275] Stage 7 on NPU - 80000n 160000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
9.44ms ±1.69
[262/275] Stage 7 on NPU - 80000n 240000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
9.85ms ±1.81
[263/275] Stage 7 on NPU - 80000n 400000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
9.28ms ±1.76
[264/275] Stage 7 on NPU - 80000n 560000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
9.95ms ±2.29
[265/275] Stage 7 on NPU - 80000n 800000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
9.87ms ±1.83
[266/275] Stage 7 on NPU - 100000n 200000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
11.89ms ±2.31
[267/275] Stage 7 on NPU - 100000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
11.88ms ±2.31
[268/275] Stage 7 on NPU - 100000n 500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
11.82ms ±2.32
[269/275] Stage 7 on NPU - 100000n 700000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
12.63ms ±3.30
[270/275] Stage 7 on NPU - 100000n 1000000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
11.55ms ±1.78
[271/275] Stage 7 on NPU - 150000n 300000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
16.56ms ±1.93
[272/275] Stage 7 on NPU - 150000n 450000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
16.18ms ±1.38
[273/275] Stage 7 on NPU - 150000n 750000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
17.62ms ±2.64
[274/275] Stage 7 on NPU - 150000n 1050000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
17.23ms ±3.05
[275/275] Stage 7 on NPU - 150000n 1500000e...     ⚠ OpenVINO measurement failed: Exception from src\inference\src\cpp\core.cpp:107:
Exception from src\inference\src\dev\plugin.cpp:53:
Exception from src\plugins\intel_npu\src\plugin\src\plugin.cpp:717:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:148:
L0 zeCommandQueueCreate result: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, code 0x70000002 - insufficient host memory to satisfy call



, using PyTorch
17.13ms ±2.41

✓ Measured 275 configurations for NPU

[Step 6/6] Saving NPU checkpoint...
✓ Checkpoint saved: checkpoint_npu.json (275 entries)

✓ Phase 2 completed! NPU data saved.

======================================================================
PHASE 3: Merging and Analyzing
======================================================================
✓ Loaded checkpoint: checkpoint_cpugpu.json (770 entries)
✓ Loaded checkpoint: checkpoint_npu.json (275 entries)
✓ Merged: 770 CPU/GPU + 275 NPU = 1045 total

======================================================================
=== Estimating Bandwidth and Compute Time ===
======================================================================
  CPU Stage 1: 2457.98 MB/s (R²=0.985)
  GPU Stage 1: 2678.34 MB/s (R²=0.998)
  NPU Stage 1: 154668.89 MB/s (R²=0.000)
  CPU Stage 2: 6377.50 MB/s (R²=0.999)
  GPU Stage 2: 4167.32 MB/s (R²=0.999)
  NPU Stage 2: inf MB/s (R²=0.019)
  CPU Stage 3: 1102.28 MB/s (R²=0.977)
  GPU Stage 3: 3773.70 MB/s (R²=0.995)
  ⚠ Insufficient data for NPU Stage 3, skipping regression
  CPU Stage 4: 1181.91 MB/s (R²=0.997)
  GPU Stage 4: 5953.51 MB/s (R²=0.985)
  ⚠ Insufficient data for NPU Stage 4, skipping regression
  CPU Stage 5: 5322.79 MB/s (R²=0.985)
  GPU Stage 5: 2979.24 MB/s (R²=0.992)
  NPU Stage 5: 34298.32 MB/s (R²=0.987)
  CPU Stage 6: 1540.00 MB/s (R²=0.800)
  GPU Stage 6: 3939.65 MB/s (R²=0.999)
  NPU Stage 6: 2898.78 MB/s (R²=0.997)
  CPU Stage 7: 5599.09 MB/s (R²=0.980)
  GPU Stage 7: 3760.56 MB/s (R²=0.997)
  NPU Stage 7: 34269.48 MB/s (R²=0.985)
✓ Bandwidth estimation completed

======================================================================
=== Saving Results ===
======================================================================
  ✓ Saved: C:\Private\Research\GNX_final\GNX\profiling\results\raw_measurements.json
  ✓ Saved: C:\Private\Research\GNX_final\GNX\profiling\results\lookup_table.json
  ✓ Saved: C:\Private\Research\GNX_final\GNX\profiling\results\bandwidth_table.json

================================================================================
GNX Stage Profiling Report (精简版)
================================================================================

Test Configuration:
  - Test cases: 55
  - Node sizes: [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 80000, 100000, 150000]
  - Edge sizes: [2000, 3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 120000, 140000, 150000, 160000, 180000, 200000, 210000, 240000, 250000, 280000, 300000, 350000, 400000, 420000, 450000, 500000, 560000, 600000, 700000, 750000, 800000, 1000000, 1050000, 1500000]
  - Total measurements: 1045

Average Compute Time by Stage (ms):
--------------------------------------------------------------------------------
Stage      CPU             GPU             NPU
--------------------------------------------------------------------------------
Stage 1   4.16            1.65            102.17
Stage 2   3.72            10.52           0.00
Stage 3   30.52           7.38            0.00
Stage 4   0.04            0.68            0.00
Stage 5   0.60            0.34            0.46
Stage 6   25.75           0.78            6.24
Stage 7   2.17            0.50            0.82

Estimated Bandwidth by Stage (MB/s):
--------------------------------------------------------------------------------
Stage 1: CPU=2458.0 | GPU=2678.3 | NPU=154668.9
Stage 2: CPU=6377.5 | GPU=4167.3 | NPU=inf
Stage 3: CPU=1102.3 | GPU=3773.7 | NPU=0.0
Stage 4: CPU=1181.9 | GPU=5953.5 | NPU=0.0
Stage 5: CPU=5322.8 | GPU=2979.2 | NPU=34298.3
Stage 6: CPU=1540.0 | GPU=3939.7 | NPU=2898.8
Stage 7: CPU=5599.1 | GPU=3760.6 | NPU=34269.5

Speedup Analysis (GPU vs CPU, NPU vs CPU):
--------------------------------------------------------------------------------
Stage 1: GPU speedup = 2.53x, NPU speedup = 0.04x
Stage 2: GPU speedup = 0.35x, NPU speedup = 2688.97x
Stage 3: GPU speedup = 4.13x
Stage 4: GPU speedup = 0.05x
Stage 5: GPU speedup = 1.76x, NPU speedup = 1.31x
Stage 6: GPU speedup = 33.03x, NPU speedup = 4.13x
Stage 7: GPU speedup = 4.35x, NPU speedup = 2.66x

================================================================================

✓ Report saved: C:\Private\Research\GNX_final\GNX\profiling\results\profiling_report.txt

======================================================================
✓ Profiling completed successfully!
======================================================================

Results saved in: C:\Private\Research\GNX_final\GNX\profiling\results
  - lookup_table.json      (计算时间)
  - bandwidth_table.json   (PU间带宽)
  - profiling_report.txt   (统计报告)

Checkpoints:
  - checkpoint_cpugpu.json (CPU/GPU数据)
  - checkpoint_npu.json    (NPU数据)

💡 即使NPU失败，您仍然有CPU/GPU的完整数据可用！

========================================================================
Profiling completed!
========================================================================

Results are saved in: profiling\results\
  - lookup_table.json      (Compute time for each configuration)
  - bandwidth_table.json   (Bandwidth estimates per PU)
  - profiling_report.txt   (Human-readable statistics)
