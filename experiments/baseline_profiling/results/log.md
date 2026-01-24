========================================================================
Baseline Profiling - CPU + GPU Testing
========================================================================

Activating MIX environment...

Test Configuration:
  Models: GraphSAGE, GCN, GAT (1 layer each)
  Devices: CPU, GPU
  Test cases: 42 (6 node sizes x 7 edge ratios)

Node sizes: 5k, 10k, 20k, 50k, 80k, 100k
Edge ratios: 10, 25, 40, 50, 60, 75, 100


[Step 1/3] Checking and exporting CPU/GPU models...
  Exporting CPU/GPU dynamic models...
======================================================================
Baseline Profiling - Complete 1-Layer GNN Models
======================================================================
Models: graphsage, gcn, gat
Test cases: 42
Feature dim: 500, Output dim: 256

======================================================================
Exporting CPU/GPU Dynamic Models (Complete 1-Layer GNNs)
======================================================================

--- GRAPHSAGE ---
  Exporting to ONNX: graphsage_dynamic.onnx
  Converting to CPU IR: graphsage_cpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  Converting to GPU IR: graphsage_gpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html

--- GCN ---
  Exporting to ONNX: gcn_dynamic.onnx
C:\Env\Anaconda\envs\MIX\lib\site-packages\torch\onnx\symbolic_opset9.py:5804: UserWarning: Exporting aten::index operator with indices of type Byte. Only 1-D indices are supported. In any other case, this will produce an incorrect ONNX graph.
  warnings.warn(
  Converting to CPU IR: gcn_cpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  Converting to GPU IR: gcn_gpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html

--- GAT ---
  Exporting to ONNX: gat_dynamic.onnx
  Converting to CPU IR: gat_cpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
  Converting to GPU IR: gat_gpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html

CPU/GPU models exported: 6 files

[Step 2/3] Measuring CPU and GPU latencies (42 test cases x 3 models x 2 devices)...
======================================================================
Baseline Profiling - Complete 1-Layer GNN Models
======================================================================
Models: graphsage, gcn, gat
Test cases: 42
Feature dim: 500, Output dim: 256


======================================================================
Measuring CPU/GPU Latencies (Complete 1-Layer GNNs)
======================================================================

--- GRAPHSAGE ---

  [CPU]
    [5000n, 50000e]... 254.23ms
    [5000n, 125000e]... 733.75ms
    [5000n, 200000e]... 643.64ms
    [5000n, 250000e]... 295.92ms
    [5000n, 300000e]... 386.69ms
    [5000n, 375000e]... 497.68ms
    [5000n, 500000e]... 564.08ms
    [10000n, 100000e]... 138.52ms
    [10000n, 250000e]... 670.26ms
    [10000n, 400000e]... 1837.73ms
    [10000n, 500000e]... 714.09ms
    [10000n, 600000e]... 809.51ms
    [10000n, 750000e]... 1116.84ms
    [10000n, 1000000e]... 2124.05ms
    [20000n, 200000e]... 398.77ms
    [20000n, 500000e]... 837.12ms
    [20000n, 800000e]... 3614.76ms
    [20000n, 1000000e]... 1440.88ms
    [20000n, 1200000e]... 4165.99ms
    [20000n, 1500000e]... 2306.84ms
    [20000n, 2000000e]... 2808.21ms
    [50000n, 500000e]... 1590.76ms
    [50000n, 1250000e]... 2083.62ms
    [50000n, 2000000e]... 4928.43ms
    [50000n, 2500000e]... 4967.45ms
    [50000n, 3000000e]... 7620.53ms
    [50000n, 3750000e]... 10119.13ms
    [50000n, 5000000e]... 9728.69ms
    [80000n, 800000e]... 3953.00ms
    [80000n, 2000000e]... 4999.34ms
    [80000n, 3200000e]... 6852.70ms
    [80000n, 4000000e]... 7733.35ms
    [80000n, 4800000e]... 7901.80ms
    [80000n, 6000000e]... 12784.72ms
    [80000n, 8000000e]... 118326.66ms
    [100000n, 1000000e]... 4062.51ms
    [100000n, 2500000e]... 5510.61ms
    [100000n, 4000000e]... 8065.10ms
    [100000n, 5000000e]... 11335.67ms
    [100000n, 6000000e]... 13230.78ms
    [100000n, 7500000e]... 77957.09ms
    [100000n, 10000000e]... 263834.91ms

  [GPU]
    [5000n, 50000e]... 15.88ms
    [5000n, 125000e]... 28.15ms
    [5000n, 200000e]... 43.45ms
    [5000n, 250000e]... 56.00ms
    [5000n, 300000e]... 65.19ms
    [5000n, 375000e]... 82.12ms
    [5000n, 500000e]... 106.02ms
    [10000n, 100000e]... 27.55ms
    [10000n, 250000e]... 57.87ms
    [10000n, 400000e]... 92.45ms
    [10000n, 500000e]... 113.75ms
    [10000n, 600000e]... 134.74ms
    [10000n, 750000e]... 164.59ms
    [10000n, 1000000e]... 213.98ms
    [20000n, 200000e]... 57.20ms
    [20000n, 500000e]... 122.28ms
    [20000n, 800000e]... 183.63ms
    [20000n, 1000000e]... 224.87ms
    [20000n, 1200000e]... 267.49ms
    [20000n, 1500000e]... 319.07ms
    [20000n, 2000000e]... 419.95ms
    [50000n, 500000e]... 148.51ms
    [50000n, 1250000e]... 303.04ms
    [50000n, 2000000e]... 464.38ms
    [50000n, 2500000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [50000n, 3000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [50000n, 3750000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [50000n, 5000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [80000n, 800000e]... 235.67ms
    [80000n, 2000000e]... 488.64ms
    [80000n, 3200000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [80000n, 4000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [80000n, 4800000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [80000n, 6000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [80000n, 8000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 1000000e]... 291.76ms
    [100000n, 2500000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 4000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 5000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 6000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 7500000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 10000000e]... FAILED: Exception from src\inference\src\cpp\infer_request

--- GCN ---

  [CPU]
    [5000n, 50000e]... 48.20ms
    [5000n, 125000e]... 91.35ms
    [5000n, 200000e]... 324.94ms
    [5000n, 250000e]... 412.57ms
    [5000n, 300000e]... 471.39ms
    [5000n, 375000e]... 638.12ms
    [5000n, 500000e]... 589.01ms
    [10000n, 100000e]... 86.25ms
    [10000n, 250000e]... 195.36ms
    [10000n, 400000e]... 317.03ms
    [10000n, 500000e]... 438.35ms
    [10000n, 600000e]... 518.22ms
    [10000n, 750000e]... 695.83ms
    [10000n, 1000000e]... 1400.85ms
    [20000n, 200000e]... 355.18ms
    [20000n, 500000e]... 443.84ms
    [20000n, 800000e]... 783.04ms
    [20000n, 1000000e]... 912.19ms
    [20000n, 1200000e]... 2170.32ms
    [20000n, 1500000e]... 1340.39ms
    [20000n, 2000000e]... 2424.99ms
    [50000n, 500000e]... 978.00ms
    [50000n, 1250000e]... 1482.26ms
    [50000n, 2000000e]... 3818.99ms
    [50000n, 2500000e]... 3517.35ms
    [50000n, 3000000e]... 3137.12ms
    [50000n, 3750000e]... 7073.02ms
    [50000n, 5000000e]... 8812.12ms
    [80000n, 800000e]... 1045.59ms
    [80000n, 2000000e]... 4747.85ms
    [80000n, 3200000e]... 6065.91ms
    [80000n, 4000000e]... 5509.72ms
    [80000n, 4800000e]... 6819.50ms
    [80000n, 6000000e]... 8930.51ms
    [80000n, 8000000e]... 10121.60ms
    [100000n, 1000000e]... 1849.10ms
    [100000n, 2500000e]... 4516.25ms
    [100000n, 4000000e]... 6476.28ms
    [100000n, 5000000e]... 7207.72ms
    [100000n, 6000000e]... 9145.54ms
    [100000n, 7500000e]... 10873.01ms
    [100000n, 10000000e]... FAILED: Exception from src\inference\src\cpp\infer_request

  [GPU]
    [5000n, 50000e]... 14.12ms
    [5000n, 125000e]... 31.15ms
    [5000n, 200000e]... 46.09ms
    [5000n, 250000e]... 57.14ms
    [5000n, 300000e]... 68.87ms
    [5000n, 375000e]... 82.20ms
    [5000n, 500000e]... 107.21ms
    [10000n, 100000e]... 31.25ms
    [10000n, 250000e]... 61.70ms
    [10000n, 400000e]... 92.32ms
    [10000n, 500000e]... 113.29ms
    [10000n, 600000e]... 134.51ms
    [10000n, 750000e]... 164.77ms
    [10000n, 1000000e]... 218.61ms
    [20000n, 200000e]... 59.30ms
    [20000n, 500000e]... 127.15ms
    [20000n, 800000e]... 188.21ms
    [20000n, 1000000e]... 229.38ms
    [20000n, 1200000e]... 278.43ms
    [20000n, 1500000e]... 344.29ms
    [20000n, 2000000e]... 452.23ms
    [50000n, 500000e]... 148.38ms
    [50000n, 1250000e]... 306.86ms
    [50000n, 2000000e]... 465.46ms
    [50000n, 2500000e]... 587.23ms
    [50000n, 3000000e]... 679.09ms
    [50000n, 3750000e]... 835.54ms
    [50000n, 5000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [80000n, 800000e]... 240.46ms
    [80000n, 2000000e]... 497.31ms
    [80000n, 3200000e]... 749.48ms
    [80000n, 4000000e]... 928.71ms
    [80000n, 4800000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [80000n, 6000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [80000n, 8000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 1000000e]... 308.36ms
    [100000n, 2500000e]... 615.43ms
    [100000n, 4000000e]... 928.59ms
    [100000n, 5000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 6000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 7500000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 10000000e]... FAILED: Exception from src\inference\src\cpp\infer_request

--- GAT ---

  [CPU]
    [5000n, 50000e]... 47.61ms
    [5000n, 125000e]... 102.99ms
    [5000n, 200000e]... 140.08ms
    [5000n, 250000e]... 681.27ms
    [5000n, 300000e]... 710.34ms
    [5000n, 375000e]... 622.26ms
    [5000n, 500000e]... 792.46ms
    [10000n, 100000e]... 221.82ms
    [10000n, 250000e]... 461.11ms
    [10000n, 400000e]... 315.46ms
    [10000n, 500000e]... 416.45ms
    [10000n, 600000e]... 534.28ms
    [10000n, 750000e]... 680.73ms
    [10000n, 1000000e]... 1597.60ms
    [20000n, 200000e]... 485.63ms
    [20000n, 500000e]... 720.51ms
    [20000n, 800000e]... 866.98ms
    [20000n, 1000000e]... 960.92ms
    [20000n, 1200000e]... 2767.90ms
    [20000n, 1500000e]... 1436.13ms
    [20000n, 2000000e]... 3467.71ms
    [50000n, 500000e]... 751.78ms
    [50000n, 1250000e]... 3843.00ms
    [50000n, 2000000e]... 4549.61ms
    [50000n, 2500000e]... 3105.50ms
    [50000n, 3000000e]... 2909.78ms
    [50000n, 3750000e]... 5250.57ms
    [50000n, 5000000e]... 7121.07ms
    [80000n, 800000e]... 2527.99ms
    [80000n, 2000000e]... 2309.14ms
    [80000n, 3200000e]... 5680.50ms
    [80000n, 4000000e]... 6213.32ms
    [80000n, 4800000e]... 6273.21ms
    [80000n, 6000000e]... 10416.88ms
    [80000n, 8000000e]... 12295.76ms
    [100000n, 1000000e]... 1566.72ms
    [100000n, 2500000e]... 4130.85ms
    [100000n, 4000000e]... 6202.46ms
    [100000n, 5000000e]... 7095.93ms
    [100000n, 6000000e]... 10892.36ms
    [100000n, 7500000e]... 11974.14ms
    [100000n, 10000000e]... FAILED: Exception from src\inference\src\cpp\infer_request

  [GPU]
    [5000n, 50000e]... 16.89ms
    [5000n, 125000e]... 33.81ms
    [5000n, 200000e]... 49.83ms
    [5000n, 250000e]... 62.54ms
    [5000n, 300000e]... 74.86ms
    [5000n, 375000e]... 90.53ms
    [5000n, 500000e]... 120.59ms
    [10000n, 100000e]... 33.52ms
    [10000n, 250000e]... 64.65ms
    [10000n, 400000e]... 99.90ms
    [10000n, 500000e]... 123.80ms
    [10000n, 600000e]... 149.39ms
    [10000n, 750000e]... 185.00ms
    [10000n, 1000000e]... 248.64ms
    [20000n, 200000e]... 63.45ms
    [20000n, 500000e]... 141.81ms
    [20000n, 800000e]... 211.72ms
    [20000n, 1000000e]... 262.19ms
    [20000n, 1200000e]... 326.88ms
    [20000n, 1500000e]... 400.69ms
    [20000n, 2000000e]... 529.97ms
    [50000n, 500000e]... 168.00ms
    [50000n, 1250000e]... 352.28ms
    [50000n, 2000000e]... 538.43ms
    [50000n, 2500000e]... 658.22ms
    [50000n, 3000000e]... 781.50ms
    [50000n, 3750000e]... 967.79ms
    [50000n, 5000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [80000n, 800000e]... 272.19ms
    [80000n, 2000000e]... 574.50ms
    [80000n, 3200000e]... 864.07ms
    [80000n, 4000000e]... 1073.95ms
    [80000n, 4800000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [80000n, 6000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [80000n, 8000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 1000000e]... 348.70ms
    [100000n, 2500000e]... 709.80ms
    [100000n, 4000000e]... 1080.72ms
    [100000n, 5000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 6000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 7500000e]... FAILED: Exception from src\inference\src\cpp\infer_request
    [100000n, 10000000e]... FAILED: Exception from src\inference\src\cpp\infer_request
Saved: C:\Private\Research\GNX_final\GNX\experiments\baseline_profiling\results\cpugpu_results.json

CPU/GPU results saved to: results\cpugpu_results.json

[Step 3/3] Generating analysis...
======================================================================
Baseline Profiling - Complete 1-Layer GNN Models
======================================================================
Models: graphsage, gcn, gat
Test cases: 42
Feature dim: 500, Output dim: 256


CSV saved: C:\Private\Research\GNX_final\GNX\experiments\baseline_profiling\results\baseline_latency.csv

======================================================================
Baseline Profiling Summary (Complete 1-Layer GNNs)
======================================================================

Models tested: GraphSAGE, GCN, GAT (1 layer each)

--- GRAPHSAGE ---
  CPU: mean=14617.77ms, min=138.52ms, max=263834.91ms (42 tests)
  GPU: mean=175.12ms, min=15.88ms, max=488.64ms (27 tests)

--- GCN ---
  CPU: mean=3092.31ms, min=48.20ms, max=10873.01ms (41 tests)
  GPU: mean=299.80ms, min=14.12ms, max=928.71ms (34 tests)

--- GAT ---
  CPU: mean=3247.34ms, min=47.61ms, max=12295.76ms (41 tests)
  GPU: mean=343.55ms, min=16.89ms, max=1080.72ms (34 tests)

========================================================================
CPU + GPU Testing Complete
========================================================================

Results:
  - CPU/GPU: results\cpugpu_results.json
  - Summary CSV: results\baseline_latency.csv

Press any key to continue . . .

