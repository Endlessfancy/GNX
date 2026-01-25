========================================================================
PEP3 CPU/GPU Profiling - Fused Block 0 (Stages 1-4)
========================================================================

Activating MIX environment...

Fused Block 0: GATHER + MESSAGE + REDUCE_SUM + REDUCE_COUNT

Test Configuration:
  Node sizes: 5k, 10k, 20k, 50k, 80k, 100k (6 levels)
  Edge ratios: 10, 25, 40, 50, 60, 75, 100 (7 levels)
  Total test cases: 42 x 2 devices = 84 measurements

CPU/GPU fused models not found Exporting...
C:\Users\29067\miniconda3\envs\MIX\lib\site-packages\openvino\runtime\__init__.py:10: DeprecationWarning: The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release. Please replace `openvino.runtime` with `openvino`.
  warnings.warn(
======================================================================
PEP3 Profiling - Fused Block Testing
======================================================================
Test cases: 42
Feature dim: 500

PEP3 Configuration:
  Block 0: CPU + GPU (30%:70% DP) → Fused Stages 1-4
  Block 1: NPU (100%) → Fused Stages 5-7

======================================================================
Exporting CPU/GPU Dynamic Models (FusedBlock0: Stages 1-4)
======================================================================
Exporting FusedBlock0 to ONNX: block0_fused_dynamic.onnx
C:\Users\29067\miniconda3\envs\MIX\lib\site-packages\torch\onnx\symbolic_opset9.py:6075: UserWarning: Warning: ONNX export does not support duplicated values in 'index' field, this will cause the ONNX model to be incorrect.
  warnings.warn(
  Converting to CPU IR: block0_fused_cpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
C:\Users\29067\miniconda3\envs\MIX\lib\site-packages\openvino\runtime\__init__.py:10: DeprecationWarning: The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release. Please replace `openvino.runtime` with `openvino`.
  warnings.warn(
  Converting to GPU IR: block0_fused_gpu.xml
[ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release.
In 2025.0 MO command line tool and openvino.tools.mo.convert_model() will be removed. Please use OpenVINO Model Converter (OVC) or openvino.convert_model(). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
CPU/GPU FusedBlock0 models exported (2 files)

Starting CPU/GPU measurement...
Estimated time: ~1-2 hours

C:\Users\29067\miniconda3\envs\MIX\lib\site-packages\openvino\runtime\__init__.py:10: DeprecationWarning: The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release. Please replace `openvino.runtime` with `openvino`.
  warnings.warn(
======================================================================
PEP3 Profiling - Fused Block Testing
======================================================================
Test cases: 42
Feature dim: 500

PEP3 Configuration:
  Block 0: CPU + GPU (30%:70% DP) → Fused Stages 1-4
  Block 1: NPU (100%) → Fused Stages 5-7


======================================================================
Measuring Block 0: CPU + GPU (FusedBlock0: Stages 1-4)
======================================================================

--- CPU ---
  [5000n, 50000e]... 32.15ms
  [5000n, 118525000e]... 82.88ms
  [5000n, 200000e]... 131.18ms
  [5000n, 250000e]... 164.37ms
  [5000n, 300000e]... 200.42ms
  [5000n, 375000e]... 246.84ms
  [5000n, 500000e]... 312.26ms
  [10000n, 100000e]... 61.90ms
  [10000n, 250000e]... 173.53ms
  [10000n, 400000e]... 262.46ms
  [10000n, 500000e]... 322.06ms
  [10000n, 600000e]... 384.03ms
  [10000n, 750000e]... 479.12ms
  [10000n, 1000000e]... 627.93ms
  [20000n, 200000e]... 138.35ms
  [20000n, 500000e]... 344.92ms
  [20000n, 800000e]... 546.34ms
  [20000n, 1000000e]... 672.94ms
  [20000n, 1200000e]... 813.08ms
  [20000n, 1500000e]... 999.60ms
  [20000n, 2000000e]... 1365.47ms
  [50000n, 500000e]... 414.55ms
  [50000n, 1250000e]... 918.96ms
  [50000n, 2000000e]... 1433.91ms
  [50000n, 2500000e]... 1790.87ms
  [50000n, 3000000e]... 2234.15ms
  [50000n, 3750000e]... 2680.79ms
  [50000n, 5000000e]... 48202.84ms
  [80000n, 800000e]... 656.00ms
  [80000n, 2000000e]... 1345.45ms
  [80000n, 3200000e]... 2303.32ms
  [80000n, 4000000e]... 2962.02ms
  [80000n, 4800000e]... 42994.59ms
  [80000n, 6000000e]... 80277.14ms
  [80000n, 8000000e]... 165839.49ms
  [100000n, 1000000e]... 820.60ms
  [100000n, 2500000e]... 1785.21ms
  [100000n, 4000000e]... 2825.61ms
  [100000n, 5000000e]... 53173.66ms
  [100000n, 6000000e]...






