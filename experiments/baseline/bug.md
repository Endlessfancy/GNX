(MIX) PS C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only> python .\cpu_batch_4096_baseline.py --device NPU
================================================================================
NPU Batch-4096 Baseline Test (1 Layer, 1-hop) - OpenVINO
  (Using HETERO:NPU,CPU for automatic fallback)
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
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch0_n64759_e618876.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch1_n47966_e426540.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch2_n39375_e332844.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch3_n33931_e277624.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch4_n31768_e255618.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch5_n28781_e228462.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch6_n27423_e214524.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch7_n26274_e204478.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch8_n24784_e191408.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch9_n24165_e185456.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch10_n22908_e174022.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch11_n22685_e172422.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch12_n21551_e161304.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch13_n21062_e157548.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch14_n20117_e150008.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch15_n20003_e150044.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch16_n19575_e145752.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch17_n19061_e142904.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch18_n18555_e137708.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch19_n18048_e134254.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch20_n17561_e130658.onnx
  ONNX model exists (static shape): C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\models\graphsage_npu_batch21_n13769_e96832.onnx
  All 22 ONNX models ready (will compile lazily during inference)

Partition Summary:
  Total batches: 22
  Total target nodes: 89,250
  Total halo nodes (with duplicates): 494,871
  Total subgraph nodes (with duplicates): 584,121
  Expansion ratio: 6.54x

Compiling and warming up 22 models (compile → warmup each)...
  Batch 1/22: compiling...   Using HETERO mode: HETERO:NPU,CPU
  OpenVINO model compiled for HETERO:NPU,CPU
done, warming up... Traceback (most recent call last):
  File "C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\cpu_batch_4096_baseline.py", line 608, in <module>
    results = run_batch_baseline(
  File "C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\cpu_batch_4096_baseline.py", line 432, in run_batch_baseline
    _ = run_openvino_inference(compiled_models_cache[i], x_np, edge_index_np)
  File "C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only\cpu_batch_4096_baseline.py", line 315, in run_openvino_inference
    infer_request.infer()
  File "C:\Env\Anaconda\envs\MIX\lib\site-packages\openvino\runtime\ie_api.py", line 132, in infer
    return OVDict(super().infer(_data_dispatch(
RuntimeError: Exception from src\inference\src\cpp\infer_request.cpp:223:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:93:
L0 pfnAppendGraphExecute result: ZE_RESULT_ERROR_INVALID_NATIVE_BINARY, code 0x7800000f - native binary is not supported by the device . elf_parsing_t exception caught: BufferInfo already exists at requested index

(MIX) PS C:\Private\Research\GNX_final\GNX\experiments\baseline\CPU_only>