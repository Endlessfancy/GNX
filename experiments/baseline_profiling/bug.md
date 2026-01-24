============================================================
GPU Async vs Sync Verification
============================================================

Activating MIX environment...
Running async verification test...

==========================================================================================
GPU Latency Verification - Sync vs Async Comparison
==========================================================================================

Comparing inference methods:
  Sync:           compiled_model(inputs) + data access
  Async Full:     set_input + start_async + wait + get_output + data access
  Async Compute:  start_async + wait only (no input/output in timing)

Available devices: ['CPU', 'GPU', 'NPU']


==========================================================================================
Model: GRAPHSAGE
==========================================================================================
  Loading model: graphsage_gpu.xml

  Test Case                 Sync         AsyncFull    AsyncComp    Sync-Full
  ------------------------- ------------ ------------ ------------ ------------
  5k nodes, 50k edges       FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  10k nodes, 100k edges     FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  20k nodes, 200k edges     FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  50k nodes, 500k edges     FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  50k nodes, 2500k edges    FAILED: Exception from src\inference\src\cpp\infer_request.cpp:223:

  80k nodes, 800k edges     FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  100k nodes, 1000k edges   FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:


==========================================================================================
Model: GCN
==========================================================================================
  Loading model: gcn_gpu.xml

  Test Case                 Sync         AsyncFull    AsyncComp    Sync-Full
  ------------------------- ------------ ------------ ------------ ------------
  5k nodes, 50k edges       FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  10k nodes, 100k edges     FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  20k nodes, 200k edges     FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  50k nodes, 500k edges     FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  50k nodes, 2500k edges    FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  80k nodes, 800k edges     FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  100k nodes, 1000k edges   FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:


==========================================================================================
Model: GAT
==========================================================================================
  Loading model: gat_gpu.xml

  Test Case                 Sync         AsyncFull    AsyncComp    Sync-Full
  ------------------------- ------------ ------------ ------------ ------------
  5k nodes, 50k edges       FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  10k nodes, 100k edges     FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  20k nodes, 200k edges     FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  50k nodes, 500k edges     FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  50k nodes, 2500k edges    FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  80k nodes, 800k edges     FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:

  100k nodes, 1000k edges   FAILED: Exception from src\inference\src\cpp\infer_request.cpp:143:


==========================================================================================
Summary
==========================================================================================

Results saved to: C:\Private\Research\GNX_final\GNX\experiments\baseline_profiling\results\gpu_async_verification_results.json

============================================================
Verification complete
============================================================
Results saved to: results\gpu_async_verification_results.json