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
  5k nodes, 50k edges          14.75 ms     16.58 ms     12.17 ms     -11.0%
  10k nodes, 100k edges        30.27 ms     33.04 ms     24.58 ms      -8.4%
  20k nodes, 200k edges        61.80 ms     69.72 ms     52.39 ms     -11.4%
  50k nodes, 500k edges       159.28 ms    175.52 ms    135.11 ms      -9.2%
  50k nodes, 2500k edges    FAILED: Exception from src\inference\src\cpp\infer_request.cpp:223:

  80k nodes, 800k edges       249.10 ms    293.66 ms    222.70 ms     -15.2%
  100k nodes, 1000k edges     318.93 ms    358.46 ms    271.15 ms     -11.0%

==========================================================================================
Model: GCN
==========================================================================================
  Loading model: gcn_gpu.xml

  Test Case                 Sync         AsyncFull    AsyncComp    Sync-Full
  ------------------------- ------------ ------------ ------------ ------------
  5k nodes, 50k edges          22.75 ms     22.19 ms     13.92 ms      +2.5%
  10k nodes, 100k edges        33.98 ms     37.97 ms     28.28 ms     -10.5%
  20k nodes, 200k edges        64.29 ms     71.41 ms     55.44 ms     -10.0%
  50k nodes, 500k edges       167.47 ms    181.70 ms    138.16 ms      -7.8%
  50k nodes, 2500k edges      598.85 ms    620.78 ms    560.18 ms      -3.5%
  80k nodes, 800k edges       255.35 ms    274.91 ms    217.28 ms      -7.1%
  100k nodes, 1000k edges     321.19 ms    356.93 ms    278.89 ms     -10.0%

==========================================================================================
Model: GAT
==========================================================================================
  Loading model: gat_gpu.xml

  Test Case                 Sync         AsyncFull    AsyncComp    Sync-Full
  ------------------------- ------------ ------------ ------------ ------------
  5k nodes, 50k edges          18.01 ms     19.11 ms     15.26 ms      -5.7%
  10k nodes, 100k edges        35.71 ms     39.03 ms     30.02 ms      -8.5%
  20k nodes, 200k edges        68.53 ms     75.32 ms     58.16 ms      -9.0%
  50k nodes, 500k edges       175.70 ms    191.61 ms    153.19 ms      -8.3%
  50k nodes, 2500k edges      665.20 ms    693.40 ms    643.54 ms      -4.1%
  80k nodes, 800k edges       284.96 ms    310.77 ms    254.26 ms      -8.3%
  100k nodes, 1000k edges     364.35 ms    398.44 ms    322.21 ms      -8.6%

==========================================================================================
Summary
==========================================================================================

Sync inference (compiled_model + data access):
  Mean: 195.52 ms
  Range: 14.75 - 665.20 ms

Async inference (full end-to-end with data access):
  Mean: 212.03 ms
  Range: 16.58 - 693.40 ms

Async inference (compute only, no I/O in timing):
  Mean: 174.34 ms
  Range: 12.17 - 643.54 ms

Sync vs Async Full difference: -8.3%
⚠ Sync is 8.3% faster than Async

Results saved to: C:\Private\Research\GNX_final\GNX\experiments\baseline_profiling\results\gpu_async_verification_results.json

============================================================
Verification complete
============================================================
Results saved to: results\gpu_async_verification_results.json
