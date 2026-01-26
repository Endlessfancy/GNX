(MIX) PS C:\Private\Research\GNX_final\GNX\profiling> python .\gat_profile_npu.py --node 5000 --stage 1
============================================================
GAT NPU Test: Stage 1, Nodes=5000
NPU Skip Stages: 4, 6 (scatter operations)
Edges to test: [25000, 50000, 100000, 200000, 300000, 400000, 600000]
============================================================
  [5000n, 25000e] Testing... 4.24ms +/- 0.83
  [5000n, 50000e] Testing... 4.23ms +/- 0.31
  [5000n, 100000e] Testing... 3.92ms +/- 0.34
  [5000n, 200000e] Testing... 4.20ms +/- 0.77
  [5000n, 300000e] Testing... 4.44ms +/- 1.00
  [5000n, 400000e] Testing... 3.98ms +/- 0.42
  [5000n, 600000e] Testing... 3.76ms +/- 0.32

============================================================
Results: 7 passed, 0 failed
Saved to: C:\Private\Research\GNX_final\GNX\profiling\results\gat\npu_stage1_n5000.json
============================================================
(MIX) PS C:\Private\Research\GNX_final\GNX\profiling>


