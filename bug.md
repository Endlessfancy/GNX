======================================================================
Execution Mode: Sequential
======================================================================


======================================================================
Cluster 0: custom_pep_0
  PEP: [[['CPU', 'GPU'], [1, 2, 3, 4, 5], [0.5, 0.5]], [['NPU'], [6, 7]]]
  Subgraphs: [0, 1, 2, 3, 4, 5, 6, 7]
======================================================================

  Subgraph 0... 1412.57ms
  Subgraph 1... 1081.60ms
  Subgraph 2... 741.07ms
  Subgraph 3... 708.59ms
  Subgraph 4... 638.58ms
  Subgraph 5... 667.03ms
  Subgraph 6... 533.52ms
  Subgraph 7... 512.03ms

✓ Cluster 0 completed in 17144.59ms


======================================================================
Cluster 1: custom_pep_1
  PEP: [[['CPU'], [1, 2, 3, 4]], [['GPU', 'NPU'], [5, 6, 7], [0.7, 0.3]]]
  Subgraphs: [8, 9, 10, 11, 12, 13, 14, 15]
======================================================================

  Subgraph 8... 387.03ms
  Subgraph 9... 343.51ms
  Subgraph 10... 385.03ms
  Subgraph 11... 382.03ms
  Subgraph 12... 419.01ms
  Subgraph 13... 247.59ms
  Subgraph 14... 328.36ms
  Subgraph 15... 477.54ms

✓ Cluster 1 completed in 11620.16ms


✓ All clusters executed
  Total time: 29643.32ms
  ✓ Sequential execution complete

[5/5] Executing pipeline (Pipeline Parallel mode)...

======================================================================
Execution Mode: Pipeline Parallel
======================================================================


======================================================================
Cluster 0: custom_pep_0 (Pipeline Mode)
  PEP: [[['CPU', 'GPU'], [1, 2, 3, 4, 5], [0.5, 0.5]], [['NPU'], [6, 7]]]
  Subgraphs: [0, 1, 2, 3, 4, 5, 6, 7]
  Blocks: 2
======================================================================

  Starting pipeline execution with 2 blocks...
  [Block0 Thread-1760] Starting SG0 at 1767150902.948s
  [Block1 Thread-2024] Starting SG0 at 1767150902.954s
  [Block0 Thread-1760] Finished SG0: wait=0ms, exec=1127ms, total=3018ms
  [Block0 Thread-1760] Starting SG1 at 1767150905.968s
  [Block1 Thread-2024] Finished SG0: wait=3012ms, exec=234ms, total=3245ms
  Subgraph 0 completed: 3245.14ms
  [Block1 Thread-2024] Starting SG1 at 1767150906.340s
  [Block0 Thread-1760] Finished SG1: wait=0ms, exec=805ms, total=2632ms
  [Block0 Thread-1760] Starting SG2 at 1767150908.659s
  [Block1 Thread-2024] Finished SG1: wait=2185ms, exec=282ms, total=2543ms
  Subgraph 1 completed: 2542.69ms
  [Block1 Thread-2024] Starting SG2 at 1767150909.159s
  [Block0 Thread-1760] Finished SG2: wait=0ms, exec=658ms, total=2124ms
  [Block0 Thread-1760] Starting SG3 at 1767150910.912s
  [Block1 Thread-2024] Finished SG2: wait=1595ms, exec=198ms, total=1822ms
  Subgraph 2 completed: 1822.27ms
  [Block1 Thread-2024] Starting SG3 at 1767150911.251s
  [Block0 Thread-1760] Finished SG3: wait=0ms, exec=559ms, total=1747ms
  [Block1 Thread-2024] Finished SG3: wait=1342ms, exec=62ms, total=1470ms
  [Block0 Thread-1760] Starting SG4 at 1767150912.770s
  Subgraph 3 completed: 1470.15ms
  [Block1 Thread-2024] Starting SG4 at 1767150913.183s
  [Block0 Thread-1760] Finished SG4: wait=0ms, exec=623ms, total=1879ms
  [Block1 Thread-2024] Finished SG4: wait=1353ms, exec=124ms, total=1592ms
  [Block0 Thread-1760] Starting SG5 at 1767150914.777s
  Subgraph 4 completed: 1591.54ms
  [Block1 Thread-2024] Starting SG5 at 1767150915.040s
  [Block0 Thread-1760] Finished SG5: wait=0ms, exec=590ms, total=1806ms
  [Block0 Thread-1760] Starting SG6 at 1767150916.714s
  [Block1 Thread-2024] Finished SG5: wait=1413ms, exec=200ms, total=1743ms
  Subgraph 5 completed: 1743.26ms
  [Block1 Thread-2024] Starting SG6 at 1767150917.069s
  [Block0 Thread-1760] Finished SG6: wait=0ms, exec=210ms, total=1475ms
  [Block0 Thread-1760] Starting SG7 at 1767150918.190s
  [Block1 Thread-2024] Finished SG6: wait=1023ms, exec=83ms, total=1203ms
  Subgraph 6 completed: 1202.82ms
  [Block1 Thread-2024] Starting SG7 at 1767150918.434s
  [Block0 Thread-1760] Finished SG7: wait=0ms, exec=529ms, total=1868ms
  [Block1 Thread-2024] Finished SG7: wait=1559ms, exec=196ms, total=1822ms
  Subgraph 7 completed: 1822.33ms

✓ Cluster 0 completed in 17657.64ms


======================================================================
Cluster 1: custom_pep_1 (Pipeline Mode)
  PEP: [[['CPU'], [1, 2, 3, 4]], [['GPU', 'NPU'], [5, 6, 7], [0.7, 0.3]]]
  Subgraphs: [8, 9, 10, 11, 12, 13, 14, 15]
  Blocks: 2
======================================================================

  Starting pipeline execution with 2 blocks...
  [Block0 Thread-2056] Starting SG8 at 1767150921.015s
  [Block1 Thread-8944] Starting SG8 at 1767150921.016s
  [Block0 Thread-2056] Finished SG8: wait=0ms, exec=136ms, total=600ms
  [Block0 Thread-2056] Starting SG9 at 1767150921.621s
  [Block1 Thread-8944] Finished SG8: wait=599ms, exec=612ms, total=1211ms
  Subgraph 8 completed: 1210.87ms
  [Block1 Thread-8944] Starting SG9 at 1767150922.459s
  [Block0 Thread-2056] Finished SG9: wait=0ms, exec=153ms, total=1965ms
  [Block0 Thread-2056] Starting SG10 at 1767150923.590s
  [Block1 Thread-8944] Finished SG9: wait=1022ms, exec=585ms, total=1712ms
  Subgraph 9 completed: 1711.93ms
  [Block1 Thread-8944] Starting SG10 at 1767150924.409s
  [Block0 Thread-2056] Finished SG10: wait=0ms, exec=174ms, total=1963ms
  [Block0 Thread-2056] Starting SG11 at 1767150925.554s
  [Block1 Thread-8944] Finished SG10: wait=1036ms, exec=493ms, total=1637ms
  Subgraph 10 completed: 1636.80ms
  [Block1 Thread-8944] Starting SG11 at 1767150926.274s
  [Block0 Thread-2056] Finished SG11: wait=0ms, exec=113ms, total=1752ms
  [Block0 Thread-2056] Starting SG12 at 1767150927.308s
  [Block1 Thread-8944] Finished SG11: wait=908ms, exec=436ms, total=1469ms
  Subgraph 11 completed: 1468.78ms
  [Block1 Thread-8944] Starting SG12 at 1767150927.989s
  [Block0 Thread-2056] Finished SG12: wait=0ms, exec=145ms, total=1879ms
  [Block0 Thread-2056] Starting SG13 at 1767150929.188s
  [Block1 Thread-8944] Finished SG12: wait=1089ms, exec=772ms, total=1970ms
  Subgraph 12 completed: 1970.05ms
  [Block1 Thread-8944] Starting SG13 at 1767150930.075s
  [Block0 Thread-2056] Finished SG13: wait=0ms, exec=271ms, total=1327ms
  [Block0 Thread-2056] Starting SG14 at 1767150930.614s
  [Block1 Thread-8944] Finished SG13: wait=391ms, exec=528ms, total=967ms
  Subgraph 13 completed: 967.07ms
  [Block1 Thread-8944] Starting SG14 at 1767150931.165s
  [Block0 Thread-2056] Finished SG14: wait=0ms, exec=144ms, total=1219ms
  [Block0 Thread-2056] Starting SG15 at 1767150931.838s
  [Block1 Thread-8944] Finished SG14: wait=618ms, exec=568ms, total=1236ms
  Subgraph 14 completed: 1236.37ms
  [Block1 Thread-8944] Starting SG15 at 1767150932.639s
  [Block0 Thread-2056] Finished SG15: wait=0ms, exec=100ms, total=1689ms
  [Block1 Thread-8944] Finished SG15: wait=770ms, exec=449ms, total=1338ms
  Subgraph 15 completed: 1337.82ms

✓ Cluster 1 completed in 13357.73ms


✓ All clusters executed
  Total time: 31699.00ms
  ✓ Pipeline parallel execution complete

================================================================================
Execution Results Comparison
================================================================================

Sequential Execution:
  Total time: 29643.32ms
  Output shape: torch.Size([89250, 256])
  Per-cluster times: ['17144.59ms', '11620.16ms']

Pipeline Parallel Execution:
  Total time: 31699.00ms
  Output shape: torch.Size([89250, 256])
  Per-cluster times: ['17657.64ms', '13357.73ms']

Performance Comparison:
  Speedup: 0.94x
  Time saved: -2055.68ms (-6.9%)

✓ Results verification: PASSED (outputs match)


================================================================================
Detailed Timing Statistics (by Subgraph and Block)
================================================================================


Block 0 (Thread-2056):
Subgraph     Wait Time    Exec Time    Total Time
--------------------------------------------------
SG8                   0ms        136ms        600ms
SG9                   0ms        153ms       1965ms
SG10                  0ms        174ms       1963ms
SG11                  0ms        113ms       1752ms
SG12                  0ms        145ms       1879ms
SG13                  0ms        271ms       1327ms
SG14                  0ms        144ms       1219ms
SG15                  0ms        100ms       1689ms
--------------------------------------------------
Total                 0ms       1235ms      12395ms

Block 1 (Thread-8944):
Subgraph     Wait Time    Exec Time    Total Time
--------------------------------------------------
SG8                 599ms        612ms       1211ms
SG9                1022ms        585ms       1712ms
SG10               1036ms        493ms       1637ms
SG11                908ms        436ms       1469ms
SG12               1089ms        772ms       1970ms
SG13                391ms        528ms        967ms
SG14                618ms        568ms       1236ms
SG15                770ms        449ms       1338ms
--------------------------------------------------
Total              6433ms       4442ms      11540ms

================================================================================
Pipeline Performance Analysis
================================================================================

Theoretical Performance:
  Sum of all block execution times (sequential): 5677ms
  Actual pipeline wallclock time:               13339ms

Speedup Analysis:
  Theoretical maximum speedup (ideal):          2.00x
  Actual speedup achieved:                      0.43x
  Pipeline efficiency:                          21.3%

Block Execution Times (excluding wait):
  Block 0: total_exec=1235ms, total_wait=0ms, avg_exec=154ms
  Block 1: total_exec=4442ms, total_wait=6433ms, avg_exec=555ms

================================================================================
Generating Gantt Chart...
================================================================================
================================================================================
Pipeline Execution Gantt Chart
================================================================================

Total duration: 12962ms
Time scale: ~185.2ms per character

Block 0 (Thread-2056): 8889999999999AAAAAAAAAA BBBBBBBBBCCCCCCCCCC DDDDDDDEEEEEE FFFFFFFFF
Block 1 (Thread-8944): 888888 999999999  AAAAAAAA  BBBBBBB  CCCCCCCCCC DDDDD EEEEEE  FFFFFFF

Time axis:   0.0s 1.0s 2.0s  3.0s 4.0s  5.0s 6.0s 7.0s  8.0s 9.0s  10.0s11.0s12.0s 13.0s

Legend:
  Each character represents a time slice of the pipeline execution
  Numbers/letters indicate which subgraph is being processed
  Gaps indicate waiting time or idle periods
  Overlapping execution shows pipeline parallelism

Pipeline Overlap Analysis:
  Overlapping execution detected: 15 periods
  Total overlap time: 10973ms (84.7% of total)
  This demonstrates true pipeline parallelism!

================================================================================

✓ Gantt chart saved to: pipeline_gantt.txt

✓ Pipeline execution test completed successfully!