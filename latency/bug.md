(MIX) PS C:\Private\Research\GNX_final\GNX\latency> git pull origin latency
remote: Enumerating objects: 7, done.
remote: Counting objects: 100% (7/7), done.
remote: Compressing objects: 100% (1/1), done.
remote: Total 4 (delta 3), reused 4 (delta 3), pack-reused 0 (from 0)
Unpacking objects: 100% (4/4), 801 bytes | 29.00 KiB/s, done.
From https://github.com/Endlessfancy/GNX
 * branch            latency    -> FETCH_HEAD
   156ee89..6872f57  latency    -> origin/latency
Updating 156ee89..6872f57
Fast-forward
 latency/test_sequential_latency.py | 4 +++-
 1 file changed, 3 insertions(+), 1 deletion(-)
(MIX) PS C:\Private\Research\GNX_final\GNX\latency> python .\test_sequential_latency.py --pep pep2 --num-iterations 3
======================================================================
Sequential Profiling for Cost Modeling
======================================================================
PEP: pep2
Subgraphs: 8
Iterations: 3
Warmup: 1
Output: cost_model.csv
Available devices: ['CPU', 'GPU', 'NPU']

Loading Flickr dataset...
Loading Flickr dataset...
  Loaded Flickr dataset:
  Nodes: 89,250
  Edges: 899,756
  Features: 500
Partitioned into 8 subgraphs:
  Max nodes per subgraph: 89518
  Max edges per subgraph: 307822
Max subgraph size: 89518 nodes, 307822 edges

======================================================================
Sequential Profiling: PEP2
======================================================================

PEP2:
------------------------------------------------------------
  Block 0: [CPU] Stages [1,2,3,4]
  Block 1: [GPU+NPU] Stages [5,6,7] (GPU:70%, NPU:30%)
------------------------------------------------------------

Exporting models...

============================================================
Exporting models for PEP configuration
============================================================

Block 0: stages [1, 2, 3, 4]

Exporting stages [1, 2, 3, 4] for CPU:
  ONNX exists: stages_1_2_3_4_CPU.onnx (6.4 KB)
  IR exists: stages_1_2_3_4_CPU_dynamic.xml

Block 1: stages [5, 6, 7]

Exporting stages [5, 6, 7] for GPU:
  ONNX exists: stages_5_6_7_GPU.onnx (1002.1 KB)
  IR exists: stages_5_6_7_GPU_dynamic.xml

Exporting stages [5, 6, 7] for NPU:
  ONNX exists: stages_5_6_7_NPU.onnx (1002.1 KB)
  IR exists: stages_5_6_7_NPU_n89518_e307822.xml

============================================================
Exported 3 models
============================================================

Creating stage executors...
  Loading Block0_S1_2_3_4_CPU on CPU (stream 0)...
    Compiled successfully on CPU
  Created executor: Block0_S1_2_3_4_CPU
  Created stage: Block0_S1_2_3_4 with devices ['CPU']
  Loading Block1_S5_6_7_GPU on GPU (stream 0)...
    Compiled successfully on GPU
  Created executor: Block1_S5_6_7_GPU
  Loading Block1_S5_6_7_NPU on NPU (stream 1)...
    Compiled successfully on NPU
    NPU static shape from filename: n=89518, e=307822
  Created executor: Block1_S5_6_7_NPU
  Created stage: Block1_S5_6_7 with devices ['GPU', 'NPU']

Profiling 8 subgraphs x 3 iterations
Warmup: 1 iterations
  Subgraph 0: 81380 nodes, 279839 edges -> total=1540.78ms (S0:841.32ms, S1:150.83ms, S1:528.14ms)
  Subgraph 1: 61287 nodes, 131659 edges -> total=1093.18ms (S0:475.90ms, S1:114.83ms, S1:487.55ms)
  Subgraph 2: 53027 nodes, 105627 edges -> total=933.08ms (S0:334.27ms, S1:96.81ms, S1:489.03ms)
  Subgraph 3: 47876 nodes, 91656 edges -> total=1190.26ms (S0:502.67ms, S1:84.60ms, S1:588.33ms)
  Subgraph 4: 44460 nodes, 82764 edges -> total=864.51ms (S0:251.42ms, S1:88.96ms, S1:512.48ms)
  Subgraph 5: 41181 nodes, 74900 edges -> total=816.49ms (S0:234.13ms, S1:72.45ms, S1:497.38ms)
  Subgraph 6: 39214 nodes, 69793 edges -> total=805.83ms (S0:234.52ms, S1:69.32ms, S1:493.15ms)
  Subgraph 7: 36692 nodes, 63518 edges -> total=766.83ms (S0:199.78ms, S1:66.70ms, S1:490.60ms)

======================================================================
Statistics (PERF_COUNT Profiling Results)
======================================================================

  stage0_CPU (CPU):
    wall    : avg= 388.42ms, std=226.03ms
    device  : avg= 537.11ms, std=236.57ms
    compute : avg= 521.11ms, std=230.57ms

  stage1_GPU (GPU):
    wall    : avg=  94.90ms, std=29.86ms
    device  : avg=  59.23ms, std=11.04ms
    compute : avg=   9.39ms, std= 1.29ms

  stage1_NPU (NPU):
    wall    : avg= 509.98ms, std=29.31ms
    device  : avg= 460.59ms, std= 0.00ms
    compute : avg= 410.00ms, std= 0.00ms

  Total (wall time):
    avg=1006.13ms, std=274.59ms, min=766.83ms, max=1718.96ms

  Throughput: 0.99 subgraphs/sec

======================================================================
Cost table saved to: C:\Private\Research\GNX_final\GNX\latency\results\cost_model.csv
======================================================================

[Cost Table Preview]
 sg_id  iter  num_nodes  num_edges  stage0_CPU_wall_ms  stage0_CPU_device_ms  stage0_CPU_compute_ms  stage1_GPU_wall_ms  stage1_GPU_device_ms  stage1_GPU_compute_ms  stage1_NPU_wall_ms  stage1_NPU_device_ms  stage1_NPU_compute_ms  partition_ms  merge_ms  total_ms
     0     0      81380     279839            841.3211              1150.366               1116.939            150.8314                78.053                 11.885            528.1377               460.592                409.995           0.0   20.3997 1540.7772
     0     1      81380     279839            953.0077              1037.624               1008.286            154.8496                78.851                 11.782            523.5863               460.592                409.995           0.0   23.9214 1655.4736
     0     2      81380     279839            958.3328               978.628                952.916            179.1650                80.109                 11.680            562.0746               460.592                409.995           0.0   19.3006 1718.9621
     1     0      61287     131659            475.8968               767.039                746.212            114.8345                73.576                 10.969            487.5539               460.592                409.995           0.0   14.8134 1093.1818
     1     1      61287     131659            513.4238               716.313                696.846            109.2723                71.211                 10.598            574.2327               460.592                409.995           0.0   14.7448 1211.7499
     1     2      61287     131659            421.9565               665.924                647.113            108.9315                69.650                 10.411            475.1322               460.592                409.995           0.0   15.1836 1021.2928
     2     0      53027     105627            334.2664               580.927                563.867             96.8100                64.659                  9.963            489.0297               460.592                409.995           0.0   12.8891  933.0770
     2     1      53027     105627            389.5499               555.856                539.420             96.5326                62.886                  9.835            500.1403               460.592                409.995           0.0   17.8334 1004.1336
     2     2      53027     105627            412.0701               536.986                520.832             99.7888                62.950                  9.777            495.2429               460.592                409.995           0.0   12.5774 1019.7521
     3     0      47876      91656            502.6732               504.331                489.296             84.5996                60.404                  9.549            588.3268               460.592                409.995           0.0   14.5725 1190.2555
     3     1      47876      91656            295.4312               485.101                470.487             84.1600                59.293                  9.396            488.1680               460.592                409.995           0.0   11.9671  879.8050
     3     2      47876      91656            365.4567               472.570                458.316             98.7920                58.338                  9.277            499.9606               460.592                409.995           0.0   11.1190  975.4065
     4     0      44460      82764            251.4223               439.534                425.883             88.9583                55.639                  9.083            512.4768               460.592                409.995           0.0   11.5806  864.5121
     4     1      44460      82764            281.5574               425.926                412.634             86.3040                54.604                  8.918            516.4184               460.592                409.995           0.0   10.5567  894.9271
     4     2      44460      82764            259.5734               413.438                400.403             86.7806                53.556                  8.774            534.5271               460.592                409.995           0.0    9.8594  890.8228
     5     0      41181      74900            234.1276               390.194                377.730             72.4542                51.823                  8.580            497.3783               460.592                409.995           0.0   12.4504  816.4876
     5     1      41181      74900            250.9601               381.038                368.843             79.8402                51.046                  8.496            509.7959               460.592                409.995           0.0    9.3448  850.0188
     5     2      41181      74900            245.9541               372.285                360.334             77.4693                50.365                  8.448            479.7074               460.592                409.995           0.0    8.4749  811.6873
     6     0      39214      69793            234.5218               356.394                344.774             69.3237                48.954                  8.254            493.1477               460.592                409.995           0.0    8.7620  805.8312
     6     1      39214      69793            232.9439               349.284                337.929             69.5282                48.261                  8.158            495.4722               460.592                409.995           0.0    9.3215  807.3510
     6     2      39214      69793            237.4252               342.872                331.727             70.4774                47.654                  8.054            513.0990               460.592                409.995           0.0   10.5612  831.6335
     7     0      36692      63518            199.7821               328.491                317.617             66.7047                46.925                  7.872            490.6021               460.592                409.995           0.0    9.6631  766.8260
     7     1      36692      63518            206.0351               322.336                311.604             67.4546                46.569                  7.795            491.2690               460.592                409.995           0.0    7.9788  772.8145
     7     2      36692      63518            224.3408               317.280                306.699             63.7765                46.159                  7.738            493.9481               460.592                409.995           0.0    8.2613  790.399