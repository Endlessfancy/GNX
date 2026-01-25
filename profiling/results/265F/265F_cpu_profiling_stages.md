========================================================================
GNN Stage Profiling - CPU Only (All 7 Stages)
========================================================================

Activating MIX environment...

Test Configuration:
  Stages: 1-7 (GATHER, MESSAGE, REDUCE_SUM, REDUCE_COUNT, NORMALIZE, TRANSFORM, ACTIVATE)
  Device: CPU only


Starting CPU measurement...
C:\Users\29067\miniconda3\envs\MIX\lib\site-packages\openvino\runtime\__init__.py:10: DeprecationWarning: The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release. Please replace `openvino.runtime` with `openvino`.
  warnings.warn(
======================================================================
GNX Stage Profiling - Incremental Pipeline
======================================================================
Test cases: 55
Node sizes: [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 80000, 100000, 150000]

Workflow:
  1. python profile_stages.py --export          # Export all models
  2. python profile_stages.py --measure         # Measure CPU/GPU
  3. run_profiling.bat                          # Run NPU tests (isolated)
  4. python profile_stages.py --merge-npu       # Merge NPU results
  5. python profile_stages.py --analyze         # Generate final report


======================================================================
Measuring CPU Latencies Only
======================================================================

======================================================================
=== Measuring Latencies (CPU) ===
======================================================================
[1/385] Stage 1 on CPU - 1000n 2000e... 1.61ms ±0.12
[2/385] Stage 1 on CPU - 1000n 3000e... 2.15ms ±0.16
[3/385] Stage 1 on CPU - 1000n 5000e... 3.01ms ±0.17
[4/385] Stage 1 on CPU - 1000n 7000e... 4.31ms ±0.21
[5/385] Stage 1 on CPU - 1000n 10000e... 6.19ms ±0.25
[6/385] Stage 1 on CPU - 5000n 10000e... 6.29ms ±0.19
[7/385] Stage 1 on CPU - 5000n 15000e... 9.30ms ±0.20
[8/385] Stage 1 on CPU - 5000n 25000e... 15.03ms ±1.02
[9/385] Stage 1 on CPU - 5000n 35000e... 21.30ms ±0.19
[10/385] Stage 1 on CPU - 5000n 50000e... 29.94ms ±0.33
[11/385] Stage 1 on CPU - 10000n 20000e... 12.49ms ±0.24
[12/385] Stage 1 on CPU - 10000n 30000e... 18.52ms ±0.20
[13/385] Stage 1 on CPU - 10000n 50000e... 30.51ms ±0.27
[14/385] Stage 1 on CPU - 10000n 70000e... 42.66ms ±0.35
[15/385] Stage 1 on CPU - 10000n 100000e... 60.77ms ±0.62
[16/385] Stage 1 on CPU - 20000n 40000e... 25.01ms ±0.53
[17/385] Stage 1 on CPU - 20000n 60000e... 37.49ms ±0.49
[18/385] Stage 1 on CPU - 20000n 100000e... 61.81ms ±0.45
[19/385] Stage 1 on CPU - 20000n 140000e... 85.36ms ±0.68
[20/385] Stage 1 on CPU - 20000n 200000e... 121.76ms ±0.80
[21/385] Stage 1 on CPU - 30000n 60000e... 37.33ms ±0.23
[22/385] Stage 1 on CPU - 30000n 90000e... 56.22ms ±0.53
[23/385] Stage 1 on CPU - 30000n 150000e... 92.61ms ±0.86
[24/385] Stage 1 on CPU - 30000n 210000e... 129.30ms ±1.01
[25/385] Stage 1 on CPU - 30000n 300000e... 184.84ms ±1.55
[26/385] Stage 1 on CPU - 40000n 80000e... 50.27ms ±0.52
[27/385] Stage 1 on CPU - 40000n 120000e... 74.82ms ±0.78
[28/385] Stage 1 on CPU - 40000n 200000e... 125.62ms ±1.71
[29/385] Stage 1 on CPU - 40000n 280000e... 173.61ms ±1.59
[30/385] Stage 1 on CPU - 40000n 400000e... 248.69ms ±1.67
[31/385] Stage 1 on CPU - 50000n 100000e... 62.35ms ±0.66
[32/385] Stage 1 on CPU - 50000n 150000e... 92.83ms ±0.83
[33/385] Stage 1 on CPU - 50000n 250000e... 156.71ms ±1.26
[34/385] Stage 1 on CPU - 50000n 350000e... 217.77ms ±1.55
[35/385] Stage 1 on CPU - 50000n 500000e... 320.77ms ±5.72
[36/385] Stage 1 on CPU - 60000n 120000e... 75.24ms ±1.63
[37/385] Stage 1 on CPU - 60000n 180000e... 112.80ms ±1.99
[38/385] Stage 1 on CPU - 60000n 300000e... 187.86ms ±2.97
[39/385] Stage 1 on CPU - 60000n 420000e... 262.35ms ±3.64
[40/385] Stage 1 on CPU - 60000n 600000e... 373.07ms ±4.81
[41/385] Stage 1 on CPU - 80000n 160000e... 101.39ms ±2.58
[42/385] Stage 1 on CPU - 80000n 240000e... 153.13ms ±2.21
[43/385] Stage 1 on CPU - 80000n 400000e... 250.46ms ±3.16
[44/385] Stage 1 on CPU - 80000n 560000e... 350.11ms ±3.13
[45/385] Stage 1 on CPU - 80000n 800000e... 501.08ms ±5.18
[46/385] Stage 1 on CPU - 100000n 200000e... 125.58ms ±2.32
[47/385] Stage 1 on CPU - 100000n 300000e... 188.81ms ±3.36
[48/385] Stage 1 on CPU - 100000n 500000e... 313.38ms ±3.22
[49/385] Stage 1 on CPU - 100000n 700000e... 440.08ms ±4.18
[50/385] Stage 1 on CPU - 100000n 1000000e... 626.69ms ±5.36
[51/385] Stage 1 on CPU - 150000n 300000e... 189.28ms ±2.79
[52/385] Stage 1 on CPU - 150000n 450000e... 282.62ms ±3.30
[53/385] Stage 1 on CPU - 150000n 750000e... 472.06ms ±3.97
[54/385] Stage 1 on CPU - 150000n 1050000e... 661.57ms ±7.78
[55/385] Stage 1 on CPU - 150000n 1500000e... 941.77ms ±6.33
[56/385] Stage 2 on CPU - 1000n 2000e... 1.57ms ±0.07
[57/385] Stage 2 on CPU - 1000n 3000e... 2.18ms ±0.13
[58/385] Stage 2 on CPU - 1000n 5000e... 3.32ms ±0.17
[59/385] Stage 2 on CPU - 1000n 7000e... 4.45ms ±0.14
[60/385] Stage 2 on CPU - 1000n 10000e... 6.04ms ±0.14
[61/385] Stage 2 on CPU - 5000n 10000e... 6.10ms ±0.24
[62/385] Stage 2 on CPU - 5000n 15000e... 9.00ms ±0.11
[63/385] Stage 2 on CPU - 5000n 25000e... 15.13ms ±0.44
[64/385] Stage 2 on CPU - 5000n 35000e... 21.51ms ±0.67
[65/385] Stage 2 on CPU - 5000n 50000e... 30.47ms ±0.92
[66/385] Stage 2 on CPU - 10000n 20000e... 12.13ms ±0.27
[67/385] Stage 2 on CPU - 10000n 30000e... 18.24ms ±0.43
[68/385] Stage 2 on CPU - 10000n 50000e... 30.26ms ±0.60
[69/385] Stage 2 on CPU - 10000n 70000e... 42.22ms ±0.90
[70/385] Stage 2 on CPU - 10000n 100000e... 60.46ms ±1.38
[71/385] Stage 2 on CPU - 20000n 40000e... 24.58ms ±0.87
[72/385] Stage 2 on CPU - 20000n 60000e... 36.07ms ±0.67
[73/385] Stage 2 on CPU - 20000n 100000e... 60.14ms ±1.16
[74/385] Stage 2 on CPU - 20000n 140000e... 83.97ms ±1.65
[75/385] Stage 2 on CPU - 20000n 200000e... 119.67ms ±2.30
[76/385] Stage 2 on CPU - 30000n 60000e... 36.13ms ±0.70
[77/385] Stage 2 on CPU - 30000n 90000e... 54.13ms ±1.04
[78/385] Stage 2 on CPU - 30000n 150000e... 89.19ms ±5.42
[79/385] Stage 2 on CPU - 30000n 210000e... 94.04ms ±23.06
[80/385] Stage 2 on CPU - 30000n 300000e... 167.53ms ±24.80
[81/385] Stage 2 on CPU - 40000n 80000e... 48.47ms ±0.93
[82/385] Stage 2 on CPU - 40000n 120000e... 71.89ms ±1.25
[83/385] Stage 2 on CPU - 40000n 200000e... 120.38ms ±2.17
[84/385] Stage 2 on CPU - 40000n 280000e... 168.46ms ±2.44
[85/385] Stage 2 on CPU - 40000n 400000e... 239.77ms ±3.41
[86/385] Stage 2 on CPU - 50000n 100000e... 60.58ms ±1.53
[87/385] Stage 2 on CPU - 50000n 150000e... 90.30ms ±1.82
[88/385] Stage 2 on CPU - 50000n 250000e... 150.71ms ±2.14
[89/385] Stage 2 on CPU - 50000n 350000e... 210.33ms ±2.60
[90/385] Stage 2 on CPU - 50000n 500000e... 300.26ms ±4.61
[91/385] Stage 2 on CPU - 60000n 120000e... 72.21ms ±1.28
[92/385] Stage 2 on CPU - 60000n 180000e... 107.90ms ±2.10
[93/385] Stage 2 on CPU - 60000n 300000e... 180.31ms ±2.25
[94/385] Stage 2 on CPU - 60000n 420000e... 252.34ms ±2.76
[95/385] Stage 2 on CPU - 60000n 600000e... 360.53ms ±4.62
[96/385] Stage 2 on CPU - 80000n 160000e... 95.96ms ±1.51
[97/385] Stage 2 on CPU - 80000n 240000e... 144.63ms ±1.99
[98/385] Stage 2 on CPU - 80000n 400000e... 240.19ms ±3.46
[99/385] Stage 2 on CPU - 80000n 560000e... 337.14ms ±4.65
[100/385] Stage 2 on CPU - 80000n 800000e... 485.33ms ±7.19
[101/385] Stage 2 on CPU - 100000n 200000e... 122.33ms ±3.26
[102/385] Stage 2 on CPU - 100000n 300000e... 182.09ms ±2.67
[103/385] Stage 2 on CPU - 100000n 500000e... 301.07ms ±3.44
[104/385] Stage 2 on CPU - 100000n 700000e... 421.75ms ±4.05
[105/385] Stage 2 on CPU - 100000n 1000000e... 602.42ms ±4.37
[106/385] Stage 2 on CPU - 150000n 300000e... 180.60ms ±2.68
[107/385] Stage 2 on CPU - 150000n 450000e... 270.19ms ±2.54
[108/385] Stage 2 on CPU - 150000n 750000e... 450.47ms ±3.30
[109/385] Stage 2 on CPU - 150000n 1050000e... 630.17ms ±3.24
[110/385] Stage 2 on CPU - 150000n 1500000e... 901.55ms ±4.93
[111/385] Stage 3 on CPU - 1000n 2000e... 2.16ms ±0.15
[112/385] Stage 3 on CPU - 1000n 3000e... 2.46ms ±0.13
[113/385] Stage 3 on CPU - 1000n 5000e... 3.48ms ±0.20
[114/385] Stage 3 on CPU - 1000n 7000e... 4.68ms ±0.18
[115/385] Stage 3 on CPU - 1000n 10000e... 6.42ms ±0.42
[116/385] Stage 3 on CPU - 5000n 10000e... 10.44ms ±0.30
[117/385] Stage 3 on CPU - 5000n 15000e... 12.98ms ±0.43
[118/385] Stage 3 on CPU - 5000n 25000e... 18.59ms ±0.38
[119/385] Stage 3 on CPU - 5000n 35000e... 23.81ms ±0.97
[120/385] Stage 3 on CPU - 5000n 50000e... 31.36ms ±0.86
[121/385] Stage 3 on CPU - 10000n 20000e... 20.33ms ±1.00
[122/385] Stage 3 on CPU - 10000n 30000e... 25.01ms ±0.85
[123/385] Stage 3 on CPU - 10000n 50000e... 36.45ms ±1.49
[124/385] Stage 3 on CPU - 10000n 70000e... 47.02ms ±1.43
[125/385] Stage 3 on CPU - 10000n 100000e... 63.51ms ±2.16
[126/385] Stage 3 on CPU - 20000n 40000e... 39.38ms ±1.33
[127/385] Stage 3 on CPU - 20000n 60000e... 50.17ms ±1.64
[128/385] Stage 3 on CPU - 20000n 100000e... 72.56ms ±2.59
[129/385] Stage 3 on CPU - 20000n 140000e... 94.28ms ±2.93
[130/385] Stage 3 on CPU - 20000n 200000e... 125.85ms ±2.80
[131/385] Stage 3 on CPU - 30000n 60000e... 60.03ms ±1.97
[132/385] Stage 3 on CPU - 30000n 90000e... 76.47ms ±1.43
[133/385] Stage 3 on CPU - 30000n 150000e... 110.46ms ±3.20
[134/385] Stage 3 on CPU - 30000n 210000e... 143.15ms ±3.09
[135/385] Stage 3 on CPU - 30000n 300000e... 190.47ms ±4.00
[136/385] Stage 3 on CPU - 40000n 80000e... 79.30ms ±2.41
[137/385] Stage 3 on CPU - 40000n 120000e... 102.09ms ±2.92
[138/385] Stage 3 on CPU - 40000n 200000e... 145.13ms ±3.61
[139/385] Stage 3 on CPU - 40000n 280000e... 192.67ms ±4.53
[140/385] Stage 3 on CPU - 40000n 400000e... 254.03ms ±5.46
[141/385] Stage 3 on CPU - 50000n 100000e... 99.90ms ±3.74
[142/385] Stage 3 on CPU - 50000n 150000e... 127.07ms ±2.86
[143/385] Stage 3 on CPU - 50000n 250000e... 181.98ms ±3.75
[144/385] Stage 3 on CPU - 50000n 350000e... 235.92ms ±4.80
[145/385] Stage 3 on CPU - 50000n 500000e... 334.76ms ±8.96
[146/385] Stage 3 on CPU - 60000n 120000e... 120.34ms ±2.96
[147/385] Stage 3 on CPU - 60000n 180000e... 152.27ms ±3.52
[148/385] Stage 3 on CPU - 60000n 300000e... 219.17ms ±6.74
[149/385] Stage 3 on CPU - 60000n 420000e... 287.12ms ±5.16
[150/385] Stage 3 on CPU - 60000n 600000e... 385.03ms ±8.28
[151/385] Stage 3 on CPU - 80000n 160000e... 160.30ms ±4.13
[152/385] Stage 3 on CPU - 80000n 240000e... 203.81ms ±5.43
[153/385] Stage 3 on CPU - 80000n 400000e... 294.17ms ±8.91
[154/385] Stage 3 on CPU - 80000n 560000e... 381.70ms ±9.62
[155/385] Stage 3 on CPU - 80000n 800000e... 520.53ms ±19.57
[156/385] Stage 3 on CPU - 100000n 200000e... 201.88ms ±6.24
[157/385] Stage 3 on CPU - 100000n 300000e... 278.45ms ±29.95
[158/385] Stage 3 on CPU - 100000n 500000e... 374.59ms ±15.02
[159/385] Stage 3 on CPU - 100000n 700000e... 481.73ms ±8.59
[160/385] Stage 3 on CPU - 100000n 1000000e... 656.73ms ±12.58
[161/385] Stage 3 on CPU - 150000n 300000e... 301.61ms ±6.27
[162/385] Stage 3 on CPU - 150000n 450000e... 387.39ms ±9.06
[163/385] Stage 3 on CPU - 150000n 750000e... 555.67ms ±10.90
[164/385] Stage 3 on CPU - 150000n 1050000e... 730.05ms ±16.30
[165/385] Stage 3 on CPU - 150000n 1500000e... 981.76ms ±18.06
[166/385] Stage 4 on CPU - 1000n 2000e... 0.12ms ±0.01
[167/385] Stage 4 on CPU - 1000n 3000e... 0.12ms ±0.01
[168/385] Stage 4 on CPU - 1000n 5000e... 0.18ms ±0.01
[169/385] Stage 4 on CPU - 1000n 7000e... 0.15ms ±0.02
[170/385] Stage 4 on CPU - 1000n 10000e... 0.15ms ±0.01
[171/385] Stage 4 on CPU - 5000n 10000e... 0.20ms ±0.04
[172/385] Stage 4 on CPU - 5000n 15000e... 0.20ms ±0.06
[173/385] Stage 4 on CPU - 5000n 25000e... 0.20ms ±0.01
[174/385] Stage 4 on CPU - 5000n 35000e... 0.24ms ±0.02
[175/385] Stage 4 on CPU - 5000n 50000e... 0.38ms ±0.06
[176/385] Stage 4 on CPU - 10000n 20000e... 0.21ms ±0.04
[177/385] Stage 4 on CPU - 10000n 30000e... 0.22ms ±0.02
[178/385] Stage 4 on CPU - 10000n 50000e... 0.32ms ±0.04
[179/385] Stage 4 on CPU - 10000n 70000e... 0.38ms ±0.03
[180/385] Stage 4 on CPU - 10000n 100000e... 0.48ms ±0.05
[181/385] Stage 4 on CPU - 20000n 40000e... 0.29ms ±0.04
[182/385] Stage 4 on CPU - 20000n 60000e... 0.33ms ±0.02
[183/385] Stage 4 on CPU - 20000n 100000e... 0.50ms ±0.06
[184/385] Stage 4 on CPU - 20000n 140000e... 1.34ms ±0.07
[185/385] Stage 4 on CPU - 20000n 200000e... 1.73ms ±0.09
[186/385] Stage 4 on CPU - 30000n 60000e... 0.37ms ±0.04
[187/385] Stage 4 on CPU - 30000n 90000e... 0.50ms ±0.06
[188/385] Stage 4 on CPU - 30000n 150000e... 1.45ms ±0.08
[189/385] Stage 4 on CPU - 30000n 210000e... 1.81ms ±0.10
[190/385] Stage 4 on CPU - 30000n 300000e... 2.56ms ±0.13
[191/385] Stage 4 on CPU - 40000n 80000e... 0.44ms ±0.05
[192/385] Stage 4 on CPU - 40000n 120000e... 0.60ms ±0.08
[193/385] Stage 4 on CPU - 40000n 200000e... 1.74ms ±0.07
[194/385] Stage 4 on CPU - 40000n 280000e... 2.34ms ±0.09
[195/385] Stage 4 on CPU - 40000n 400000e... 3.35ms ±0.17
[196/385] Stage 4 on CPU - 50000n 100000e... 0.52ms ±0.07
[197/385] Stage 4 on CPU - 50000n 150000e... 1.47ms ±0.09
[198/385] Stage 4 on CPU - 50000n 250000e... 2.14ms ±0.12
[199/385] Stage 4 on CPU - 50000n 350000e... 2.88ms ±0.11
[200/385] Stage 4 on CPU - 50000n 500000e... 4.27ms ±0.12
[201/385] Stage 4 on CPU - 60000n 120000e... 0.63ms ±0.07
[202/385] Stage 4 on CPU - 60000n 180000e... 1.72ms ±0.12
[203/385] Stage 4 on CPU - 60000n 300000e... 2.53ms ±0.10
[204/385] Stage 4 on CPU - 60000n 420000e... 3.58ms ±0.17
[205/385] Stage 4 on CPU - 60000n 600000e... 5.15ms ±0.20
[206/385] Stage 4 on CPU - 80000n 160000e... 1.59ms ±0.08
[207/385] Stage 4 on CPU - 80000n 240000e... 2.12ms ±0.15
[208/385] Stage 4 on CPU - 80000n 400000e... 3.32ms ±0.19
[209/385] Stage 4 on CPU - 80000n 560000e... 4.82ms ±0.17
[210/385] Stage 4 on CPU - 80000n 800000e... 6.90ms ±0.49
[211/385] Stage 4 on CPU - 100000n 200000e... 1.76ms ±0.08
[212/385] Stage 4 on CPU - 100000n 300000e... 2.60ms ±0.12
[213/385] Stage 4 on CPU - 100000n 500000e... 4.35ms ±0.13
[214/385] Stage 4 on CPU - 100000n 700000e... 6.19ms ±0.45
[215/385] Stage 4 on CPU - 100000n 1000000e... 8.58ms ±0.34
[216/385] Stage 4 on CPU - 150000n 300000e... 2.58ms ±0.12
[217/385] Stage 4 on CPU - 150000n 450000e... 4.15ms ±0.40
[218/385] Stage 4 on CPU - 150000n 750000e... 6.50ms ±0.24
[219/385] Stage 4 on CPU - 150000n 1050000e... 8.96ms ±0.54
[220/385] Stage 4 on CPU - 150000n 1500000e... 12.47ms ±0.50
[221/385] Stage 5 on CPU - 1000n 2000e... 0.92ms ±0.14
[222/385] Stage 5 on CPU - 1000n 3000e... 0.93ms ±0.10
[223/385] Stage 5 on CPU - 1000n 5000e... 0.93ms ±0.09
[224/385] Stage 5 on CPU - 1000n 7000e... 0.90ms ±0.09
[225/385] Stage 5 on CPU - 1000n 10000e... 0.88ms ±0.11
[226/385] Stage 5 on CPU - 5000n 10000e... 3.35ms ±0.20
[227/385] Stage 5 on CPU - 5000n 15000e... 3.36ms ±0.17
[228/385] Stage 5 on CPU - 5000n 25000e... 3.41ms ±0.17
[229/385] Stage 5 on CPU - 5000n 35000e... 3.36ms ±0.21
[230/385] Stage 5 on CPU - 5000n 50000e... 3.41ms ±0.18
[231/385] Stage 5 on CPU - 10000n 20000e... 6.36ms ±0.18
[232/385] Stage 5 on CPU - 10000n 30000e... 6.34ms ±0.17
[233/385] Stage 5 on CPU - 10000n 50000e... 6.44ms ±0.19
[234/385] Stage 5 on CPU - 10000n 70000e... 6.35ms ±0.21
[235/385] Stage 5 on CPU - 10000n 100000e... 6.35ms ±0.21
[236/385] Stage 5 on CPU - 20000n 40000e... 12.68ms ±0.26
[237/385] Stage 5 on CPU - 20000n 60000e... 12.69ms ±0.36
[238/385] Stage 5 on CPU - 20000n 100000e... 12.72ms ±0.34
[239/385] Stage 5 on CPU - 20000n 140000e... 12.68ms ±0.33
[240/385] Stage 5 on CPU - 20000n 200000e... 12.70ms ±0.31
[241/385] Stage 5 on CPU - 30000n 60000e... 18.84ms ±0.41
[242/385] Stage 5 on CPU - 30000n 90000e... 18.87ms ±0.46
[243/385] Stage 5 on CPU - 30000n 150000e... 18.90ms ±0.45
[244/385] Stage 5 on CPU - 30000n 210000e... 19.10ms ±0.50
[245/385] Stage 5 on CPU - 30000n 300000e... 18.93ms ±0.41
[246/385] Stage 5 on CPU - 40000n 80000e... 25.13ms ±0.56
[247/385] Stage 5 on CPU - 40000n 120000e... 25.07ms ±0.69
[248/385] Stage 5 on CPU - 40000n 200000e... 25.09ms ±0.47
[249/385] Stage 5 on CPU - 40000n 280000e... 25.21ms ±0.44
[250/385] Stage 5 on CPU - 40000n 400000e... 25.48ms ±0.77
[251/385] Stage 5 on CPU - 50000n 100000e... 31.57ms ±0.67
[252/385] Stage 5 on CPU - 50000n 150000e... 31.69ms ±0.89
[253/385] Stage 5 on CPU - 50000n 250000e... 31.63ms ±0.69
[254/385] Stage 5 on CPU - 50000n 350000e... 31.48ms ±0.43
[255/385] Stage 5 on CPU - 50000n 500000e... 31.64ms ±0.64
[256/385] Stage 5 on CPU - 60000n 120000e... 37.96ms ±0.86
[257/385] Stage 5 on CPU - 60000n 180000e... 37.85ms ±0.74
[258/385] Stage 5 on CPU - 60000n 300000e... 37.88ms ±0.78
[259/385] Stage 5 on CPU - 60000n 420000e... 38.27ms ±1.09
[260/385] Stage 5 on CPU - 60000n 600000e... 37.77ms ±0.73
[261/385] Stage 5 on CPU - 80000n 160000e... 50.50ms ±1.04
[262/385] Stage 5 on CPU - 80000n 240000e... 50.58ms ±1.21
[263/385] Stage 5 on CPU - 80000n 400000e... 50.47ms ±0.83
[264/385] Stage 5 on CPU - 80000n 560000e... 50.76ms ±1.01
[265/385] Stage 5 on CPU - 80000n 800000e... 50.63ms ±1.37
[266/385] Stage 5 on CPU - 100000n 200000e... 62.98ms ±1.36
[267/385] Stage 5 on CPU - 100000n 300000e... 63.32ms ±1.43
[268/385] Stage 5 on CPU - 100000n 500000e... 63.10ms ±1.32
[269/385] Stage 5 on CPU - 100000n 700000e... 63.44ms ±1.21
[270/385] Stage 5 on CPU - 100000n 1000000e... 63.38ms ±1.48
[271/385] Stage 5 on CPU - 150000n 300000e... 94.89ms ±1.51
[272/385] Stage 5 on CPU - 150000n 450000e... 95.25ms ±1.51
[273/385] Stage 5 on CPU - 150000n 750000e... 94.95ms ±1.70
[274/385] Stage 5 on CPU - 150000n 1050000e... 94.86ms ±1.40
[275/385] Stage 5 on CPU - 150000n 1500000e... 94.84ms ±1.43
[276/385] Stage 6 on CPU - 1000n 2000e... 2.10ms ±0.20
[277/385] Stage 6 on CPU - 1000n 3000e... 2.03ms ±0.14
[278/385] Stage 6 on CPU - 1000n 5000e... 1.99ms ±0.20
[279/385] Stage 6 on CPU - 1000n 7000e... 2.01ms ±0.19
[280/385] Stage 6 on CPU - 1000n 10000e... 1.99ms ±0.15
[281/385] Stage 6 on CPU - 5000n 10000e... 8.25ms ±0.67
[282/385] Stage 6 on CPU - 5000n 15000e... 8.17ms ±0.44
[283/385] Stage 6 on CPU - 5000n 25000e... 8.15ms ±0.38
[284/385] Stage 6 on CPU - 5000n 35000e... 8.12ms ±0.39
[285/385] Stage 6 on CPU - 5000n 50000e... 8.27ms ±0.31
[286/385] Stage 6 on CPU - 10000n 20000e... 15.99ms ±0.47
[287/385] Stage 6 on CPU - 10000n 30000e... 15.88ms ±0.47
[288/385] Stage 6 on CPU - 10000n 50000e... 16.05ms ±0.57
[289/385] Stage 6 on CPU - 10000n 70000e... 15.92ms ±0.31
[290/385] Stage 6 on CPU - 10000n 100000e... 16.11ms ±0.47
[291/385] Stage 6 on CPU - 20000n 40000e... 31.78ms ±1.40
[292/385] Stage 6 on CPU - 20000n 60000e... 31.60ms ±0.93
[293/385] Stage 6 on CPU - 20000n 100000e... 32.05ms ±1.15
[294/385] Stage 6 on CPU - 20000n 140000e... 31.85ms ±0.94
[295/385] Stage 6 on CPU - 20000n 200000e... 31.39ms ±0.76
[296/385] Stage 6 on CPU - 30000n 60000e... 47.20ms ±1.03
[297/385] Stage 6 on CPU - 30000n 90000e... 50.48ms ±6.26
[298/385] Stage 6 on CPU - 30000n 150000e... 47.07ms ±1.03
[299/385] Stage 6 on CPU - 30000n 210000e... 47.04ms ±1.06
[300/385] Stage 6 on CPU - 30000n 300000e... 47.17ms ±1.39
[301/385] Stage 6 on CPU - 40000n 80000e... 63.09ms ±1.98
[302/385] Stage 6 on CPU - 40000n 120000e... 63.19ms ±2.20
[303/385] Stage 6 on CPU - 40000n 200000e... 63.51ms ±3.11
[304/385] Stage 6 on CPU - 40000n 280000e... 62.69ms ±1.72
[305/385] Stage 6 on CPU - 40000n 400000e... 62.70ms ±1.36
[306/385] Stage 6 on CPU - 50000n 100000e... 77.91ms ±1.66
[307/385] Stage 6 on CPU - 50000n 150000e... 78.20ms ±1.51
[308/385] Stage 6 on CPU - 50000n 250000e... 78.17ms ±1.72
[309/385] Stage 6 on CPU - 50000n 350000e... 77.82ms ±1.37
[310/385] Stage 6 on CPU - 50000n 500000e... 78.17ms ±2.05
[311/385] Stage 6 on CPU - 60000n 120000e... 94.08ms ±2.85
[312/385] Stage 6 on CPU - 60000n 180000e... 94.23ms ±2.28
[313/385] Stage 6 on CPU - 60000n 300000e... 94.62ms ±3.16
[314/385] Stage 6 on CPU - 60000n 420000e... 94.43ms ±2.06
[315/385] Stage 6 on CPU - 60000n 600000e... 94.12ms ±4.41
[316/385] Stage 6 on CPU - 80000n 160000e... 125.49ms ±2.81
[317/385] Stage 6 on CPU - 80000n 240000e... 126.79ms ±5.08
[318/385] Stage 6 on CPU - 80000n 400000e... 124.85ms ±2.07
[319/385] Stage 6 on CPU - 80000n 560000e... 124.68ms ±2.51
[320/385] Stage 6 on CPU - 80000n 800000e... 125.11ms ±2.48
[321/385] Stage 6 on CPU - 100000n 200000e... 156.52ms ±2.94
[322/385] Stage 6 on CPU - 100000n 300000e... 156.08ms ±3.59
[323/385] Stage 6 on CPU - 100000n 500000e... 155.47ms ±3.02
[324/385] Stage 6 on CPU - 100000n 700000e... 155.83ms ±3.16
[325/385] Stage 6 on CPU - 100000n 1000000e... 156.13ms ±2.89
[326/385] Stage 6 on CPU - 150000n 300000e... 232.55ms ±3.35
[327/385] Stage 6 on CPU - 150000n 450000e... 237.17ms ±7.36
[328/385] Stage 6 on CPU - 150000n 750000e... 233.20ms ±3.35
[329/385] Stage 6 on CPU - 150000n 1050000e... 232.95ms ±2.81
[330/385] Stage 6 on CPU - 150000n 1500000e... 233.50ms ±4.89
[331/385] Stage 7 on CPU - 1000n 2000e... 0.94ms ±0.11
[332/385] Stage 7 on CPU - 1000n 3000e... 0.94ms ±0.09
[333/385] Stage 7 on CPU - 1000n 5000e... 0.96ms ±0.10
[334/385] Stage 7 on CPU - 1000n 7000e... 0.96ms ±0.09
[335/385] Stage 7 on CPU - 1000n 10000e... 0.94ms ±0.09
[336/385] Stage 7 on CPU - 5000n 10000e... 3.47ms ±0.18
[337/385] Stage 7 on CPU - 5000n 15000e... 3.49ms ±0.14
[338/385] Stage 7 on CPU - 5000n 25000e... 3.40ms ±0.15
[339/385] Stage 7 on CPU - 5000n 35000e... 3.39ms ±0.17
[340/385] Stage 7 on CPU - 5000n 50000e... 3.42ms ±0.19
[341/385] Stage 7 on CPU - 10000n 20000e... 6.36ms ±0.20
[342/385] Stage 7 on CPU - 10000n 30000e... 6.47ms ±0.33
[343/385] Stage 7 on CPU - 10000n 50000e... 6.49ms ±0.27
[344/385] Stage 7 on CPU - 10000n 70000e... 6.42ms ±0.28
[345/385] Stage 7 on CPU - 10000n 100000e... 6.39ms ±0.19
[346/385] Stage 7 on CPU - 20000n 40000e... 12.83ms ±0.50
[347/385] Stage 7 on CPU - 20000n 60000e... 12.84ms ±0.39
[348/385] Stage 7 on CPU - 20000n 100000e... 12.80ms ±0.35
[349/385] Stage 7 on CPU - 20000n 140000e... 12.88ms ±0.36
[350/385] Stage 7 on CPU - 20000n 200000e... 12.87ms ±0.32
[351/385] Stage 7 on CPU - 30000n 60000e... 19.14ms ±0.45
[352/385] Stage 7 on CPU - 30000n 90000e... 19.02ms ±0.47
[353/385] Stage 7 on CPU - 30000n 150000e... 19.02ms ±0.34
[354/385] Stage 7 on CPU - 30000n 210000e... 19.10ms ±0.38
[355/385] Stage 7 on CPU - 30000n 300000e... 18.94ms ±0.40
[356/385] Stage 7 on CPU - 40000n 80000e... 25.35ms ±0.49
[357/385] Stage 7 on CPU - 40000n 120000e... 25.44ms ±0.68
[358/385] Stage 7 on CPU - 40000n 200000e... 25.58ms ±0.54
[359/385] Stage 7 on CPU - 40000n 280000e... 25.45ms ±0.82
[360/385] Stage 7 on CPU - 40000n 400000e... 25.33ms ±0.67
[361/385] Stage 7 on CPU - 50000n 100000e... 31.74ms ±0.82
[362/385] Stage 7 on CPU - 50000n 150000e... 31.69ms ±0.76
[363/385] Stage 7 on CPU - 50000n 250000e... 31.70ms ±0.55
[364/385] Stage 7 on CPU - 50000n 350000e... 31.72ms ±0.76
[365/385] Stage 7 on CPU - 50000n 500000e... 32.00ms ±0.91
[366/385] Stage 7 on CPU - 60000n 120000e... 38.09ms ±0.72
[367/385] Stage 7 on CPU - 60000n 180000e... 38.21ms ±0.99
[368/385] Stage 7 on CPU - 60000n 300000e... 38.35ms ±0.98
[369/385] Stage 7 on CPU - 60000n 420000e... 38.11ms ±0.93
[370/385] Stage 7 on CPU - 60000n 600000e... 38.15ms ±0.88
[371/385] Stage 7 on CPU - 80000n 160000e... 51.02ms ±1.22
[372/385] Stage 7 on CPU - 80000n 240000e... 50.91ms ±1.00
[373/385] Stage 7 on CPU - 80000n 400000e... 50.94ms ±1.09
[374/385] Stage 7 on CPU - 80000n 560000e... 50.98ms ±0.96
[375/385] Stage 7 on CPU - 80000n 800000e... 51.08ms ±1.05
[376/385] Stage 7 on CPU - 100000n 200000e... 63.67ms ±1.17
[377/385] Stage 7 on CPU - 100000n 300000e... 63.91ms ±1.16
[378/385] Stage 7 on CPU - 100000n 500000e... 63.64ms ±1.24
[379/385] Stage 7 on CPU - 100000n 700000e... 63.97ms ±1.51
[380/385] Stage 7 on CPU - 100000n 1000000e... 63.53ms ±1.06
[381/385] Stage 7 on CPU - 150000n 300000e... 95.70ms ±1.60
[382/385] Stage 7 on CPU - 150000n 450000e... 95.91ms ±2.17
[383/385] Stage 7 on CPU - 150000n 750000e... 96.04ms ±1.50
[384/385] Stage 7 on CPU - 150000n 1050000e... 95.64ms ±1.44
[385/385] Stage 7 on CPU - 150000n 1500000e... 95.52ms ±1.49

✓ Measured 385 configurations for CPU
✓ Checkpoint saved: checkpoint_cpu.json (385 entries)

✓ CPU measurements completed and saved!
  Total: 385 entries
  Checkpoint: results/checkpoint_cpu.json

CPU measurement completed successfully

========================================================================
CPU Profiling Complete
========================================================================

Results saved to: profiling\results\checkpoint_cpu.json

Press any key to continue . . .