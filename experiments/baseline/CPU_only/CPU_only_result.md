(GNX) PS D:\Research\GNX\experiments\baseline\CPU_only> python .\run_cpu_baseline.py
================================================================================
CPU-only Baseline Test
================================================================================

Loading Flickr dataset...
Downloading https://drive.usercontent.google.com/download?id=1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy&confirm=t
Downloading https://drive.usercontent.google.com/download?id=1join-XdvX3anJU_MLVtick7MgeAQiWIZ&confirm=t
Downloading https://drive.usercontent.google.com/download?id=1uxIkbtg5drHTsKt-PAsZZ4_yJmgFmle9&confirm=t
Downloading https://drive.usercontent.google.com/download?id=1htXCtuktuCW8TR8KiKfrFDAxUgekQoV7&confirm=t
Processing...
Done!
  Nodes: 89,250
  Edges: 899,756
  Features: 500

Initializing GraphSAGE model...
  Model parameters: 387,584
  Device: cpu

Running 2 warmup iterations...
  Warmup 1/2 completed
  Warmup 2/2 completed

Running 10 timed iterations...
  Run 1/10: 534.20ms
  Run 2/10: 492.14ms
  Run 3/10: 501.76ms
  Run 4/10: 484.50ms
  Run 5/10: 480.89ms
  Run 6/10: 444.82ms
  Run 7/10: 428.23ms
  Run 8/10: 426.76ms
  Run 9/10: 432.51ms
  Run 10/10: 443.90ms

================================================================================
Results
================================================================================
Output shape: torch.Size([89250, 256])

Timing Statistics (10 runs):
  Mean:   466.97 ms
  Std:    36.81 ms
  Min:    426.76 ms
  Max:    534.20 ms

Results saved to: D:\Research\GNX\experiments\baseline\CPU_only\cpu_baseline_results.txt

================================================================================
CPU Baseline Test Completed!