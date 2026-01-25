============================================================
CPU Thread Configuration Test (Stage 1-4 Fused Block)
============================================================

Activating MIX environment...

Running all configurations...
============================================================
System Information
============================================================
OS CPU count: 22
Available devices: ['CPU', 'GPU', 'NPU']
CPU full name: Intel(R) Core(TM) Ultra 9 185H
Default threads: 0

============================================================
Model: block0_fused_cpu.xml
Config: Default (actual threads: 0)
Loading model...
Warmup...
Measuring...

Results: 96.11 ms (std: 10.81, min: 83.31, max: 139.97)

============================================================
Model: block0_fused_cpu.xml
Config: 1 threads (actual threads: 1)
Loading model...
Warmup...
Measuring...

Results: 313.77 ms (std: 20.71, min: 283.46, max: 360.76)

============================================================
Model: block0_fused_cpu.xml
Config: 4 threads (actual threads: 4)
Loading model...
Warmup...
Measuring...

Results: 312.80 ms (std: 28.47, min: 233.58, max: 370.16)

============================================================
Model: block0_fused_cpu.xml
Config: 8 threads (actual threads: 8)
Loading model...
Warmup...
Measuring...

Results: 381.39 ms (std: 86.75, min: 257.72, max: 498.60)

============================================================
Model: block0_fused_cpu.xml
Config: 16 threads (actual threads: 16)
Loading model...
Warmup...
Measuring...

Results: 93.08 ms (std: 8.32, min: 79.84, max: 116.18)

============================================================
Model: block0_fused_cpu.xml
Config: Throughput mode (actual threads: 0)
Loading model...
Warmup...
Measuring...

Results: 235.58 ms (std: 149.86, min: 110.24, max: 495.20)

============================================================
Summary
============================================================
  Default                 96.11 ms  (1.00x)
  1 thread               313.77 ms  (0.31x)
  4 threads              312.80 ms  (0.31x)
  8 threads              381.39 ms  (0.25x)
  16 threads              93.08 ms  (1.03x)
  Throughput             235.58 ms  (0.41x)

============================================================
Test completed
=======================================================