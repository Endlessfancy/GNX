# bug 1
2025.12.14.19:24

========================================================================
GNN Stage Profiling - Intel AI PC
========================================================================

Activating MIX environment...

Environment activated. Running profiling...

This will take approximately 3-4 hours to complete.
The script will:
  1. Export 14 dynamic models (CPU/GPU) + 105 static models (NPU)
  2. Measure 315 latencies (15 sizes x 3 PUs x 7 stages)
  3. Estimate bandwidth and separate compute time
  4. Generate lookup_table.json and bandwidth_table.json

Traceback (most recent call last):
  File "C:\Private\Research\GNX\profiling_v8_incremental\profiling\profile_stages.py", line 33, in <module>
    from models.Model_sage import (
ModuleNotFoundError: No module named 'models'

========================================================================
Profiling completed!
========================================================================

Results are saved in: profiling\results\
  - lookup_table.json      (Compute time for each configuration)
  - bandwidth_table.json   (Bandwidth estimates per PU)
  - profiling_report.txt   (Human-readable statistics)

Press any key to continue . . .