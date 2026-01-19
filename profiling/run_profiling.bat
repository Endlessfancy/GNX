@echo off
REM Batch script to run profiling on Intel AI PC (Windows)
REM
REM This script profiles the 7 GraphSAGE stages across different input sizes
REM and generates lookup tables for the compilation phase.

echo ========================================================================
echo GNN Stage Profiling - Intel AI PC
echo ========================================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

echo.
echo Environment activated. Running profiling...
echo.
echo This will take approximately 4-6 hours to complete.
echo The script will:
echo   1. Export 14 dynamic models (CPU/GPU) + 90 static models (NPU, skip Stage 3/4)
echo   2. Measure 342 latencies (18 sizes x 3 PUs x 7 stages, NPU skip Stage 3/4)
echo   3. Estimate bandwidth and separate compute time
echo   4. Generate lookup_table.json and bandwidth_table.json
echo.
echo Test cases: 1k-100k nodes (18 combinations)
echo   - Small:  1k, 5k nodes
echo   - Medium: 10k, 20k nodes
echo   - Large:  50k, 100k nodes (covers actual subgraph sizes)
echo.

REM Run profiling with all steps
python profile_stages.py --all

echo.
echo ========================================================================
echo Profiling completed!
echo ========================================================================
echo.
echo Results are saved in: profiling\results\
echo   - lookup_table.json      (Compute time for each configuration)
echo   - bandwidth_table.json   (Bandwidth estimates per PU)
echo   - profiling_report.txt   (Human-readable statistics)
echo.

pause
