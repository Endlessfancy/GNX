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
echo This will take approximately 3-4 hours to complete.
echo The script will:
echo   1. Export 14 dynamic models (CPU/GPU) + 105 static models (NPU)
echo   2. Measure 315 latencies (15 sizes x 3 PUs x 7 stages)
echo   3. Estimate bandwidth and separate compute time
echo   4. Generate lookup_table.json and bandwidth_table.json
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
