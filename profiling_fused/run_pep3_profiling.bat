@echo off
REM ========================================================================
REM PEP3 Profiling - Full Workflow (Fused Blocks)
REM ========================================================================
REM
REM PEP3 Configuration:
REM   Block 0: CPU + GPU (30%%:70%% DP) -> FusedBlock0 (Stages 1-4)
REM   Block 1: NPU (100%%) -> FusedBlock1 (Stages 5-7)
REM
REM Test Configuration (based on real dataset analysis):
REM   - Node sizes: 5k, 10k, 20k, 50k, 80k, 100k (6 levels)
REM   - Edge ratios: 10, 25, 40, 50, 60, 75, 100 (7 levels)
REM   - Total: 42 test cases
REM
REM This script runs the profiling in safe order:
REM   1. Export CPU/GPU fused model
REM   2. Measure CPU/GPU latencies -> SAVE IMMEDIATELY
REM   3. Export NPU fused models (42 static models)
REM   4. Measure NPU latencies (with incremental saving)
REM   5. Generate summary

setlocal EnableDelayedExpansion

echo ========================================================================
echo PEP3 Profiling - Full Workflow (Fused Blocks)
echo ========================================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

echo.
echo PEP3 Configuration:
echo   Block 0: CPU + GPU (30%%:70%% DP) - FusedBlock0 (Stages 1-4)
echo   Block 1: NPU (100%%) - FusedBlock1 (Stages 5-7)
echo.
echo Fused Blocks:
echo   Block 0: GATHER + MESSAGE + REDUCE_SUM + REDUCE_COUNT
echo   Block 1: NORMALIZE + TRANSFORM + ACTIVATE
echo.
echo Test Configuration:
echo   Node sizes: 5k, 10k, 20k, 50k, 80k, 100k
echo   Edge ratios: 10, 25, 40, 50, 60, 75, 100
echo   Total: 42 test cases
echo.
echo Estimated time: ~4-6 hours total
echo   - CPU/GPU: ~1-2 hours
echo   - NPU: ~2-4 hours
echo.

REM ========================================================================
REM Step 1: Export and measure CPU/GPU (safe first)
REM ========================================================================
echo.
echo [Step 1/4] Exporting CPU/GPU fused model...
python profile_pep3.py --export-cpugpu
if errorlevel 1 (
    echo ERROR: Failed to export CPU/GPU models
    pause
    exit /b 1
)

echo.
echo [Step 2/4] Measuring CPU/GPU latencies...
python profile_pep3.py --measure-cpugpu
if errorlevel 1 (
    echo WARNING: Some CPU/GPU measurements may have failed
)

echo.
echo ========================================================================
echo CPU/GPU results saved to: profiling_fused\results\block0_cpugpu.json
echo ========================================================================
echo.

REM ========================================================================
REM Step 2: Export and measure NPU (risky, done after CPU/GPU is saved)
REM ========================================================================
echo [Step 3/4] Exporting NPU fused models (42 static models)...
python profile_pep3.py --export-npu
if errorlevel 1 (
    echo ERROR: Failed to export NPU models
    echo CPU/GPU results are still saved, you can run NPU separately later.
    pause
    exit /b 1
)

echo.
echo [Step 4/4] Measuring NPU latencies (with incremental saving)...
python profile_pep3.py --measure-npu
if errorlevel 1 (
    echo WARNING: Some NPU measurements failed (likely at larger sizes)
    echo Partial results have been saved.
)

REM ========================================================================
REM Step 3: Generate analysis
REM ========================================================================
echo.
echo Generating analysis...
python profile_pep3.py --analyze

echo.
echo ========================================================================
echo PEP3 Profiling Complete!
echo ========================================================================
echo.
echo Results saved in: profiling_fused\results\
echo   - block0_cpugpu.json   (CPU/GPU fused block results)
echo   - block1_npu.json      (NPU fused block results)
echo   - pep3_latency.csv     (Summary CSV for analysis)
echo.

pause
