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
REM   4. Measure NPU latencies (PROCESS ISOLATION per node size)
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
echo   Block 0: GATHER + REDUCE (aggregation phase)
echo   Block 1: NORMALIZE + TRANSFORM + ACTIVATE (update phase)
echo.
echo Test Configuration:
echo   Node sizes: 5k, 10k, 20k, 50k, 80k, 100k
echo   Edge ratios: 10, 25, 40, 50, 60, 75, 100
echo   Total: 42 test cases
echo.
echo Estimated time: ~4-6 hours total
echo   - CPU/GPU: ~1-2 hours
echo   - NPU: ~2-4 hours (with process isolation)
echo.

REM Create results directory
if not exist "results" mkdir results

REM ========================================================================
REM Step 1: Export and measure CPU/GPU (safe first)
REM ========================================================================
echo.
echo [Step 1/5] Exporting CPU/GPU fused model...
python profile_pep3.py --export-cpugpu
if errorlevel 1 (
    echo ERROR: Failed to export CPU/GPU models
    pause
    exit /b 1
)

echo.
echo [Step 2/5] Measuring CPU/GPU latencies...
python profile_pep3.py --measure-cpugpu
if errorlevel 1 (
    echo WARNING: Some CPU/GPU measurements may have failed
)

echo.
echo ========================================================================
echo CPU/GPU results saved to: results\block0_cpugpu.json
echo ========================================================================
echo.

REM ========================================================================
REM Step 2: Export NPU models
REM ========================================================================
echo [Step 3/5] Exporting NPU fused models (42 static models)...
python profile_pep3.py --export-npu
if errorlevel 1 (
    echo ERROR: Failed to export NPU models
    echo CPU/GPU results are still saved, you can run NPU separately later.
    pause
    exit /b 1
)

REM ========================================================================
REM Step 3: Measure NPU with PROCESS ISOLATION
REM ========================================================================
echo.
echo [Step 4/5] Measuring NPU latencies (PROCESS ISOLATION mode)...
echo Each node size runs in a separate Python process to handle crashes.
echo.

echo   [4.1/6] Testing 5000 nodes...
python profile_npu_isolated.py --nodes 5000
if errorlevel 1 echo     Some tests failed for 5k nodes

echo   [4.2/6] Testing 10000 nodes...
python profile_npu_isolated.py --nodes 10000
if errorlevel 1 echo     Some tests failed for 10k nodes

echo   [4.3/6] Testing 20000 nodes...
python profile_npu_isolated.py --nodes 20000
if errorlevel 1 echo     Some tests failed for 20k nodes

echo   [4.4/6] Testing 50000 nodes...
python profile_npu_isolated.py --nodes 50000
if errorlevel 1 echo     Some tests failed for 50k nodes

echo   [4.5/6] Testing 80000 nodes...
python profile_npu_isolated.py --nodes 80000
if errorlevel 1 echo     Some tests failed for 80k nodes

echo   [4.6/6] Testing 100000 nodes...
python profile_npu_isolated.py --nodes 100000
if errorlevel 1 echo     Some tests failed for 100k nodes

echo.
echo Merging NPU results...
python merge_npu_results.py

REM ========================================================================
REM Step 4: Generate analysis
REM ========================================================================
echo.
echo [Step 5/5] Generating analysis...
python profile_pep3.py --analyze

echo.
echo ========================================================================
echo PEP3 Profiling Complete!
echo ========================================================================
echo.
echo Results saved in: profiling_fused\results\
echo   - block0_cpugpu.json   (CPU/GPU fused block results)
echo   - block1_npu.json      (NPU fused block results - merged)
echo   - npu_n*.json          (NPU individual node size results)
echo   - pep3_latency.csv     (Summary CSV for analysis)
echo.

pause
