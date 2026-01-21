@echo off
REM ========================================================================
REM GPU + NPU Testing - Fused Blocks (Skip CPU)
REM ========================================================================
REM
REM This script tests GPU and NPU only (skips CPU for faster testing):
REM   Block 0: GPU (FusedBlock0 - Stages 1-4: GATHER + REDUCE)
REM            Tests all 42 node×edge combinations
REM   Block 1: NPU (FusedBlock1 - Stages 5-7: NORMALIZE + TRANSFORM + ACTIVATE)
REM            Tests 6 node sizes only (edge-independent!)
REM
REM NPU uses PROCESS ISOLATION to handle potential crashes gracefully.

setlocal EnableDelayedExpansion

echo ========================================================================
echo GPU + NPU Testing - Fused Blocks
echo ========================================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

echo.
echo Test Configuration:
echo   Block 0 (GPU): 42 tests (node x edge combinations)
echo   Block 1 (NPU): 6 tests (node sizes only - edge independent!)
echo.
echo Node sizes: 5k, 10k, 20k, 50k, 80k, 100k
echo.

REM Create results directory
if not exist "results" mkdir results

REM ========================================================================
REM Step 1: Export models (if needed)
REM ========================================================================
echo.
echo [Step 1/4] Checking and exporting models...

REM Check GPU model
if not exist "exported_models\block0_fused_gpu.xml" (
    echo   Exporting CPU/GPU fused model...
    python profile_pep3.py --export-cpugpu
    if errorlevel 1 (
        echo ERROR: Failed to export CPU/GPU models
        pause
        exit /b 1
    )
) else (
    echo   CPU/GPU model already exists, skipping export.
)

REM Check NPU models (new format: no edge in filename)
if not exist "exported_models\block1_fused_npu_n5000.xml" (
    echo   Exporting NPU fused models (6 static models - one per node size)...
    python profile_pep3.py --export-npu
    if errorlevel 1 (
        echo ERROR: Failed to export NPU models
        pause
        exit /b 1
    )
) else (
    echo   NPU models already exist, skipping export.
)

REM ========================================================================
REM Step 2: Measure GPU
REM ========================================================================
echo.
echo [Step 2/4] Measuring GPU latencies (42 test cases)...
python profile_pep3.py --measure-gpu
if errorlevel 1 (
    echo WARNING: Some GPU measurements may have failed
)

echo.
echo GPU results saved to: results\block0_gpu.json

REM ========================================================================
REM Step 3: Measure NPU with PROCESS ISOLATION
REM ========================================================================
echo.
echo [Step 3/4] Measuring NPU latencies (6 node sizes, edge-independent)...
echo Each node size runs in a separate Python process.
echo.

echo   [3.1/6] Testing 5000 nodes...
python profile_npu_isolated.py --nodes 5000
if errorlevel 1 (
    echo     ^> FAILED for 5k nodes
) else (
    echo     ^> PASSED for 5k nodes
)

echo   [3.2/6] Testing 10000 nodes...
python profile_npu_isolated.py --nodes 10000
if errorlevel 1 (
    echo     ^> FAILED for 10k nodes
) else (
    echo     ^> PASSED for 10k nodes
)

echo   [3.3/6] Testing 20000 nodes...
python profile_npu_isolated.py --nodes 20000
if errorlevel 1 (
    echo     ^> FAILED for 20k nodes
) else (
    echo     ^> PASSED for 20k nodes
)

echo   [3.4/6] Testing 50000 nodes...
python profile_npu_isolated.py --nodes 50000
if errorlevel 1 (
    echo     ^> FAILED for 50k nodes
) else (
    echo     ^> PASSED for 50k nodes
)

echo   [3.5/6] Testing 80000 nodes...
python profile_npu_isolated.py --nodes 80000
if errorlevel 1 (
    echo     ^> FAILED for 80k nodes
) else (
    echo     ^> PASSED for 80k nodes
)

echo   [3.6/6] Testing 100000 nodes...
python profile_npu_isolated.py --nodes 100000
if errorlevel 1 (
    echo     ^> FAILED for 100k nodes
) else (
    echo     ^> PASSED for 100k nodes
)

echo.
echo Merging NPU results...
python merge_npu_results.py

REM ========================================================================
REM Step 4: Generate analysis
REM ========================================================================
echo.
echo [Step 4/4] Generating analysis...
python profile_pep3.py --analyze

echo.
echo ========================================================================
echo GPU + NPU Testing Complete!
echo ========================================================================
echo.
echo Results:
echo   - GPU:  results\block0_gpu.json (42 test cases)
echo   - NPU:  results\block1_npu.json (6 node sizes)
echo   - Summary CSV: results\pep3_latency.csv
echo.

pause
