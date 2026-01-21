@echo off
REM ========================================================================
REM PEP3 NPU Only Profiling - Process Isolation Mode
REM ========================================================================
REM
REM This script tests FUSED Block 1 (Stages 5-7) on NPU with PROCESS ISOLATION
REM   - FusedBlock1: NORMALIZE + TRANSFORM + ACTIVATE
REM
REM IMPORTANT: Each node size runs in a SEPARATE Python process!
REM   - If NPU crashes at 50k nodes, 80k/100k tests can still run
REM   - This helps find the NPU memory boundary
REM
REM Test Configuration:
REM   - Node sizes: 5k, 10k, 20k, 50k, 80k, 100k (6 groups)
REM   - Edge ratios: 10, 25, 40, 50, 60, 75, 100 (7 per group)
REM   - Total: 42 static NPU models

setlocal EnableDelayedExpansion

echo ========================================================================
echo PEP3 NPU Profiling - Fused Block 1 (Process Isolation Mode)
echo ========================================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

echo.
echo Fused Block 1: NORMALIZE + TRANSFORM + ACTIVATE
echo.
echo PROCESS ISOLATION: Each node size runs in a separate Python process
echo   - If NPU crashes at one size, next size starts fresh
echo   - Helps find NPU memory boundaries
echo.
echo Node sizes: 5k, 10k, 20k, 50k, 80k, 100k (6 groups)
echo Edge ratios: 10, 25, 40, 50, 60, 75, 100 (7 per group)
echo.

REM Check if NPU models exist
if not exist "exported_models\block1_fused_npu_n5000_e50000.xml" (
    echo NPU fused models not found! Exporting...
    echo This will create 42 static models...
    echo.
    python profile_pep3.py --export-npu
    if errorlevel 1 (
        echo ERROR: Failed to export NPU models
        pause
        exit /b 1
    )
)

REM Create results directory
if not exist "results" mkdir results

echo.
echo Starting NPU measurements with process isolation...
echo.

REM Test each node size in a separate process
set "ALL_SUCCESS=1"

echo [1/6] Testing 5000 nodes...
python profile_npu_isolated.py --nodes 5000
if errorlevel 2 (
    echo   ^> All tests FAILED for 5k nodes
    set "ALL_SUCCESS=0"
) else if errorlevel 1 (
    echo   ^> Some tests failed for 5k nodes
    set "ALL_SUCCESS=0"
) else (
    echo   ^> All tests PASSED for 5k nodes
)
echo.

echo [2/6] Testing 10000 nodes...
python profile_npu_isolated.py --nodes 10000
if errorlevel 2 (
    echo   ^> All tests FAILED for 10k nodes
    set "ALL_SUCCESS=0"
) else if errorlevel 1 (
    echo   ^> Some tests failed for 10k nodes
    set "ALL_SUCCESS=0"
) else (
    echo   ^> All tests PASSED for 10k nodes
)
echo.

echo [3/6] Testing 20000 nodes...
python profile_npu_isolated.py --nodes 20000
if errorlevel 2 (
    echo   ^> All tests FAILED for 20k nodes
    set "ALL_SUCCESS=0"
) else if errorlevel 1 (
    echo   ^> Some tests failed for 20k nodes
    set "ALL_SUCCESS=0"
) else (
    echo   ^> All tests PASSED for 20k nodes
)
echo.

echo [4/6] Testing 50000 nodes...
python profile_npu_isolated.py --nodes 50000
if errorlevel 2 (
    echo   ^> All tests FAILED for 50k nodes
    set "ALL_SUCCESS=0"
) else if errorlevel 1 (
    echo   ^> Some tests failed for 50k nodes
    set "ALL_SUCCESS=0"
) else (
    echo   ^> All tests PASSED for 50k nodes
)
echo.

echo [5/6] Testing 80000 nodes...
python profile_npu_isolated.py --nodes 80000
if errorlevel 2 (
    echo   ^> All tests FAILED for 80k nodes
    set "ALL_SUCCESS=0"
) else if errorlevel 1 (
    echo   ^> Some tests failed for 80k nodes
    set "ALL_SUCCESS=0"
) else (
    echo   ^> All tests PASSED for 80k nodes
)
echo.

echo [6/6] Testing 100000 nodes...
python profile_npu_isolated.py --nodes 100000
if errorlevel 2 (
    echo   ^> All tests FAILED for 100k nodes
    set "ALL_SUCCESS=0"
) else if errorlevel 1 (
    echo   ^> Some tests failed for 100k nodes
    set "ALL_SUCCESS=0"
) else (
    echo   ^> All tests PASSED for 100k nodes
)
echo.

REM Merge all results
echo ========================================================================
echo Merging results from all node sizes...
echo ========================================================================
python merge_npu_results.py

echo.
echo ========================================================================
echo NPU Profiling Complete!
echo ========================================================================
echo.
echo Individual results: results\npu_n*.json
echo Merged results: results\block1_npu.json
echo.
echo Next steps:
echo   1. Run analysis: python profile_pep3.py --analyze
echo   2. Check CSV: results\pep3_latency.csv
echo.

pause
