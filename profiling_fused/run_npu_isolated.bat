@echo off
REM ========================================================================
REM NPU Isolated Testing - Process Isolation per Node Size
REM ========================================================================
REM
REM This script tests NPU FusedBlock1 with PROCESS ISOLATION:
REM   - Each node size runs in a SEPARATE Python process
REM   - If NPU crashes at 50k nodes, 80k/100k tests can still run
REM   - Helps find the NPU memory boundary for each node size
REM
REM Test order (small to large):
REM   5k -> 10k -> 20k -> 50k -> 80k -> 100k
REM
REM Each node size tests all 7 edge ratios: 10, 25, 40, 50, 60, 75, 100

setlocal EnableDelayedExpansion

echo ========================================================================
echo NPU Isolated Testing - Process Isolation per Node Size
echo ========================================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

echo.
echo Strategy: Each node size runs in a separate Python process
echo If NPU crashes at one size, the next size starts fresh!
echo.
echo Node sizes to test: 5k, 10k, 20k, 50k, 80k, 100k
echo Edge ratios per node: 10, 25, 40, 50, 60, 75, 100
echo.

REM Check if NPU models exist (new format: no edge in filename)
if not exist "exported_models\block1_fused_npu_n5000.xml" (
    echo NPU models not found! Please run: python profile_pep3.py --export-npu
    pause
    exit /b 1
)

REM Create results directory
if not exist "results" mkdir results

REM Test each node size in a separate process
echo.
echo ========== Testing 5k nodes ==========
python profile_npu_isolated.py --nodes 5000
echo Exit code: %ERRORLEVEL%
echo.

echo ========== Testing 10k nodes ==========
python profile_npu_isolated.py --nodes 10000
echo Exit code: %ERRORLEVEL%
echo.

echo ========== Testing 20k nodes ==========
python profile_npu_isolated.py --nodes 20000
echo Exit code: %ERRORLEVEL%
echo.

echo ========== Testing 50k nodes ==========
python profile_npu_isolated.py --nodes 50000
echo Exit code: %ERRORLEVEL%
echo.

echo ========== Testing 80k nodes ==========
python profile_npu_isolated.py --nodes 80000
echo Exit code: %ERRORLEVEL%
echo.

echo ========== Testing 100k nodes ==========
python profile_npu_isolated.py --nodes 100000
echo Exit code: %ERRORLEVEL%
echo.

REM Merge results
echo.
echo ========================================================================
echo Merging results...
echo ========================================================================
python merge_npu_results.py

echo.
echo ========================================================================
echo NPU Isolated Testing Complete!
echo ========================================================================
echo.
echo Individual results: results\npu_n*.json
echo Merged results: results\block1_npu.json
echo.
echo Exit codes meaning:
echo   0 = All tests passed
echo   1 = Some tests failed (partial success)
echo   2 = All tests failed (likely NPU limit reached)
echo.

pause
