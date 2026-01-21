@echo off
REM ========================================================================
REM PEP3 NPU Only Profiling - Fused Block 1
REM ========================================================================
REM
REM This script tests FUSED Block 1 (Stages 5-7) on NPU
REM   - FusedBlock1: NORMALIZE + TRANSFORM + ACTIVATE
REM
REM Run this AFTER CPU/GPU tests are complete and saved.
REM
REM WARNING: NPU may fail at large sizes. This script will:
REM   - Save results incrementally (every 5 tests)
REM   - Continue even if some tests fail
REM   - Support resume from previous runs
REM
REM Test Configuration:
REM   - Node sizes: 5k, 10k, 20k, 50k, 80k, 100k
REM   - Edge ratios: 10, 25, 40, 50, 60, 75, 100
REM   - Total: 42 static NPU models

setlocal EnableDelayedExpansion

echo ========================================================================
echo PEP3 NPU Profiling - Fused Block 1 (Stages 5-7)
echo ========================================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

echo.
echo Fused Block 1: NORMALIZE + TRANSFORM + ACTIVATE
echo.
echo WARNING: NPU tests may fail at large data sizes
echo Results will be saved incrementally to prevent data loss.
echo Supports resume from previous runs (skips successful tests).
echo.
echo Test Configuration:
echo   Node sizes: 5k, 10k, 20k, 50k, 80k, 100k
echo   Edge ratios: 10, 25, 40, 50, 60, 75, 100
echo   Total: 42 static NPU models
echo.

REM Check if any NPU models exist (check smallest size first)
if not exist "exported_models\block1_fused_npu_n5000_e50000.xml" (
    echo NPU fused models not found! Exporting...
    echo This will create 42 static models (may take 5-10 minutes)
    echo.
    python profile_pep3.py --export-npu
    if errorlevel 1 (
        echo ERROR: Failed to export NPU models
        pause
        exit /b 1
    )
)

REM Measure NPU only
echo.
echo Starting NPU measurement...
echo Estimated time: ~2-4 hours
echo Results are saved every 5 tests - safe to interrupt if needed.
echo.

python profile_pep3.py --measure-npu

if errorlevel 1 (
    echo.
    echo ========================================================================
    echo WARNING: NPU measurement encountered errors
    echo ========================================================================
    echo.
    echo Some NPU tests may have failed (likely at larger sizes).
    echo Partial results have been saved to: profiling_fused\results\block1_npu.json
    echo.
    echo You can re-run this script to retry failed tests.
    echo.
) else (
    echo.
    echo ========================================================================
    echo NPU Profiling Complete!
    echo ========================================================================
    echo.
    echo Results saved to: profiling_fused\results\block1_npu.json
    echo.
)

echo.
echo Next steps:
echo   1. Run analysis: python profile_pep3.py --analyze
echo   2. Check results: profiling_fused\results\pep3_latency.csv
echo.

pause
