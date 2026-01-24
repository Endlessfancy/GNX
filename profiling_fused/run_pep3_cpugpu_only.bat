@echo off
REM ========================================================================
REM PEP3 CPU/GPU Only Profiling - Fused Block 0
REM ========================================================================
REM
REM This script tests FUSED Block 0 (Stages 1-4) on CPU and GPU
REM   - FusedBlock0: GATHER + MESSAGE + REDUCE_SUM + REDUCE_COUNT
REM
REM Results are saved IMMEDIATELY after completion to protect against NPU failures.
REM
REM Test Configuration (based on real dataset analysis):
REM   - Node sizes: 5k, 10k, 20k, 50k, 80k, 100k
REM   - Edge ratios: 10, 25, 40, 50, 60, 75, 100
REM   - Total: 42 test cases x 2 devices = 84 measurements

setlocal EnableDelayedExpansion

echo ========================================================================
echo PEP3 CPU/GPU Profiling - Fused Block 0 (Stages 1-4)
echo ========================================================================
echo.

REM Try multiple conda paths
echo Activating MIX environment...
if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    CALL "%USERPROFILE%\anaconda3\Scripts\activate.bat" MIX
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    CALL "%USERPROFILE%\miniconda3\Scripts\activate.bat" MIX
) else if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    CALL "C:\ProgramData\anaconda3\Scripts\activate.bat" MIX
) else (
    echo WARNING: Could not find conda. Please activate MIX environment manually first.
    echo Then run: python profile_pep3.py --measure-cpugpu
)

echo.
echo Fused Block 0: GATHER + MESSAGE + REDUCE_SUM + REDUCE_COUNT
echo.
echo Test Configuration:
echo   Node sizes: 5k, 10k, 20k, 50k, 80k, 100k (6 levels)
echo   Edge ratios: 10, 25, 40, 50, 60, 75, 100 (7 levels)
echo   Total test cases: 42 x 2 devices = 84 measurements
echo.

REM Check if models exist
if not exist "exported_models\block0_fused_cpu.xml" (
    echo CPU/GPU fused models not found! Exporting...
    python profile_pep3.py --export-cpugpu
    if errorlevel 1 (
        echo ERROR: Failed to export CPU/GPU models
        pause
        exit /b 1
    )
)

REM Measure CPU/GPU only
echo.
echo Starting CPU/GPU measurement...
echo Estimated time: ~1-2 hours
echo.

python profile_pep3.py --measure-cpugpu

if errorlevel 1 (
    echo.
    echo WARNING: CPU/GPU measurement encountered errors
    echo Check the results directory for partial results
) else (
    echo.
    echo CPU/GPU measurement completed successfully!
)

echo.
echo ========================================================================
echo CPU/GPU Profiling Complete!
echo ========================================================================
echo.
echo Results saved to: profiling_fused\results\block0_cpugpu.json
echo.
echo IMPORTANT: CPU/GPU results are now safely saved.
echo You can run NPU tests separately with: run_pep3_npu_only.bat
echo.

pause
