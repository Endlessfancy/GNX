@echo off
REM ========================================================================
REM Fused Block 0-7 Baseline - GPU Only
REM ========================================================================

setlocal EnableDelayedExpansion

echo ========================================================================
echo Fused Block 0-7 Baseline - GPU Only
echo ========================================================================
echo.

CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

cd /d "%~dp0"

REM Check if model exists, export if not
if not exist "exported_models\fused_block0_7_gpu.xml" (
    echo Exporting models...
    python profile_fused_baseline.py --export
)

echo.
echo Measuring GPU latencies...
python profile_fused_baseline.py --measure-gpu

echo.
echo Generating summary...
python profile_fused_baseline.py --analyze

echo.
echo ========================================================================
echo GPU Profiling Complete!
echo ========================================================================
echo Results saved to: results\fused_block0_7_gpu.json
echo.

pause
