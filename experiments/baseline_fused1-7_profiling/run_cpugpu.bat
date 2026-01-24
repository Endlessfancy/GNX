@echo off
REM ========================================================================
REM Fused Block 0-7 Baseline - CPU + GPU
REM ========================================================================

setlocal EnableDelayedExpansion

echo ========================================================================
echo Fused Block 0-7 Baseline - CPU + GPU
echo ========================================================================
echo.

CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

cd /d "%~dp0"

REM Check if models exist, export if not
if not exist "exported_models\fused_block0_7_cpu.xml" (
    echo Exporting models...
    python profile_fused_baseline.py --export
)

echo.
echo Measuring CPU and GPU latencies...
python profile_fused_baseline.py --measure-cpugpu

echo.
echo Generating summary...
python profile_fused_baseline.py --analyze

echo.
echo ========================================================================
echo CPU + GPU Profiling Complete!
echo ========================================================================
echo Results saved to: results\fused_block0_7_cpugpu.json
echo Summary saved to: results\summary.md
echo.

pause
