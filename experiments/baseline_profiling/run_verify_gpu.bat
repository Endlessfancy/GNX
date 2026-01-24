@echo off
REM GPU Latency Verification Script
REM Compares different measurement methods to verify GPU profiling accuracy

setlocal EnableDelayedExpansion

echo ============================================================
echo GPU Latency Verification
echo ============================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

cd /d "%~dp0"

echo Running GPU verification test...
echo.

python verify_gpu_latency.py

echo.
echo ============================================================
echo Verification complete
echo ============================================================
echo Results saved to: results\gpu_verification_results.json
echo.

pause
