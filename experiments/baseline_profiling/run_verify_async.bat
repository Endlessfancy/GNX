@echo off
REM GPU Async vs Sync Verification Script
REM Compares synchronous and asynchronous inference methods

setlocal EnableDelayedExpansion

echo ============================================================
echo GPU Async vs Sync Verification
echo ============================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

cd /d "%~dp0"

echo Running async verification test...
echo.

python verify_gpu_async.py

echo.
echo ============================================================
echo Verification complete
echo ============================================================
echo Results saved to: results\gpu_async_verification_results.json
echo.

pause
