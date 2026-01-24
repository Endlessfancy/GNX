@echo off
REM CPU Async vs Sync Verification Script
REM Uses smaller test cases for faster CPU verification

setlocal EnableDelayedExpansion

echo ============================================================
echo CPU Async vs Sync Verification
echo ============================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

cd /d "%~dp0"

echo Running CPU async verification test...
echo (Using smaller test cases for faster results)
echo.

python verify_cpu_async.py

echo.
echo ============================================================
echo Verification complete
echo ============================================================
echo Results saved to: results\cpu_async_verification_results.json
echo.

pause
