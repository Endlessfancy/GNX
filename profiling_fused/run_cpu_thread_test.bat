@echo off
REM CPU Thread Configuration Test for OpenVINO
REM Tests different thread settings to find optimal CPU utilization

setlocal EnableDelayedExpansion

echo ============================================================
echo CPU Thread Configuration Test (Stage 1-4 Fused Block)
echo ============================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

cd /d "%~dp0"

echo Test Configuration:
echo   Model: block0_fused_cpu.xml (Stage 1-4 Fused)
echo   Data:  10k nodes, 100k edges
echo   Configs: Default, 1/4/8/16 threads, Throughput mode
echo.

python test_cpu_threads.py --all

echo.
echo ============================================================
echo Test completed!
echo ============================================================
echo.

pause
