@echo off
REM ========================================================================
REM Measure Model Export Time
REM ========================================================================

setlocal EnableDelayedExpansion

echo ========================================================================
echo Measure Model Export Time
echo ========================================================================
echo.
echo CPU/GPU: FusedBlock0 (stages 1-4)
echo NPU:     FusedBlock1 (stages 5-7)
echo.

if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else (
    CALL "C:\Users\29067\anaconda3\Scripts\activate.bat" MIX
)
cd /d "%~dp0"

echo Running export time measurement...
echo.

python measure_export_time.py --all

echo.
echo ========================================================================
echo Measurement Complete!
echo ========================================================================
echo Results saved in: experiments\time\results\
echo.

pause
