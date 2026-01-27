@echo off
REM ========================================================================
REM Measure Graph Partition Time
REM ========================================================================

setlocal EnableDelayedExpansion

echo ========================================================================
echo Measure Graph Partition Time
echo ========================================================================
echo.
echo Datasets: Flickr, Reddit2, ogbn-products
echo Partition K values: 2, 4, 8, 16
echo.

if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else (
    CALL "C:\Users\29067\anaconda3\Scripts\activate.bat" MIX
)
cd /d "%~dp0"

echo Running partition time measurement...
echo.

python measure_partition_time.py --all

echo.
echo ========================================================================
echo Measurement Complete!
echo ========================================================================
echo Results saved in: experiments\time\results\
echo.

pause
