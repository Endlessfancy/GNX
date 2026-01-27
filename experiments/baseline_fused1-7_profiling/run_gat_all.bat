@echo off
REM GAT Fused Baseline - Full Profiling (CPU + GPU)
setlocal EnableDelayedExpansion

echo ========================================================================
echo GAT Fused Baseline - Full Profiling (CPU + GPU)
echo ========================================================================
echo.

if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else (
    CALL "C:\Users\29067\anaconda3\Scripts\activate.bat" MIX
)
cd /d "%~dp0"

echo Running full GAT baseline profiling...
python profile_gat_baseline.py --all --platform 265V

echo.
echo ========================================================================
echo GAT Profiling Complete!
echo ========================================================================
echo Results: results\265V\gat\
echo.

pause
