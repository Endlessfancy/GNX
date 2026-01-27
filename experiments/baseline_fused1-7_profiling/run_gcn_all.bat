@echo off
REM GCN Fused Baseline - Full Profiling (CPU + GPU)
setlocal EnableDelayedExpansion

echo ========================================================================
echo GCN Fused Baseline - Full Profiling (CPU + GPU)
echo ========================================================================
echo.

if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else (
    CALL "C:\Users\29067\anaconda3\Scripts\activate.bat" MIX
)
cd /d "%~dp0"

echo Running full GCN baseline profiling...
python profile_gcn_baseline.py --all --platform 265V

echo.
echo ========================================================================
echo GCN Profiling Complete!
echo ========================================================================
echo Results: results\265V\gcn\
echo.

pause
