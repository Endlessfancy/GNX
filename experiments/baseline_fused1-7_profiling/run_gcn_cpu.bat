@echo off
REM GCN Fused Baseline - CPU Profiling
setlocal EnableDelayedExpansion

echo ========================================================================
echo GCN Fused Baseline - CPU Profiling
echo ========================================================================
echo.

if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else (
    CALL "C:\Users\29067\anaconda3\Scripts\activate.bat" MIX
)
cd /d "%~dp0"

if not exist "exported_models\fused_gcn_cpu.xml" (
    echo Exporting GCN models...
    python profile_gcn_baseline.py --export
)

echo.
echo Running CPU measurements...
python profile_gcn_baseline.py --measure-cpu --platform 265V

echo.
echo Done! Results: results\265V\gcn\fused_gcn_cpu.json
pause
