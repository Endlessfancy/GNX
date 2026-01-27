@echo off
REM GCN Fused Baseline - GPU Profiling
setlocal EnableDelayedExpansion

echo ========================================================================
echo GCN Fused Baseline - GPU Profiling
echo ========================================================================
echo.

if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else (
    CALL "C:\Users\29067\anaconda3\Scripts\activate.bat" MIX
)
cd /d "%~dp0"

if not exist "exported_models\fused_gcn_gpu.xml" (
    echo Exporting GCN models...
    python profile_gcn_baseline.py --export
)

echo.
echo Running GPU measurements...
python profile_gcn_baseline.py --measure-gpu --platform 265V

echo.
echo Done! Results: results\265V\gcn\fused_gcn_gpu.json
pause
