@echo off
REM GCN GPU Profiling
setlocal EnableDelayedExpansion

echo ========================================================================
echo GCN GPU Profiling
echo ========================================================================
echo.

CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
cd /d "%~dp0"

if not exist "gcn_exported_models\stage1_gpu.xml" (
    echo Exporting CPU/GPU models...
    python gcn_profile_stages.py --export-cpugpu
)

echo.
echo Running GPU measurements...
python gcn_profile_stages.py --measure-gpu --platform 185H

echo.
echo Done! Results: results\185H\gcn\checkpoint_gpu.json
pause
