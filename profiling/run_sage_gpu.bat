@echo off
REM GraphSAGE GPU Profiling
setlocal EnableDelayedExpansion

echo ========================================================================
echo GraphSAGE GPU Profiling
echo ========================================================================
echo.

CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
cd /d "%~dp0"

if not exist "exported_models\stage1_gpu.xml" (
    echo Exporting CPU/GPU models...
    python profile_stages.py --export-cpugpu
)

echo.
echo Running GPU measurements...
python profile_stages.py --measure-gpu --platform 265V

echo.
echo Done! Results: results\265V\sage\checkpoint_gpu.json
pause
