@echo off
REM GraphSAGE CPU Profiling
setlocal EnableDelayedExpansion

echo ========================================================================
echo GraphSAGE CPU Profiling
echo ========================================================================
echo.

CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
cd /d "%~dp0"

if not exist "exported_models\stage1_cpu.xml" (
    echo Exporting CPU/GPU models...
    python profile_stages.py --export-cpugpu
)

echo.
echo Running CPU measurements...
python profile_stages.py --measure-cpu --platform 185H

echo.
echo Done! Results: results\185H\sage\checkpoint_cpu.json
pause
