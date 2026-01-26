@echo off
REM GAT CPU Profiling
setlocal EnableDelayedExpansion

echo ========================================================================
echo GAT CPU Profiling
echo ========================================================================
echo.

CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
cd /d "%~dp0"

if not exist "gat_exported_models\stage1_cpu.xml" (
    echo Exporting CPU/GPU models...
    python gat_profile_stages.py --export-cpugpu
)

echo.
echo Running CPU measurements...
python gat_profile_stages.py --measure-cpu --platform 185H

echo.
echo Done! Results: results\185H\gat\checkpoint_cpu.json
pause
