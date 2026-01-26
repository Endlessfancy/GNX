@echo off
REM GAT GPU Profiling
setlocal EnableDelayedExpansion

echo ========================================================================
echo GAT GPU Profiling
echo ========================================================================
echo.

if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else (
    CALL "C:\Users\29067\anaconda3\Scripts\activate.bat" MIX
)
cd /d "%~dp0"

if not exist "gat_exported_models\stage1_gpu.xml" (
    echo Exporting CPU/GPU models...
    python gat_profile_stages.py --export-cpugpu
)

echo.
echo Running GPU measurements...
python gat_profile_stages.py --measure-gpu --platform 265V

echo.
echo Done! Results: results\265V\gat\checkpoint_gpu.json
pause
