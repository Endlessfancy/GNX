@echo off
REM ========================================================================
REM Fused Block 0-7 Baseline - Full Workflow
REM ========================================================================
REM Tests FusedBlock0_7 (all 7 stages combined) on CPU and GPU
REM as a fair baseline for comparing against the multi-device pipeline.

setlocal EnableDelayedExpansion

echo ========================================================================
echo Fused Block 0-7 Baseline Profiling
echo ========================================================================
echo.
echo This baseline uses the same stage implementations as the pipeline,
echo but runs entirely on a single device (CPU or GPU).
echo.

REM Try multiple conda paths
echo Activating MIX environment...
if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    CALL "%USERPROFILE%\anaconda3\Scripts\activate.bat" MIX
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    CALL "%USERPROFILE%\miniconda3\Scripts\activate.bat" MIX
) else if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    CALL "C:\ProgramData\anaconda3\Scripts\activate.bat" MIX
) else (
    echo WARNING: Could not find conda. Please activate MIX environment manually.
)

cd /d "%~dp0"

echo.
echo [Step 1/3] Exporting models...
python profile_fused_baseline.py --export

if errorlevel 1 (
    echo ERROR: Model export failed
    pause
    exit /b 1
)

echo.
echo [Step 2/3] Measuring CPU and GPU latencies...
echo This may take 1-2 hours depending on test cases.
echo.

python profile_fused_baseline.py --measure-cpugpu

echo.
echo [Step 3/3] Generating summary...
python profile_fused_baseline.py --analyze

echo.
echo ========================================================================
echo Profiling Complete!
echo ========================================================================
echo.
echo Results saved to: results\fused_block0_7_cpugpu.json
echo Summary saved to: results\summary.md
echo.

pause
