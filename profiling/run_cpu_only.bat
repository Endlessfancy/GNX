@echo off
REM ========================================================================
REM GNN Stage Profiling - CPU Only (All 7 Stages)
REM ========================================================================
REM
REM Tests all 7 stages on CPU only:
REM   Stage 1: GATHER
REM   Stage 2: MESSAGE
REM   Stage 3: REDUCE_SUM
REM   Stage 4: REDUCE_COUNT
REM   Stage 5: NORMALIZE
REM   Stage 6: TRANSFORM
REM   Stage 7: ACTIVATE

setlocal EnableDelayedExpansion

echo ========================================================================
echo GNN Stage Profiling - CPU Only (All 7 Stages)
echo ========================================================================
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
echo Test Configuration:
echo   Stages: 1-7 (GATHER, MESSAGE, REDUCE_SUM, REDUCE_COUNT, NORMALIZE, TRANSFORM, ACTIVATE)
echo   Device: CPU only
echo.

REM Check if CPU models exist, export if not
if not exist "exported_models\stage1_cpu.xml" (
    echo CPU models not found. Exporting...
    python profile_stages.py --export-cpugpu
    if errorlevel 1 (
        echo ERROR: Failed to export models
        pause
        exit /b 1
    )
)

echo.
echo Starting CPU measurement...
python profile_stages.py --measure-cpu

if errorlevel 1 (
    echo.
    echo WARNING: Some CPU measurements may have failed
) else (
    echo.
    echo CPU measurement completed successfully!
)

echo.
echo ========================================================================
echo CPU Profiling Complete!
echo ========================================================================
echo.
echo Results saved to: profiling\results\checkpoint_cpu.json
echo.

pause
