@echo off
REM ========================================================================
REM Fused Block 0-7 Baseline - CPU Only
REM ========================================================================

setlocal EnableDelayedExpansion

echo ========================================================================
echo Fused Block 0-7 Baseline - CPU Only
echo ========================================================================
echo.

REM Try multiple conda paths
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

REM Check if model exists, export if not
if not exist "exported_models\fused_block0_7_cpu.xml" (
    echo Exporting models...
    python profile_fused_baseline.py --export
)

echo.
echo Measuring CPU latencies...
python profile_fused_baseline.py --measure-cpu

echo.
echo Generating summary...
python profile_fused_baseline.py --analyze

echo.
echo ========================================================================
echo CPU Profiling Complete!
echo ========================================================================
echo Results saved to: results\fused_block0_7_cpu.json
echo.

pause
