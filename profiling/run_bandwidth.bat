@echo off
REM ========================================================================
REM OpenVINO Bandwidth Measurement - CPU, GPU, NPU
REM ========================================================================
REM
REM Measures input/output bandwidth for each device:
REM   - Slice model: input[N] -> output[1] (input bandwidth)
REM   - Identity model: input[N] -> output[N] (combined bandwidth)
REM   - Derive output bandwidth from the two measurements

setlocal EnableDelayedExpansion

echo ========================================================================
echo OpenVINO Bandwidth Measurement
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
echo   Devices: CPU, GPU, NPU (if available)
echo   Method: Slice + Identity models
echo   CPU/GPU: Dynamic shapes (compile once)
echo   NPU: Static shapes (compile per size)
echo.

echo Starting bandwidth measurement...
echo.

python measure_bandwidth.py

if errorlevel 1 (
    echo.
    echo WARNING: Some measurements may have failed
) else (
    echo.
    echo Bandwidth measurement completed successfully!
)

echo.
echo ========================================================================
echo Bandwidth Measurement Complete!
echo ========================================================================
echo.
echo Results saved to: profiling\results\bandwidth\bandwidth_async.json
echo.

pause
