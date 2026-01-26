@echo off
REM ========================================================================
REM OpenVINO GPU Data Transfer Time Test
REM ========================================================================
REM
REM Tests:
REM   1. set_input_tensor every iteration vs set once
REM   2. Input transfer + compute vs output transfer time
REM   3. Stage 6, Fused 1-4, Fused 1-7 comparison

setlocal EnableDelayedExpansion

echo ========================================================================
echo OpenVINO GPU Data Transfer Time Test
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
echo Test Purpose:
echo   - Verify if set_input_tensor triggers CPU-GPU data transfer
echo   - Measure input transfer + compute time separately from output transfer
echo   - Compare "set every iteration" vs "set once" timing difference
echo.

REM Check if models exist
set NEED_EXPORT=0

if not exist "exported_models\stage3_gpu.xml" set NEED_EXPORT=1
if not exist "exported_models\stage6_gpu.xml" set NEED_EXPORT=1
if not exist "..\profiling_fused\exported_models\block0_fused_gpu.xml" set NEED_EXPORT=1
if not exist "..\experiments\baseline_fused1-7_profiling\exported_models\fused_block0_7_gpu.xml" set NEED_EXPORT=1

if %NEED_EXPORT%==1 (
    echo Some models not found. Exporting models first...
    echo.
    python test_transfer_time.py --export
    if errorlevel 1 (
        echo.
        echo WARNING: Some model exports may have failed
        echo Continuing with available models...
        echo.
    )
)

echo.
echo ========================================================================
echo Running Transfer Time Test
echo ========================================================================
echo.

python test_transfer_time.py

if errorlevel 1 (
    echo.
    echo WARNING: Test completed with some errors
) else (
    echo.
    echo Test completed successfully!
)

echo.
echo ========================================================================
echo Transfer Time Test Complete!
echo ========================================================================
echo.
echo Key findings to look for:
echo   - If "compute diff" is large: set_input_tensor triggers data transfer
echo   - If "compute diff" is small: GPU may cache input data
echo   - output_ms shows GPU-to-CPU transfer time (get_output_tensor)
echo.

pause
