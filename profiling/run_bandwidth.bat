@echo off
REM ========================================================================
REM OpenVINO Bandwidth Measurement
REM ========================================================================
REM
REM Method:
REM   - Identity model with separated timing
REM   - Input bandwidth: start_async + wait
REM   - Output bandwidth: get_output_tensor().data (direct measurement)

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
echo Method:
echo   - Identity model with separated timing
echo   - Input bandwidth: start_async + wait
echo   - Output bandwidth: get_output_tensor().data (direct!)
echo.

echo ========================================================================
echo Running Bandwidth Test
echo ========================================================================
echo.

python measure_bandwidth.py

if errorlevel 1 (
    echo.
    echo WARNING: Test completed with some errors
) else (
    echo.
    echo Test completed successfully!
)

echo.
echo ========================================================================
echo Bandwidth Test Complete!
echo ========================================================================
echo.
echo Results saved to: results\bandwidth\bandwidth.json
echo.

pause
