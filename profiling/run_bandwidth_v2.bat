@echo off
REM ========================================================================
REM OpenVINO Bandwidth Measurement V2
REM ========================================================================
REM
REM V2 Improvements:
REM   - Direct output timing via get_output_tensor().data
REM   - No indirect calculation needed
REM   - Uses Identity model with separated timing

setlocal EnableDelayedExpansion

echo ========================================================================
echo OpenVINO Bandwidth Measurement V2
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
echo Method V2:
echo   - Identity model with separated timing
echo   - Input bandwidth: start_async + wait
echo   - Output bandwidth: get_output_tensor().data (direct!)
echo.

echo ========================================================================
echo Running Bandwidth V2 Test
echo ========================================================================
echo.

python measure_bandwidth_v2.py

if errorlevel 1 (
    echo.
    echo WARNING: Test completed with some errors
) else (
    echo.
    echo Test completed successfully!
)

echo.
echo ========================================================================
echo Bandwidth V2 Test Complete!
echo ========================================================================
echo.
echo Results saved to: results\bandwidth\bandwidth_v2.json
echo Compare with V1:  results\bandwidth\bandwidth_async.json
echo.

pause
