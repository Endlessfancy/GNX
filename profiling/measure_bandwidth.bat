@echo off
REM ========================================================================
REM Bandwidth Measurement - CPU / GPU / NPU
REM ========================================================================
setlocal EnableDelayedExpansion

echo ========================================================================
echo Bandwidth Measurement - CPU / GPU / NPU
echo ========================================================================
echo.

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

echo Running bandwidth measurement...
echo.
python measure_bandwidth.py

echo.
echo ========================================================================
echo Bandwidth Measurement Complete!
echo ========================================================================
echo Results saved to: results\bandwidth\bandwidth.json
echo.

pause
