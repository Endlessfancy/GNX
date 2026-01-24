@echo off
REM Model Verification Script
REM Verifies that exported ONNX/IR models are correct single-layer GNNs

setlocal EnableDelayedExpansion

echo ============================================================
echo Model Verification
echo ============================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

cd /d "%~dp0"

echo Running model verification...
echo.

python verify_models.py

echo.
pause
