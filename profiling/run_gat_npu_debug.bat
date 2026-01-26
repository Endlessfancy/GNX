@echo off
setlocal EnableDelayedExpansion

echo ========================================================================
echo GAT NPU Debug Script
echo ========================================================================
echo.

echo [Step 1] Activating MIX environment...
if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    CALL "%USERPROFILE%\anaconda3\Scripts\activate.bat" MIX
) else (
    echo WARNING: Could not find conda.
)

cd /d "%~dp0"
echo Current directory: %CD%
echo.
echo Checking Python...
where python
echo.
echo Press any key to continue to Step 2...
pause

echo.
echo ========================================================================
echo [Step 2] Checking OpenVINO...
echo ========================================================================
python -c "import openvino; print('OpenVINO version:', openvino.__version__)"
echo Exit code: %ERRORLEVEL%
echo.
echo Press any key to continue to Step 3...
pause

echo.
echo ========================================================================
echo [Step 3] Checking available devices...
echo ========================================================================
python -c "import openvino as ov; print(ov.Core().available_devices)"
echo Exit code: %ERRORLEVEL%
echo.
echo Press any key to continue to Step 4...
pause

echo.
echo ========================================================================
echo [Step 4] Checking GAT NPU models...
echo ========================================================================
if exist "gat_exported_models\stage1_npu_n1000_e2000.xml" (
    echo FOUND: NPU models exist
    dir gat_exported_models\*npu*.xml /b 2>nul | find /c ".xml"
) else (
    echo NOT FOUND: Need to export
    echo Running export...
    python gat_profile_stages.py --export-npu
    echo Export exit code: %ERRORLEVEL%
)
echo.
echo Press any key to continue to Step 5...
pause

echo.
echo ========================================================================
echo [Step 5] Testing single NPU inference...
echo ========================================================================
echo Running: python gat_profile_npu.py --nodes 1000 --stage 1
python gat_profile_npu.py --nodes 1000 --stage 1
echo Exit code: %ERRORLEVEL%
echo.

echo ========================================================================
echo Debug complete!
echo ========================================================================
pause
