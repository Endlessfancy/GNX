@echo off
echo ========================================================================
echo GAT NPU Debug Script v2
echo ========================================================================
echo.

echo [Step 1] Testing without conda first...
echo Current directory: %CD%
cd /d "%~dp0"
echo After cd: %CD%
echo.
pause

echo.
echo [Step 2] Now activating conda...
CALL C:\Env\Anaconda\Scripts\activate.bat MIX
echo Conda activated, continuing...
echo.
pause

echo.
echo [Step 3] Checking Python...
where python
python --version
echo.
pause

echo.
echo [Step 4] Checking OpenVINO...
python -c "import openvino; print('OK:', openvino.__version__)"
echo.
pause

echo.
echo [Step 5] Checking NPU models...
if exist "gat_exported_models\stage1_npu_n1000_e2000.xml" (
    echo FOUND
) else (
    echo NOT FOUND - will export
    python gat_profile_stages.py --export-npu
)
echo.
pause

echo.
echo [Step 6] Running NPU test...
python gat_profile_npu.py --nodes 1000 --stage 1
echo.

echo Done!
pause
