@echo off
REM ========================================================================
REM GAT NPU Debug Script - Step by step with pauses
REM ========================================================================
REM
REM Use this script to debug NPU issues. It pauses after each step.

setlocal EnableDelayedExpansion

echo ========================================================================
echo GAT NPU Debug Script
echo ========================================================================
echo.

REM Try multiple conda paths
echo [Step 1] Activating MIX environment...
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
echo Current directory: %CD%
echo.
echo Checking Python...
where python
echo.
pause

REM ========================================================================
echo.
echo [Step 2] Checking OpenVINO and available devices...
echo Running: python -c "import openvino..."
echo ========================================================================
python -c "import openvino as ov; core = ov.Core(); devs = core.available_devices; print('Available devices:', devs); print('NPU available:', 'NPU' in devs)"
if errorlevel 1 (
    echo ERROR: OpenVINO import failed!
    pause
    exit /b 1
)
echo.
pause

REM ========================================================================
echo.
echo [Step 3] Checking if GAT NPU models exist...
echo ========================================================================
echo Looking for: gat_exported_models\stage1_npu_n1000_e2000.xml
if exist "gat_exported_models\stage1_npu_n1000_e2000.xml" (
    echo FOUND: NPU models exist
    dir gat_exported_models\*npu*.xml /b | find /c ".xml"
) else (
    echo NOT FOUND: NPU models need to be exported
    echo.
    echo Running: python gat_profile_stages.py --export-npu
    python gat_profile_stages.py --export-npu
    echo.
    echo Export exit code: !ERRORLEVEL!
    if !ERRORLEVEL! NEQ 0 (
        echo ERROR: NPU model export failed!
        pause
        exit /b 1
    )
)
echo.
pause

REM ========================================================================
echo.
echo [Step 4] Testing single NPU inference (Stage 1, 1000 nodes)...
echo ========================================================================
echo Running: python gat_profile_npu.py --nodes 1000 --stage 1
echo.
python gat_profile_npu.py --nodes 1000 --stage 1
set NPU_EXIT_CODE=!ERRORLEVEL!
echo.
echo Exit code: %NPU_EXIT_CODE%
if %NPU_EXIT_CODE% EQU 0 (
    echo Result: SUCCESS
) else if %NPU_EXIT_CODE% EQU 1 (
    echo Result: PARTIAL FAILURE (some edges failed)
) else if %NPU_EXIT_CODE% EQU 2 (
    echo Result: DEVICE_LOST - NPU in bad state
) else (
    echo Result: UNKNOWN ERROR
)
echo.
pause

REM ========================================================================
echo.
echo [Step 5] Testing Stage 3 (GAT-specific, 1000 nodes)...
echo ========================================================================
echo Running: python gat_profile_npu.py --nodes 1000 --stage 3
echo.
python gat_profile_npu.py --nodes 1000 --stage 3
set NPU_EXIT_CODE=!ERRORLEVEL!
echo.
echo Exit code: %NPU_EXIT_CODE%
echo.
pause

REM ========================================================================
echo.
echo [Step 6] Checking results directory...
echo ========================================================================
if exist "gat_results" (
    echo GAT results directory exists:
    dir gat_results\*.json /b 2>nul
) else (
    echo GAT results directory does not exist yet
)
echo.
pause

echo.
echo ========================================================================
echo Debug complete!
echo ========================================================================
echo.
echo If all steps passed, you can run the full NPU test:
echo   run_gat_npu_only.bat
echo.

pause
