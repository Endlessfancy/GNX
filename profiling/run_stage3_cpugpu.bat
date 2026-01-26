@echo off
REM ========================================================================
REM Stage 3 (REDUCE_SUM) CPU/GPU Only Test - Intel AI PC (Windows)
REM ========================================================================
REM
REM This script ONLY tests Stage 3 on CPU and GPU.
REM Stage 3 uses scatter_add which is NOT supported on NPU.

setlocal EnableDelayedExpansion

echo ========================================================================
echo Stage 3 (REDUCE_SUM) CPU/GPU Only Test
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
echo Stage 3 = REDUCE_SUM (scatter_add operation)
echo NPU does NOT support scatter_add, so only CPU/GPU are tested.
echo.

REM Check and export CPU/GPU models if needed
if not exist "exported_models\stage3_cpu.xml" (
    echo CPU/GPU models not found. Exporting...
    python profile_stages.py --export-cpugpu
    if !ERRORLEVEL! NEQ 0 (
        echo ERROR: CPU/GPU model export failed!
        pause
        exit /b 1
    )
) else (
    echo CPU/GPU models found. Skipping export.
)

echo ========================================================================
echo Running Stage 3 Tests on CPU and GPU
echo ========================================================================
echo.

python profile_stage3_cpugpu.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================================
    echo All Stage 3 tests PASSED!
    echo ========================================================================
) else (
    echo.
    echo ========================================================================
    echo Some Stage 3 tests FAILED!
    echo ========================================================================
)

echo.
echo Results saved to: profiling\results\stage3_cpugpu.json
echo.

pause
