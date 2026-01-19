@echo off
REM ========================================================================
REM Stage 3 (REDUCE_SUM) CPU/GPU Only Test - Intel AI PC (Windows)
REM ========================================================================
REM
REM This script ONLY tests Stage 3 on CPU and GPU.
REM Stage 3 uses scatter_add which is NOT supported on NPU.
REM
REM Prerequisites:
REM   - CPU/GPU models must be exported: python profile_stages.py --export-cpugpu

echo ========================================================================
echo Stage 3 (REDUCE_SUM) CPU/GPU Only Test
echo ========================================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

echo.
echo Environment activated.
echo.
echo Stage 3 = REDUCE_SUM (scatter_add operation)
echo NPU does NOT support scatter_add, so only CPU/GPU are tested.
echo.

REM Check if CPU/GPU models exist
if not exist "exported_models\stage3_cpu.xml" (
    echo WARNING: CPU/GPU models not found!
    echo Run first: python profile_stages.py --export-cpugpu
    echo.
    set /p EXPORT_NOW="Export CPU/GPU models now? (Y/N): "
    if /i "!EXPORT_NOW!"=="Y" (
        python profile_stages.py --export-cpugpu
        if !ERRORLEVEL! NEQ 0 (
            echo ERROR: CPU/GPU model export failed!
            pause
            exit /b 1
        )
    ) else (
        echo Aborted.
        pause
        exit /b 1
    )
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
