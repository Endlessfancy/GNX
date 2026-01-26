@echo off
REM ========================================================================
REM NPU Only Profiling - Intel AI PC (Windows)
REM ========================================================================
REM
REM This script ONLY runs NPU tests (skips CPU/GPU).
REM Use this when you already have CPU/GPU results and just need to re-run NPU.
REM
REM Prerequisites:
REM   - NPU models must be exported: python profile_stages.py --export-npu
REM   - Or run full export first: python profile_stages.py --export

setlocal EnableDelayedExpansion

echo ========================================================================
echo NPU Only Profiling - Intel AI PC
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
echo Environment activated.
echo.
echo NPU stages: 1, 2, 5, 6, 7 (skip Stage 3/4 - no scatter_add support)
echo NPU tests: 8 node sizes x 5 stages = 40 isolated processes
echo.

REM Check and export NPU models if needed
if not exist "exported_models\stage1_npu_n1000_e2000.xml" (
    echo NPU models not found. Exporting...
    echo.
    python profile_stages.py --export-npu
    if !ERRORLEVEL! NEQ 0 (
        echo ERROR: NPU model export failed!
        pause
        exit /b 1
    )
    echo.
    echo Export complete. Continuing to NPU tests...
) else (
    echo NPU models found. Skipping export.
)
echo.

REM ========================================================================
REM NPU Testing (isolated per nodes/stage)
REM ========================================================================
echo.
echo ========================================================================
echo Running NPU Tests (Isolated Processes)
echo ========================================================================
echo.
echo Each (nodes, stage) combination runs in its own Python process.
echo If DEVICE_LOST occurs, that test fails but others continue.
echo.

REM Define node sizes and stages
set NODE_SIZES=1000 2000 5000 10000 20000 50000 80000 100000
set NPU_STAGES=1 2 5 6 7

REM Count total tests
set /a TOTAL_TESTS=0
for %%n in (%NODE_SIZES%) do (
    for %%s in (%NPU_STAGES%) do (
        set /a TOTAL_TESTS+=1
    )
)
echo Total NPU tests: %TOTAL_TESTS% (8 node sizes x 5 stages)
echo.

REM Track progress
set /a CURRENT_TEST=0
set /a PASSED=0
set /a FAILED=0
set /a DEVICE_LOST=0

REM Run each (nodes, stage) combination
for %%n in (%NODE_SIZES%) do (
    for %%s in (%NPU_STAGES%) do (
        set /a CURRENT_TEST+=1
        echo.
        echo [!CURRENT_TEST!/%TOTAL_TESTS%] Testing Stage %%s with %%n nodes...

        python profile_npu.py --nodes %%n --stage %%s
        set EXIT_CODE=!ERRORLEVEL!

        if !EXIT_CODE! EQU 0 (
            echo   Result: PASSED
            set /a PASSED+=1
        ) else if !EXIT_CODE! EQU 1 (
            echo   Result: PARTIAL FAILURE (some edges failed)
            set /a FAILED+=1
        ) else if !EXIT_CODE! EQU 2 (
            echo   Result: DEVICE_LOST - NPU in bad state
            set /a DEVICE_LOST+=1
            echo   Continuing to next test...
        ) else (
            echo   Result: UNKNOWN ERROR (code=!EXIT_CODE!)
            set /a FAILED+=1
        )
    )
)

echo.
echo ========================================================================
echo NPU Testing Summary
echo ========================================================================
echo   Passed:      %PASSED%
echo   Failed:      %FAILED%
echo   Device Lost: %DEVICE_LOST%
echo   Total:       %TOTAL_TESTS%
echo.

REM ========================================================================
REM Merge NPU results
REM ========================================================================
echo ========================================================================
echo Merging NPU Results
echo ========================================================================
echo.

python profile_stages.py --merge-npu
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: NPU merge had issues, but continuing...
)

echo.
echo ========================================================================
echo NPU Profiling Complete!
echo ========================================================================
echo.
echo Results saved in: profiling\results\sage\
echo   - checkpoint_npu.json    (merged NPU data)
echo   - npu_stage*_n*.json     (individual NPU test results)
echo.
echo Next steps:
echo   1. Run analysis: python profile_stages.py --analyze
echo   2. Or run full workflow: run_profiling.bat
echo.
echo NPU Test Summary:
echo   Passed: %PASSED% / %TOTAL_TESTS%
echo   Failed: %FAILED%
echo   Device Lost: %DEVICE_LOST%
echo.

pause
