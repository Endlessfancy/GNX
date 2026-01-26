@echo off
REM ========================================================================
REM GNN Stage Profiling - Full Test (CPU + GPU + NPU)
REM ========================================================================
REM
REM This script runs complete profiling on all devices:
REM   - CPU: All 7 stages
REM   - GPU: All 7 stages
REM   - NPU: Stages 1, 2, 5, 6, 7 (skip Stage 3/4 - no scatter_add support)
REM
REM Workflow:
REM   Phase 1: Export all models (CPU, GPU, NPU)
REM   Phase 2: Measure CPU latencies
REM   Phase 3: Measure GPU latencies
REM   Phase 4: Measure NPU latencies (isolated per nodes/stage)
REM   Phase 5: Merge results and analyze

setlocal EnableDelayedExpansion

REM ========================================================================
REM PLATFORM CONFIGURATION - Modify this for different AI PCs
REM ========================================================================
set PLATFORM=185H
REM Options: 185H, 265V, or leave empty for default location

echo ========================================================================
echo GNN Stage Profiling - Full Test (CPU + GPU + NPU)
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
echo Test Configuration:
echo   Platform: %PLATFORM%
echo   Stages: 1-7 (GATHER, MESSAGE, REDUCE_SUM, REDUCE_COUNT, NORMALIZE, TRANSFORM, ACTIVATE)
echo   CPU: All 7 stages
echo   GPU: All 7 stages
echo   NPU: Stages 1, 2, 5, 6, 7 (55 isolated processes)
echo.

REM Build platform argument
if "%PLATFORM%"=="" (
    set PLATFORM_ARG=
) else (
    set PLATFORM_ARG=--platform %PLATFORM%
)

REM ========================================================================
REM PHASE 1: Export all models
REM ========================================================================
echo ========================================================================
echo PHASE 1: Exporting All Models (CPU + GPU + NPU)
echo ========================================================================
echo.

REM Check if CPU/GPU models exist
if not exist "exported_models\stage1_cpu.xml" (
    echo CPU/GPU models not found. Exporting...
    python profile_stages.py --export-cpugpu
    if errorlevel 1 (
        echo ERROR: CPU/GPU model export failed!
        pause
        exit /b 1
    )
) else (
    echo CPU/GPU models already exist. Skipping export.
)

echo.

REM Check if NPU models exist
if not exist "exported_models\stage1_npu_n1000_e2000.xml" (
    echo NPU models not found. Exporting...
    python profile_stages.py --export-npu
    if errorlevel 1 (
        echo ERROR: NPU model export failed!
        pause
        exit /b 1
    )
) else (
    echo NPU models already exist. Skipping export.
)

echo.
echo Phase 1 complete: All models ready.
echo.

REM ========================================================================
REM PHASE 2: Measure CPU latencies
REM ========================================================================
echo ========================================================================
echo PHASE 2: Measuring CPU Latencies (All 7 Stages)
echo ========================================================================
echo.

python profile_stages.py --measure-cpu %PLATFORM_ARG%
if errorlevel 1 (
    echo.
    echo WARNING: Some CPU measurements may have failed
) else (
    echo.
    echo CPU measurement completed successfully!
)

echo.
echo Phase 2 complete: CPU measurements saved.
echo.

REM ========================================================================
REM PHASE 3: Measure GPU latencies
REM ========================================================================
echo ========================================================================
echo PHASE 3: Measuring GPU Latencies (All 7 Stages)
echo ========================================================================
echo.

python profile_stages.py --measure-gpu %PLATFORM_ARG%
if errorlevel 1 (
    echo.
    echo WARNING: Some GPU measurements may have failed
) else (
    echo.
    echo GPU measurement completed successfully!
)

echo.
echo Phase 3 complete: GPU measurements saved.
echo.

REM ========================================================================
REM PHASE 4: Measure NPU latencies (isolated per nodes/stage)
REM ========================================================================
echo ========================================================================
echo PHASE 4: Measuring NPU Latencies (Isolated Processes)
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
echo Phase 4 Summary (NPU)
echo ========================================================================
echo   Passed:      %PASSED%
echo   Failed:      %FAILED%
echo   Device Lost: %DEVICE_LOST%
echo   Total:       %TOTAL_TESTS%
echo.

REM ========================================================================
REM PHASE 5: Merge results and analyze
REM ========================================================================
echo ========================================================================
echo PHASE 5: Merging Results and Analyzing
echo ========================================================================
echo.

echo Merging NPU checkpoint files...
python profile_stages.py --merge-npu %PLATFORM_ARG%
if errorlevel 1 (
    echo WARNING: NPU merge had issues, but continuing...
)

echo.
echo Generating final analysis...
python profile_stages.py --analyze %PLATFORM_ARG%
if errorlevel 1 (
    echo ERROR: Analysis failed!
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Full Profiling Complete!
echo ========================================================================
echo.
echo Results saved in: profiling\results\sage\
echo   - checkpoint_cpu.json     (CPU data)
echo   - checkpoint_gpu.json     (GPU data)
echo   - checkpoint_npu.json     (merged NPU data)
echo   - lookup_table.json       (compute times)
echo   - bandwidth_table.json    (bandwidth estimates)
echo   - profiling_report.txt    (summary report)
echo.
echo NPU Test Summary:
echo   Passed: %PASSED% / %TOTAL_TESTS%
echo   Failed: %FAILED%
echo   Device Lost: %DEVICE_LOST%
echo.

pause
