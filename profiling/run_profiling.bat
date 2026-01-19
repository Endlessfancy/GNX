@echo off
REM ========================================================================
REM GNN Stage Profiling - Intel AI PC (Windows)
REM ========================================================================
REM
REM New architecture: NPU tests run in isolated processes per (nodes, stage)
REM This prevents DEVICE_LOST errors from cascading and allows finding NPU limits.
REM
REM Workflow:
REM   Phase 1: Export all models
REM   Phase 2: Measure CPU/GPU latencies
REM   Phase 3: Measure NPU latencies (isolated per nodes/stage)
REM   Phase 4: Merge results and analyze

setlocal EnableDelayedExpansion

echo ========================================================================
echo GNN Stage Profiling - Intel AI PC
echo ========================================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

echo.
echo Environment activated.
echo.
echo Test cases: 55 combinations (1k-150k nodes, 5 edge ratios each)
echo NPU stages: 1, 2, 5, 6, 7 (skip Stage 3/4 - no scatter_add support)
echo NPU tests: 11 node sizes x 5 stages = 55 isolated processes
echo.

REM ========================================================================
REM PHASE 1: Export all models
REM ========================================================================
echo ========================================================================
echo PHASE 1: Exporting All Models
echo ========================================================================
echo.

python profile_stages.py --export
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Model export failed!
    pause
    exit /b 1
)

echo.
echo Phase 1 complete: All models exported.
echo.

REM ========================================================================
REM PHASE 2: Measure CPU/GPU latencies
REM ========================================================================
echo ========================================================================
echo PHASE 2: Measuring CPU/GPU Latencies
echo ========================================================================
echo.

python profile_stages.py --measure
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CPU/GPU measurement failed!
    pause
    exit /b 1
)

echo.
echo Phase 2 complete: CPU/GPU measurements saved.
echo.

REM ========================================================================
REM PHASE 3: Measure NPU latencies (isolated per nodes/stage)
REM ========================================================================
echo ========================================================================
echo PHASE 3: Measuring NPU Latencies (Isolated Processes)
echo ========================================================================
echo.
echo Each (nodes, stage) combination runs in its own Python process.
echo If DEVICE_LOST occurs, that test fails but others continue.
echo.

REM Define node sizes and stages
set NODE_SIZES=1000 5000 10000 20000 30000 40000 50000 60000 80000 100000 150000
set NPU_STAGES=1 2 5 6 7

REM Count total tests
set /a TOTAL_TESTS=0
for %%n in (%NODE_SIZES%) do (
    for %%s in (%NPU_STAGES%) do (
        set /a TOTAL_TESTS+=1
    )
)
echo Total NPU tests: %TOTAL_TESTS% (11 node sizes x 5 stages)
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
echo Phase 3 Summary
echo ========================================================================
echo   Passed:      %PASSED%
echo   Failed:      %FAILED%
echo   Device Lost: %DEVICE_LOST%
echo   Total:       %TOTAL_TESTS%
echo.

REM ========================================================================
REM PHASE 4: Merge NPU results and analyze
REM ========================================================================
echo ========================================================================
echo PHASE 4: Merging Results and Analyzing
echo ========================================================================
echo.

echo Merging NPU checkpoint files...
python profile_stages.py --merge-npu
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: NPU merge had issues, but continuing...
)

echo.
echo Generating final analysis...
python profile_stages.py --analyze
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Analysis failed!
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Profiling Complete!
echo ========================================================================
echo.
echo Results saved in: profiling\results\
echo   - lookup_table.json      (compute times)
echo   - bandwidth_table.json   (bandwidth estimates)
echo   - profiling_report.txt   (summary report)
echo.
echo Checkpoints:
echo   - checkpoint_cpugpu.json (CPU/GPU data)
echo   - checkpoint_npu.json    (merged NPU data)
echo   - npu_stage*_n*.json     (individual NPU test results)
echo.
echo NPU Test Summary:
echo   Passed: %PASSED% / %TOTAL_TESTS%
echo   Failed: %FAILED%
echo   Device Lost: %DEVICE_LOST%
echo.

pause
