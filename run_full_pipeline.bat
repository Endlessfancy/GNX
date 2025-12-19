@echo off
REM ################################################################################
REM GNN Complete Pipeline - Full Workflow (Windows Version)
REM 完整流程：编译 → 模型导出 → 执行推理
REM ################################################################################

setlocal enabledelayedexpansion

REM Get script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Record start time
set PIPELINE_START=%time%

echo ################################################################################
echo #                                                                              #
echo #        GNN Complete Pipeline - Compiler -^> Executor -^> Verification          #
echo #                                                                              #
echo ################################################################################
echo.

REM ################################################################################
REM Phase 1: Compiler - Graph Partitioning and Optimization
REM ################################################################################

echo [Phase 1/3] Running Compiler...
echo   - Graph partitioning with METIS
echo   - PEP generation and optimization
echo   - Execution plan generation
echo.

REM Record compiler start time
set COMPILER_START=%time%

cd compiler

REM Clean old outputs
echo   Cleaning old compilation results...
if exist output\compilation_result.json del /f /q output\compilation_result.json
if exist output\models\*.onnx del /f /q output\models\*.onnx 2>nul
if exist output\models\*.ir del /f /q output\models\*.ir 2>nul

echo   Running compiler...
python test_compiler_flickr.py > %TEMP%\compiler_output.log 2>&1

if errorlevel 1 (
    echo   [ERROR] Compiler failed!
    echo   See log: %TEMP%\compiler_output.log
    type %TEMP%\compiler_output.log
    exit /b 1
)

set COMPILER_END=%time%

REM Verify compilation result exists
if not exist output\compilation_result.json (
    echo   [ERROR] Compilation result not found!
    exit /b 1
)

echo   [OK] Compiler completed
echo.

REM Extract compiler statistics using Python
for /f "delims=" %%i in ('python -c "import json; f=open('output/compilation_result.json'); d=json.load(f); print(f\"{d['statistics']['makespan']:.2f}\")"') do set ESTIMATED_MAKESPAN=%%i
for /f "delims=" %%i in ('python -c "import json; f=open('output/compilation_result.json'); d=json.load(f); print(d['partition_config']['num_subgraphs'])"') do set NUM_SUBGRAPHS=%%i
for /f "delims=" %%i in ('python -c "import json; f=open('output/compilation_result.json'); d=json.load(f); print(d['statistics']['num_unique_models'])"') do set NUM_MODELS=%%i

echo   Compilation Summary:
echo     - Subgraphs: %NUM_SUBGRAPHS%
echo     - Unique models: %NUM_MODELS%
echo     - Estimated makespan: %ESTIMATED_MAKESPAN%ms
echo.

cd ..

REM ################################################################################
REM Phase 2: Model Export (handled by executor automatically)
REM ################################################################################

echo [Phase 2/3] Model Export...
echo   - Model export will be handled by executor automatically
echo   - Placeholder models will be replaced with real ONNX models
echo.

REM ################################################################################
REM Phase 3: Executor - Pipeline Inference
REM ################################################################################

echo [Phase 3/3] Running Executor...
echo   - Loading graph data and partitions
echo   - Collecting ghost node features
echo   - Exporting real ONNX/IR models (if needed)
echo   - Executing inference on all subgraphs
echo.

set EXECUTOR_START=%time%

cd executer

python test_executor.py > %TEMP%\executor_output.log 2>&1

if errorlevel 1 (
    echo   [ERROR] Executor failed!
    echo   See log: %TEMP%\executor_output.log
    type %TEMP%\executor_output.log
    exit /b 1
)

set EXECUTOR_END=%time%

echo   [OK] Executor completed
echo.

cd ..

REM ################################################################################
REM Results Summary
REM ################################################################################

set PIPELINE_END=%time%

echo ################################################################################
echo #                                                                              #
echo #                          PIPELINE SUMMARY                                   #
echo #                                                                              #
echo ################################################################################
echo.

echo Execution Time Breakdown:
echo   +-------------------------------------------------------------+
echo   ^| Phase 1: Compiler                                          ^|
echo   ^| Phase 2: Model Export                (auto)                ^|
echo   ^| Phase 3: Executor                                          ^|
echo   +-------------------------------------------------------------+
echo.

REM Extract actual inference time from executor log
for /f "tokens=3" %%i in ('findstr /C:"Actual latency:" %TEMP%\executor_output.log 2^>nul') do set ACTUAL_LATENCY=%%i
for /f "tokens=3" %%i in ('findstr /C:"Estimation error:" %TEMP%\executor_output.log 2^>nul') do set ERROR_PCT=%%i

if not defined ACTUAL_LATENCY set ACTUAL_LATENCY=N/A
if not defined ERROR_PCT set ERROR_PCT=N/A

echo Performance Results:
echo   +-------------------------------------------------------------+
echo   ^| Compiler Estimated Makespan:         %ESTIMATED_MAKESPAN%ms              ^|
echo   ^| Actual Measured Latency:             %ACTUAL_LATENCY%              ^|
echo   ^| Estimation Error:                    %ERROR_PCT%              ^|
echo   +-------------------------------------------------------------+
echo.

echo Output Files:
echo   - Compilation result: compiler\output\compilation_result.json
echo   - ONNX models: compiler\output\models\*.onnx
echo   - Compiler log: %TEMP%\compiler_output.log
echo   - Executor log: %TEMP%\executor_output.log
echo.

REM Check if estimation is accurate
if "%ERROR_PCT%" NEQ "N/A" (
    echo [OK] Estimation validation completed
)

echo.
echo ################################################################################
echo Pipeline completed successfully!
echo ################################################################################
echo.

REM Save summary to file
set SUMMARY_FILE=pipeline_summary.txt
(
echo GNN Pipeline Execution Summary
echo Generated: %date% %time%
echo ================================================================================
echo.
echo TIMING BREAKDOWN
echo ----------------
echo Pipeline Start: %PIPELINE_START%
echo Pipeline End:   %PIPELINE_END%
echo.
echo PERFORMANCE METRICS
echo -------------------
echo Estimated Makespan:       %ESTIMATED_MAKESPAN%ms
echo Actual Latency:           %ACTUAL_LATENCY%
echo Estimation Error:         %ERROR_PCT%
echo.
echo CONFIGURATION
echo -------------
echo Dataset:                  Flickr (89,250 nodes, 899,756 edges^)
echo Subgraphs:                %NUM_SUBGRAPHS%
echo Unique Models:            %NUM_MODELS%
echo.
echo OUTPUT FILES
echo ------------
echo - compiler\output\compilation_result.json
echo - compiler\output\models\*.onnx
echo - %TEMP%\compiler_output.log
echo - %TEMP%\executor_output.log
echo.
echo ================================================================================
) > %SUMMARY_FILE%

echo Summary saved to: %SUMMARY_FILE%
echo.

pause
