@echo off
REM ========================================================================
REM Full Baseline Profiling - Complete 1-Layer GNN Models
REM ========================================================================
REM
REM This script runs the complete baseline profiling workflow:
REM   1. Export all models (CPU/GPU dynamic + NPU static)
REM   2. Measure CPU latencies
REM   3. Measure GPU latencies
REM   4. Measure NPU latencies (with process isolation)
REM   5. Generate analysis and CSV summary
REM
REM Models tested:
REM   - GraphSAGE (SAGEConv + ReLU)
REM   - GCN (GCNConv + ReLU)
REM   - GAT (GATConv + ReLU)
REM
REM WARNING: This script may take several hours to complete!

setlocal EnableDelayedExpansion

echo ========================================================================
echo Baseline Profiling - Full Workflow
echo ========================================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

echo.
echo Test Configuration:
echo   Models: GraphSAGE, GCN, GAT (1 layer each)
echo   Devices: CPU, GPU, NPU
echo   Test cases: 42 (6 node sizes x 7 edge ratios)
echo.
echo   CPU/GPU models: 6 (3 models x 2 devices, dynamic shape)
echo   NPU models: 126 (3 models x 42 test cases, static shape)
echo.
echo   Total measurements: 378 (42 cases x 3 models x 3 devices)
echo.
echo WARNING: This may take several hours!
echo.

REM Create directories
if not exist "exported_models" mkdir exported_models
if not exist "results" mkdir results

REM ========================================================================
REM Step 1: Export all models
REM ========================================================================
echo.
echo [Step 1/5] Exporting all models...
echo.

echo   [1.1] Exporting CPU/GPU dynamic models (6 models)...
python profile_baseline.py --export-cpugpu
if errorlevel 1 (
    echo WARNING: CPU/GPU export may have failed
)

echo.
echo   [1.2] Exporting NPU static models (126 models)...
python profile_baseline.py --export-npu
if errorlevel 1 (
    echo WARNING: NPU export may have failed
)

REM ========================================================================
REM Step 2: Measure CPU
REM ========================================================================
echo.
echo [Step 2/5] Measuring CPU latencies (126 tests: 3 models x 42 cases)...
python profile_baseline.py --measure-cpu
if errorlevel 1 (
    echo WARNING: Some CPU measurements may have failed
)

REM ========================================================================
REM Step 3: Measure GPU
REM ========================================================================
echo.
echo [Step 3/5] Measuring GPU latencies (126 tests: 3 models x 42 cases)...
python profile_baseline.py --measure-gpu
if errorlevel 1 (
    echo WARNING: Some GPU measurements may have failed
)

REM ========================================================================
REM Step 4: Measure NPU with PROCESS ISOLATION
REM ========================================================================
echo.
echo [Step 4/5] Measuring NPU latencies with process isolation...
echo Each test case runs in a separate Python process.
echo.

set /a test_num=0
set /a total_tests=126
set /a success=0
set /a failed=0

REM Model names
set MODELS=graphsage gcn gat

for %%M in (%MODELS%) do (
    echo.
    echo === Testing %%M on NPU ===

    REM 5000 nodes
    for %%E in (50000 125000 200000 250000 300000 375000 500000) do (
        set /a test_num+=1
        echo   [!test_num!/%total_tests%] %%M n5000_e%%E...
        python profile_npu_isolated.py --model %%M --nodes 5000 --edges %%E
        if errorlevel 1 (set /a failed+=1) else (set /a success+=1)
    )

    REM 10000 nodes
    for %%E in (100000 250000 400000 500000 600000 750000 1000000) do (
        set /a test_num+=1
        echo   [!test_num!/%total_tests%] %%M n10000_e%%E...
        python profile_npu_isolated.py --model %%M --nodes 10000 --edges %%E
        if errorlevel 1 (set /a failed+=1) else (set /a success+=1)
    )

    REM 20000 nodes
    for %%E in (200000 500000 800000 1000000 1200000 1500000 2000000) do (
        set /a test_num+=1
        echo   [!test_num!/%total_tests%] %%M n20000_e%%E...
        python profile_npu_isolated.py --model %%M --nodes 20000 --edges %%E
        if errorlevel 1 (set /a failed+=1) else (set /a success+=1)
    )

    REM 50000 nodes
    for %%E in (500000 1250000 2000000 2500000 3000000 3750000 5000000) do (
        set /a test_num+=1
        echo   [!test_num!/%total_tests%] %%M n50000_e%%E...
        python profile_npu_isolated.py --model %%M --nodes 50000 --edges %%E
        if errorlevel 1 (set /a failed+=1) else (set /a success+=1)
    )

    REM 80000 nodes
    for %%E in (800000 2000000 3200000 4000000 4800000 6000000 8000000) do (
        set /a test_num+=1
        echo   [!test_num!/%total_tests%] %%M n80000_e%%E...
        python profile_npu_isolated.py --model %%M --nodes 80000 --edges %%E
        if errorlevel 1 (set /a failed+=1) else (set /a success+=1)
    )

    REM 100000 nodes
    for %%E in (1000000 2500000 4000000 5000000 6000000 7500000 10000000) do (
        set /a test_num+=1
        echo   [!test_num!/%total_tests%] %%M n100000_e%%E...
        python profile_npu_isolated.py --model %%M --nodes 100000 --edges %%E
        if errorlevel 1 (set /a failed+=1) else (set /a success+=1)
    )
)

echo.
echo NPU Testing Summary: !success! success, !failed! failed out of %total_tests% tests

echo.
echo Merging NPU results...
python merge_npu_results.py

REM ========================================================================
REM Step 5: Generate analysis
REM ========================================================================
echo.
echo [Step 5/5] Generating analysis and summary...
python profile_baseline.py --analyze

echo.
echo ========================================================================
echo Baseline Profiling Complete!
echo ========================================================================
echo.
echo Results:
echo   - CPU:  results\cpu_results.json
echo   - GPU:  results\gpu_results.json
echo   - NPU:  results\npu_results.json
echo   - Summary CSV: results\baseline_latency.csv
echo.
echo Models tested: GraphSAGE, GCN, GAT (1 layer each)
echo Test cases: 42 (6 node sizes x 7 edge ratios)
echo Devices: CPU, GPU, NPU
echo.

pause
