@echo off
REM ========================================================================
REM NPU Testing - Complete 1-Layer GNN Models (Baseline)
REM ========================================================================
REM
REM This script tests NPU for complete 1-layer GNN models:
REM   - GraphSAGE (SAGEConv + ReLU)
REM   - GCN (GCNConv + ReLU)
REM   - GAT (GATConv + ReLU)
REM
REM NPU requires static shape models (one per model x node x edge combination).
REM Total: 3 models x 42 test cases = 126 models
REM
REM Uses PROCESS ISOLATION to handle potential crashes gracefully.

setlocal EnableDelayedExpansion

echo ========================================================================
echo Baseline Profiling - NPU Testing
echo ========================================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

echo.
echo Test Configuration:
echo   Models: GraphSAGE, GCN, GAT (1 layer each)
echo   Device: NPU (static shape models)
echo   Total models: 126 (3 models x 42 test cases)
echo.
echo Node sizes: 5k, 10k, 20k, 50k, 80k, 100k
echo Edge ratios: 10, 25, 40, 50, 60, 75, 100
echo.
echo WARNING: This may take several hours!
echo.

REM Create results directory
if not exist "results" mkdir results

REM ========================================================================
REM Step 1: Export NPU models
REM ========================================================================
echo.
echo [Step 1/3] Checking and exporting NPU models...

if not exist "exported_models\graphsage_npu_n5000_e50000.xml" (
    echo   Exporting NPU static models (126 models)...
    python profile_baseline.py --export-npu
    if errorlevel 1 (
        echo ERROR: Failed to export NPU models
        pause
        exit /b 1
    )
) else (
    echo   NPU models already exist, skipping export.
)

REM ========================================================================
REM Step 2: Measure NPU with PROCESS ISOLATION
REM ========================================================================
echo.
echo [Step 2/3] Measuring NPU latencies with process isolation...
echo Each test case runs in a separate Python process.
echo.

set /a test_num=0
set /a total_tests=126
set /a success=0
set /a failed=0

REM Model names
set MODELS=graphsage gcn gat

REM Node sizes
set NODES=5000 10000 20000 50000 80000 100000

REM Edge configurations per node size (edges = nodes * ratio)
REM For 5000 nodes: ratios 10,25,40,50,60,75,100 -> edges 50000,125000,200000,250000,300000,375000,500000
REM For 10000 nodes: edges 100000,250000,400000,500000,600000,750000,1000000
REM etc.

for %%M in (%MODELS%) do (
    echo.
    echo === Testing %%M ===

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

REM ========================================================================
REM Step 3: Merge results and analyze
REM ========================================================================
echo.
echo [Step 3/3] Merging NPU results and generating analysis...
python merge_npu_results.py
python profile_baseline.py --analyze

echo.
echo ========================================================================
echo NPU Testing Complete!
echo ========================================================================
echo.
echo Results:
echo   - NPU: results\npu_results.json
echo   - Summary CSV: results\baseline_latency.csv
echo.

pause
