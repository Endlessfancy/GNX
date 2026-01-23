@echo off
REM ========================================================================
REM CPU + GPU Testing - Complete 1-Layer GNN Models (Baseline)
REM ========================================================================
REM
REM This script tests CPU and GPU for complete 1-layer GNN models:
REM   - GraphSAGE (SAGEConv + ReLU)
REM   - GCN (GCNConv + ReLU)
REM   - GAT (GATConv + ReLU)
REM
REM Uses dynamic shape models (one per GNN type per device).
REM Tests all 42 node×edge combinations.

setlocal EnableDelayedExpansion

echo ========================================================================
echo Baseline Profiling - CPU + GPU Testing
echo ========================================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

echo.
echo Test Configuration:
echo   Models: GraphSAGE, GCN, GAT (1 layer each)
echo   Devices: CPU, GPU
echo   Test cases: 42 (6 node sizes x 7 edge ratios)
echo.
echo Node sizes: 5k, 10k, 20k, 50k, 80k, 100k
echo Edge ratios: 10, 25, 40, 50, 60, 75, 100
echo.

REM Create results directory
if not exist "results" mkdir results

REM ========================================================================
REM Step 1: Export CPU/GPU models
REM ========================================================================
echo.
echo [Step 1/3] Checking and exporting CPU/GPU models...

if not exist "exported_models\graphsage_cpu.xml" (
    echo   Exporting CPU/GPU dynamic models...
    python profile_baseline.py --export-cpugpu
    if errorlevel 1 (
        echo ERROR: Failed to export CPU/GPU models
        pause
        exit /b 1
    )
) else (
    echo   CPU/GPU models already exist, skipping export.
)

REM ========================================================================
REM Step 2: Measure CPU and GPU
REM ========================================================================
echo.
echo [Step 2/3] Measuring CPU and GPU latencies (42 test cases x 3 models x 2 devices)...
python profile_baseline.py --measure-cpugpu
if errorlevel 1 (
    echo WARNING: Some measurements may have failed
)

echo.
echo CPU/GPU results saved to: results\cpugpu_results.json

REM ========================================================================
REM Step 3: Generate analysis
REM ========================================================================
echo.
echo [Step 3/3] Generating analysis...
python profile_baseline.py --analyze

echo.
echo ========================================================================
echo CPU + GPU Testing Complete!
echo ========================================================================
echo.
echo Results:
echo   - CPU/GPU: results\cpugpu_results.json
echo   - Summary CSV: results\baseline_latency.csv
echo.

pause
