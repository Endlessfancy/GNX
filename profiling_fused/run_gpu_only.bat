@echo off
REM ========================================================================
REM GPU Only Testing - FusedBlock0 (Stages 1-4)
REM ========================================================================
REM
REM This script tests GPU only for Block 0:
REM   Block 0: GPU (FusedBlock0 - Stages 1-4: GATHER + REDUCE)
REM            Input:  x [num_nodes, feat], edge_index [2, num_edges]
REM            Output: sum_agg [num_nodes, feat], count [num_nodes]
REM
REM Note: Block 0 OUTPUT does NOT include edges!
REM       Edge data is consumed internally for aggregation.
REM       Only node-based tensors are passed to Block 1 (NPU).
REM
REM Test Configuration:
REM   - 42 test cases (6 node sizes × 7 edge ratios)
REM   - GPU uses dynamic model (handles variable sizes)

setlocal EnableDelayedExpansion

echo ========================================================================
echo GPU Only Testing - FusedBlock0 (Stages 1-4)
echo ========================================================================
echo.

REM Activate conda environment
echo Activating MIX environment...
CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX

echo.
echo Block 0 Configuration:
echo   Input:  x [num_nodes, feat], edge_index [2, num_edges]
echo   Output: sum_agg [num_nodes, feat], count [num_nodes]
echo.
echo NOTE: Edge data is consumed in Block 0, NOT passed to Block 1!
echo.
echo Test cases: 42 (6 node sizes x 7 edge ratios)
echo Node sizes: 5k, 10k, 20k, 50k, 80k, 100k
echo Edge ratios: 10, 25, 40, 50, 60, 75, 100
echo.

REM Create results directory
if not exist "results" mkdir results

REM ========================================================================
REM Step 1: Export GPU model (if needed)
REM ========================================================================
echo.
echo [Step 1/2] Checking GPU model...

if not exist "exported_models\block0_fused_gpu.xml" (
    echo   Exporting CPU/GPU fused model...
    python profile_pep3.py --export-cpugpu
    if errorlevel 1 (
        echo ERROR: Failed to export GPU model
        pause
        exit /b 1
    )
) else (
    echo   GPU model already exists, skipping export.
)

REM ========================================================================
REM Step 2: Measure GPU
REM ========================================================================
echo.
echo [Step 2/2] Measuring GPU latencies (42 test cases)...
echo.
python profile_pep3.py --measure-gpu

echo.
echo ========================================================================
echo GPU Testing Complete!
echo ========================================================================
echo.
echo Results saved to: results\block0_gpu.json
echo.
echo To analyze results:
echo   python profile_pep3.py --analyze
echo.

pause
