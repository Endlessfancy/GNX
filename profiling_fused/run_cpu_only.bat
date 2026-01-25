@echo off
REM ========================================================================
REM CPU Only Testing - FusedBlock0 (Stages 1-4)
REM ========================================================================
REM
REM This script tests CPU only for Block 0:
REM   Block 0: CPU (FusedBlock0 - Stages 1-4: GATHER + REDUCE)
REM            Input:  x [num_nodes, feat], edge_index [2, num_edges]
REM            Output: sum_agg [num_nodes, feat], count [num_nodes]
REM
REM Test Configuration:
REM   - 42 test cases (6 node sizes × 7 edge ratios)
REM   - CPU uses dynamic model (handles variable sizes)

setlocal EnableDelayedExpansion

echo ========================================================================
echo CPU Only Testing - FusedBlock0 (Stages 1-4)
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
echo Block 0 Configuration:
echo   Input:  x [num_nodes, feat], edge_index [2, num_edges]
echo   Output: sum_agg [num_nodes, feat], count [num_nodes]
echo.
echo Test cases: 42 (6 node sizes x 7 edge ratios)
echo Node sizes: 5k, 10k, 20k, 50k, 80k, 100k
echo Edge ratios: 10, 25, 40, 50, 60, 75, 100
echo.

REM Create results directory
if not exist "results" mkdir results

REM ========================================================================
REM Step 1: Export CPU model (if needed)
REM ========================================================================
echo.
echo [Step 1/2] Checking CPU model...

if not exist "exported_models\block0_fused_cpu.xml" (
    echo   Exporting CPU/GPU fused model...
    python profile_pep3.py --export-cpugpu
    if errorlevel 1 (
        echo ERROR: Failed to export CPU model
        pause
        exit /b 1
    )
) else (
    echo   CPU model already exists, skipping export.
)

REM ========================================================================
REM Step 2: Measure CPU
REM ========================================================================
echo.
echo [Step 2/2] Measuring CPU latencies (42 test cases)...
echo.
python profile_pep3.py --measure-cpu

echo.
echo ========================================================================
echo CPU Testing Complete!
echo ========================================================================
echo.
echo Results saved to: results\block0_cpu.json
echo.
echo To analyze results:
echo   python profile_pep3.py --analyze
echo.

pause
