@echo off
REM ========================================================================
REM Complete CPU Profiling (based on overlap_summary.md)
REM
REM Coverage: SAGE (7 stages) + GAT (7 stages) = all aten ops for SAGE/GAT/GCN
REM GCN profiling is NOT needed (100%% covered by SAGE + GAT)
REM ========================================================================
setlocal EnableDelayedExpansion

echo ========================================================================
echo Complete CPU Profiling - SAGE + GAT
echo ========================================================================
echo.
echo Based on overlap_summary.md:
echo   SAGE: 7 stages (all)
echo   GAT:  7 stages (stages 3/4/5 are unique, others for validation)
echo   GCN:  SKIPPED (100%% covered by SAGE + GAT)
echo.

REM --- Activate conda environment ---
if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else (
    CALL "C:\Users\29067\anaconda3\Scripts\activate.bat" MIX
)
cd /d "%~dp0"

REM ========================================================================
REM Step 1: Export models (if not already exported)
REM ========================================================================

echo.
echo [Step 1/3] Checking and exporting models...
echo ----------------------------------------------------------------

if not exist "exported_models\stage1_cpu.xml" (
    echo Exporting SAGE CPU/GPU models...
    python profile_stages.py --export-cpugpu
) else (
    echo SAGE models already exported, skipping.
)

if not exist "gat_exported_models\stage1_cpu.xml" (
    echo Exporting GAT CPU/GPU models...
    python gat_profile_stages.py --export-cpugpu
) else (
    echo GAT models already exported, skipping.
)

REM ========================================================================
REM Step 2: SAGE CPU Profiling (7 stages)
REM ========================================================================

echo.
echo [Step 2/3] SAGE CPU Profiling (7 stages)
echo ----------------------------------------------------------------
python profile_stages.py --measure-cpu --platform 265V

REM ========================================================================
REM Step 3: GAT CPU Profiling (7 stages)
REM ========================================================================

echo.
echo [Step 3/3] GAT CPU Profiling (7 stages)
echo ----------------------------------------------------------------
python gat_profile_stages.py --measure-cpu --platform 265V

REM ========================================================================
REM Done
REM ========================================================================

echo.
echo ========================================================================
echo CPU Profiling Complete!
echo ========================================================================
echo.
echo Results:
echo   SAGE: results\265V\sage\checkpoint_cpu.json
echo   GAT:  results\265V\gat\checkpoint_cpu.json
echo.
pause
