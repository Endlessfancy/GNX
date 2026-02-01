@echo off
REM ========================================================================
REM Complete Profiling - CPU + GPU + NPU (based on overlap_summary.md)
REM
REM Coverage: SAGE (7 stages) + GAT (3 unique stages) = all aten ops
REM GCN: 100%% covered by SAGE + GAT, skipped
REM
REM CPU/GPU: SAGE all 7 + GAT all 7 (stages 3/4/5 unique, others validation)
REM NPU:     SAGE 4 stages (1,5,6,7) + GAT 2 stages (3,5)
REM ========================================================================
setlocal EnableDelayedExpansion

echo ========================================================================
echo Complete Profiling - CPU + GPU + NPU
echo ========================================================================
echo.
echo Based on overlap_summary.md:
echo   CPU/GPU: SAGE (7 stages) + GAT (7 stages)
echo   NPU:     SAGE (4 stages: 1,5,6,7) + GAT (2 stages: 3,5)
echo   GCN:     SKIPPED (100%% covered)
echo.

REM --- Activate conda environment ---
if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else (
    CALL "C:\Users\29067\anaconda3\Scripts\activate.bat" MIX
)
cd /d "%~dp0"

REM ========================================================================
REM Step 1: Export all models (auto-skip existing files)
REM ========================================================================

echo.
echo ================================================================
echo [Step 1/9] Exporting SAGE CPU/GPU models (auto-skip existing)
echo ================================================================
python profile_stages.py --export-cpugpu

echo.
echo ================================================================
echo [Step 2/9] Exporting SAGE NPU static models (auto-skip existing)
echo ================================================================
python profile_stages.py --export-npu

echo.
echo ================================================================
echo [Step 3/9] Exporting GAT CPU/GPU models (auto-skip existing)
echo ================================================================
python gat_profile_stages.py --export-cpugpu

echo.
echo ================================================================
echo [Step 4/9] Exporting GAT NPU static models (auto-skip existing)
echo ================================================================
python gat_profile_stages.py --export-npu

REM ========================================================================
REM Step 5: SAGE CPU
REM ========================================================================

echo.
echo ================================================================
echo [Step 5/9] SAGE CPU Profiling (7 stages)
echo ================================================================
python profile_stages.py --measure-cpu --platform 265V

REM ========================================================================
REM Step 6: SAGE GPU
REM ========================================================================

echo.
echo ================================================================
echo [Step 6/9] SAGE GPU Profiling (7 stages)
echo ================================================================
python profile_stages.py --measure-gpu --platform 265V

REM ========================================================================
REM Step 7: GAT CPU
REM ========================================================================

echo.
echo ================================================================
echo [Step 7/9] GAT CPU Profiling (7 stages)
echo ================================================================
python gat_profile_stages.py --measure-cpu --platform 265V

REM ========================================================================
REM Step 8: GAT GPU
REM ========================================================================

echo.
echo ================================================================
echo [Step 8/9] GAT GPU Profiling (7 stages)
echo ================================================================
python gat_profile_stages.py --measure-gpu --platform 265V

REM ========================================================================
REM Step 9: NPU (SAGE Stage 1,5,6,7 + GAT Stage 3,5)
REM ========================================================================

echo.
echo ================================================================
echo [Step 9/9] NPU Profiling - SAGE (4 stages) + GAT (2 stages)
echo ================================================================

REM --- SAGE NPU: Stage 1, 5, 6, 7 ---

REM 1000 nodes
echo [1/48] SAGE Stage 1, 1000 nodes...
python profile_npu.py --nodes 1000 --stage 1
echo [2/48] SAGE Stage 5, 1000 nodes...
python profile_npu.py --nodes 1000 --stage 5
echo [3/48] SAGE Stage 6, 1000 nodes...
python profile_npu.py --nodes 1000 --stage 6
echo [4/48] SAGE Stage 7, 1000 nodes...
python profile_npu.py --nodes 1000 --stage 7

REM 2000 nodes
echo [5/48] SAGE Stage 1, 2000 nodes...
python profile_npu.py --nodes 2000 --stage 1
echo [6/48] SAGE Stage 5, 2000 nodes...
python profile_npu.py --nodes 2000 --stage 5
echo [7/48] SAGE Stage 6, 2000 nodes...
python profile_npu.py --nodes 2000 --stage 6
echo [8/48] SAGE Stage 7, 2000 nodes...
python profile_npu.py --nodes 2000 --stage 7

REM 5000 nodes
echo [9/48] SAGE Stage 1, 5000 nodes...
python profile_npu.py --nodes 5000 --stage 1
echo [10/48] SAGE Stage 5, 5000 nodes...
python profile_npu.py --nodes 5000 --stage 5
echo [11/48] SAGE Stage 6, 5000 nodes...
python profile_npu.py --nodes 5000 --stage 6
echo [12/48] SAGE Stage 7, 5000 nodes...
python profile_npu.py --nodes 5000 --stage 7

REM 10000 nodes
echo [13/48] SAGE Stage 1, 10000 nodes...
python profile_npu.py --nodes 10000 --stage 1
echo [14/48] SAGE Stage 5, 10000 nodes...
python profile_npu.py --nodes 10000 --stage 5
echo [15/48] SAGE Stage 6, 10000 nodes...
python profile_npu.py --nodes 10000 --stage 6
echo [16/48] SAGE Stage 7, 10000 nodes...
python profile_npu.py --nodes 10000 --stage 7

REM 20000 nodes
echo [17/48] SAGE Stage 1, 20000 nodes...
python profile_npu.py --nodes 20000 --stage 1
echo [18/48] SAGE Stage 5, 20000 nodes...
python profile_npu.py --nodes 20000 --stage 5
echo [19/48] SAGE Stage 6, 20000 nodes...
python profile_npu.py --nodes 20000 --stage 6
echo [20/48] SAGE Stage 7, 20000 nodes...
python profile_npu.py --nodes 20000 --stage 7

REM 50000 nodes
echo [21/48] SAGE Stage 1, 50000 nodes...
python profile_npu.py --nodes 50000 --stage 1
echo [22/48] SAGE Stage 5, 50000 nodes...
python profile_npu.py --nodes 50000 --stage 5
echo [23/48] SAGE Stage 6, 50000 nodes...
python profile_npu.py --nodes 50000 --stage 6
echo [24/48] SAGE Stage 7, 50000 nodes...
python profile_npu.py --nodes 50000 --stage 7

REM 80000 nodes
echo [25/48] SAGE Stage 1, 80000 nodes...
python profile_npu.py --nodes 80000 --stage 1
echo [26/48] SAGE Stage 5, 80000 nodes...
python profile_npu.py --nodes 80000 --stage 5
echo [27/48] SAGE Stage 6, 80000 nodes...
python profile_npu.py --nodes 80000 --stage 6
echo [28/48] SAGE Stage 7, 80000 nodes...
python profile_npu.py --nodes 80000 --stage 7

REM 100000 nodes
echo [29/48] SAGE Stage 1, 100000 nodes...
python profile_npu.py --nodes 100000 --stage 1
echo [30/48] SAGE Stage 5, 100000 nodes...
python profile_npu.py --nodes 100000 --stage 5
echo [31/48] SAGE Stage 6, 100000 nodes...
python profile_npu.py --nodes 100000 --stage 6
echo [32/48] SAGE Stage 7, 100000 nodes...
python profile_npu.py --nodes 100000 --stage 7

REM --- GAT NPU: Stage 3, 5 ---

REM 1000 nodes
echo [33/48] GAT Stage 3, 1000 nodes...
python gat_profile_npu.py --nodes 1000 --stage 3
echo [34/48] GAT Stage 5, 1000 nodes...
python gat_profile_npu.py --nodes 1000 --stage 5

REM 2000 nodes
echo [35/48] GAT Stage 3, 2000 nodes...
python gat_profile_npu.py --nodes 2000 --stage 3
echo [36/48] GAT Stage 5, 2000 nodes...
python gat_profile_npu.py --nodes 2000 --stage 5

REM 5000 nodes
echo [37/48] GAT Stage 3, 5000 nodes...
python gat_profile_npu.py --nodes 5000 --stage 3
echo [38/48] GAT Stage 5, 5000 nodes...
python gat_profile_npu.py --nodes 5000 --stage 5

REM 10000 nodes
echo [39/48] GAT Stage 3, 10000 nodes...
python gat_profile_npu.py --nodes 10000 --stage 3
echo [40/48] GAT Stage 5, 10000 nodes...
python gat_profile_npu.py --nodes 10000 --stage 5

REM 20000 nodes
echo [41/48] GAT Stage 3, 20000 nodes...
python gat_profile_npu.py --nodes 20000 --stage 3
echo [42/48] GAT Stage 5, 20000 nodes...
python gat_profile_npu.py --nodes 20000 --stage 5

REM 50000 nodes
echo [43/48] GAT Stage 3, 50000 nodes...
python gat_profile_npu.py --nodes 50000 --stage 3
echo [44/48] GAT Stage 5, 50000 nodes...
python gat_profile_npu.py --nodes 50000 --stage 5

REM 80000 nodes
echo [45/48] GAT Stage 3, 80000 nodes...
python gat_profile_npu.py --nodes 80000 --stage 3
echo [46/48] GAT Stage 5, 80000 nodes...
python gat_profile_npu.py --nodes 80000 --stage 5

REM 100000 nodes
echo [47/48] GAT Stage 3, 100000 nodes...
python gat_profile_npu.py --nodes 100000 --stage 3
echo [48/48] GAT Stage 5, 100000 nodes...
python gat_profile_npu.py --nodes 100000 --stage 5

REM ========================================================================
REM Merge NPU results
REM ========================================================================

echo.
echo ================================================================
echo Merging NPU results...
echo ================================================================
python profile_stages.py --merge-npu --platform 265V
python gat_profile_stages.py --merge-npu --platform 265V

REM ========================================================================
REM Done
REM ========================================================================

echo.
echo ========================================================================
echo All Profiling Complete!
echo ========================================================================
echo.
echo Results:
echo   SAGE CPU: results\265V\sage\checkpoint_cpu.json
echo   SAGE GPU: results\265V\sage\checkpoint_gpu.json
echo   SAGE NPU: results\265V\sage\checkpoint_npu.json
echo   GAT  CPU: results\265V\gat\checkpoint_cpu.json
echo   GAT  GPU: results\265V\gat\checkpoint_gpu.json
echo   GAT  NPU: results\265V\gat\checkpoint_npu.json
echo.
pause
