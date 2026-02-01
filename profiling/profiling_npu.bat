@echo off
REM ========================================================================
REM Complete NPU Profiling (based on overlap_summary.md)
REM
REM SAGE NPU: Stage 1 (Gather), 5 (Normalize), 6 (Transform), 7 (Activate)
REM GAT  NPU: Stage 3 (AttentionScore), 5 (MessageWeighted)
REM Total: 6 stages x 8 node sizes = 48 measurements
REM
REM Scatter stages excluded (must stay on CPU):
REM   SAGE 3/4, GAT 4/6, GCN 1/4
REM GCN: all NPU-compatible stages covered by SAGE + GAT, skipped
REM ========================================================================
setlocal EnableDelayedExpansion

echo ========================================================================
echo Complete NPU Profiling - SAGE (4 stages) + GAT (2 stages)
echo ========================================================================
echo.
echo Based on overlap_summary.md:
echo   SAGE: Stage 1 (Gather), 5 (Normalize), 6 (Transform), 7 (Activate)
echo   GAT:  Stage 3 (AttentionScore), 5 (MessageWeighted)
echo   GCN:  SKIPPED (100%% covered)
echo   Total: 6 stages x 8 node sizes = 48 measurements
echo.

REM --- Activate conda environment ---
if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else (
    CALL "C:\Users\29067\anaconda3\Scripts\activate.bat" MIX
)
cd /d "%~dp0"

REM ========================================================================
REM Step 1: Export NPU static models (if not already exported)
REM ========================================================================

echo.
echo [Step 1/3] Checking and exporting NPU static models...
echo ----------------------------------------------------------------

if not exist "exported_models\stage1_npu_n1000_e5000.xml" (
    echo Exporting SAGE NPU static models...
    python profile_stages.py --export-npu
) else (
    echo SAGE NPU models already exported, skipping.
)

if not exist "gat_exported_models\stage3_npu_n1000_e5000.xml" (
    echo Exporting GAT NPU static models...
    python gat_profile_stages.py --export-npu
) else (
    echo GAT NPU models already exported, skipping.
)

REM ========================================================================
REM Step 2: SAGE NPU - Stage 1, 5, 6, 7 (4 stages x 8 sizes = 32)
REM ========================================================================

echo.
echo [Step 2/3] SAGE NPU Profiling (Stage 1, 5, 6, 7)
echo ----------------------------------------------------------------

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

REM ========================================================================
REM Step 3: GAT NPU - Stage 3, 5 (2 stages x 8 sizes = 16)
REM ========================================================================

echo.
echo [Step 3/3] GAT NPU Profiling (Stage 3, 5)
echo ----------------------------------------------------------------

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
REM Merge results
REM ========================================================================

echo.
echo ========================================================================
echo Merging NPU results...
echo ========================================================================
python profile_stages.py --merge-npu --platform 265V
python gat_profile_stages.py --merge-npu --platform 265V

echo.
echo ========================================================================
echo NPU Profiling Complete!
echo ========================================================================
echo.
echo Results:
echo   SAGE NPU: results\265V\sage\checkpoint_npu.json
echo   GAT  NPU: results\265V\gat\checkpoint_npu.json
echo.
pause
