@echo off
REM ========================================================================
REM GAT NPU Profiling (Stage 3, 5)
REM
REM GAT Stage 3 (AttentionScore) + Stage 5 (MessageWeighted)
REM Total: 2 stages x 8 node sizes = 16 measurements
REM ========================================================================
setlocal EnableDelayedExpansion

echo ========================================================================
echo GAT NPU Profiling - Stage 3, 5
echo ========================================================================
echo.
echo   Stage 3: AttentionScore
echo   Stage 5: MessageWeighted
echo   Total: 2 stages x 8 node sizes = 16 measurements
echo.

REM --- Activate conda environment ---
if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else (
    CALL "C:\Users\29067\anaconda3\Scripts\activate.bat" MIX
)
cd /d "%~dp0"

REM ========================================================================
REM Step 1: Export GAT NPU static models (auto-skip existing)
REM ========================================================================

echo.
echo ================================================================
echo [Step 1/3] Exporting GAT NPU static models (auto-skip existing)
echo ================================================================
python gat_profile_stages.py --export-npu

REM ========================================================================
REM Step 2: GAT NPU Profiling
REM ========================================================================

echo.
echo ================================================================
echo [Step 2/3] GAT NPU Profiling (Stage 3, 5)
echo ================================================================

REM 1000 nodes
echo [1/16] GAT Stage 3, 1000 nodes...
python gat_profile_npu.py --nodes 1000 --stage 3
echo [2/16] GAT Stage 5, 1000 nodes...
python gat_profile_npu.py --nodes 1000 --stage 5

REM 2000 nodes
echo [3/16] GAT Stage 3, 2000 nodes...
python gat_profile_npu.py --nodes 2000 --stage 3
echo [4/16] GAT Stage 5, 2000 nodes...
python gat_profile_npu.py --nodes 2000 --stage 5

REM 5000 nodes
echo [5/16] GAT Stage 3, 5000 nodes...
python gat_profile_npu.py --nodes 5000 --stage 3
echo [6/16] GAT Stage 5, 5000 nodes...
python gat_profile_npu.py --nodes 5000 --stage 5

REM 10000 nodes
echo [7/16] GAT Stage 3, 10000 nodes...
python gat_profile_npu.py --nodes 10000 --stage 3
echo [8/16] GAT Stage 5, 10000 nodes...
python gat_profile_npu.py --nodes 10000 --stage 5

REM 20000 nodes
echo [9/16] GAT Stage 3, 20000 nodes...
python gat_profile_npu.py --nodes 20000 --stage 3
echo [10/16] GAT Stage 5, 20000 nodes...
python gat_profile_npu.py --nodes 20000 --stage 5

REM 50000 nodes
echo [11/16] GAT Stage 3, 50000 nodes...
python gat_profile_npu.py --nodes 50000 --stage 3
echo [12/16] GAT Stage 5, 50000 nodes...
python gat_profile_npu.py --nodes 50000 --stage 5

REM 80000 nodes
echo [13/16] GAT Stage 3, 80000 nodes...
python gat_profile_npu.py --nodes 80000 --stage 3
echo [14/16] GAT Stage 5, 80000 nodes...
python gat_profile_npu.py --nodes 80000 --stage 5

REM 100000 nodes
echo [15/16] GAT Stage 3, 100000 nodes...
python gat_profile_npu.py --nodes 100000 --stage 3
echo [16/16] GAT Stage 5, 100000 nodes...
python gat_profile_npu.py --nodes 100000 --stage 5

REM ========================================================================
REM Step 3: Merge results
REM ========================================================================

echo.
echo ================================================================
echo [Step 3/3] Merging GAT NPU results...
echo ================================================================
python gat_profile_stages.py --merge-npu --platform 265V

echo.
echo ========================================================================
echo GAT NPU Profiling Complete!
echo ========================================================================
echo.
echo Results:
echo   GAT NPU: results\265V\gat\checkpoint_npu.json
echo.
pause
