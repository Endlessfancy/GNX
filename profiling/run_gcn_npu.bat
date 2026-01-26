@echo off
REM ========================================================================
REM GCN NPU Simple - No for loops
REM ========================================================================

setlocal EnableDelayedExpansion

echo ========================================================================
echo GCN NPU Simple Test
echo ========================================================================
echo.

REM Activate conda
echo Activating MIX environment...
if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else (
    CALL "C:\Users\29067\anaconda3\Scripts\activate.bat" MIX
)

cd /d "%~dp0"

echo.
echo GCN NPU stages: 2, 3, 5, 6 (skip 1, 4 - scatter ops)
echo.

REM Check models
if not exist "gcn_exported_models\stage2_npu_n1000_e2000.xml" (
    echo Exporting NPU models...
    python gcn_profile_stages.py --export-npu
)

echo.
echo Starting NPU tests...
echo.

REM Test 1000 nodes - stages 2,3,5,6
echo [1/32] Stage 2, 1000 nodes...
python gcn_profile_npu.py --nodes 1000 --stage 2

echo [2/32] Stage 3, 1000 nodes...
python gcn_profile_npu.py --nodes 1000 --stage 3

echo [3/32] Stage 5, 1000 nodes...
python gcn_profile_npu.py --nodes 1000 --stage 5

echo [4/32] Stage 6, 1000 nodes...
python gcn_profile_npu.py --nodes 1000 --stage 6

REM Test 2000 nodes
echo [5/32] Stage 2, 2000 nodes...
python gcn_profile_npu.py --nodes 2000 --stage 2

echo [6/32] Stage 3, 2000 nodes...
python gcn_profile_npu.py --nodes 2000 --stage 3

echo [7/32] Stage 5, 2000 nodes...
python gcn_profile_npu.py --nodes 2000 --stage 5

echo [8/32] Stage 6, 2000 nodes...
python gcn_profile_npu.py --nodes 2000 --stage 6

REM Test 5000 nodes
echo [9/32] Stage 2, 5000 nodes...
python gcn_profile_npu.py --nodes 5000 --stage 2

echo [10/32] Stage 3, 5000 nodes...
python gcn_profile_npu.py --nodes 5000 --stage 3

echo [11/32] Stage 5, 5000 nodes...
python gcn_profile_npu.py --nodes 5000 --stage 5

echo [12/32] Stage 6, 5000 nodes...
python gcn_profile_npu.py --nodes 5000 --stage 6

REM Test 10000 nodes
echo [13/32] Stage 2, 10000 nodes...
python gcn_profile_npu.py --nodes 10000 --stage 2

echo [14/32] Stage 3, 10000 nodes...
python gcn_profile_npu.py --nodes 10000 --stage 3

echo [15/32] Stage 5, 10000 nodes...
python gcn_profile_npu.py --nodes 10000 --stage 5

echo [16/32] Stage 6, 10000 nodes...
python gcn_profile_npu.py --nodes 10000 --stage 6

REM Test 20000 nodes
echo [17/32] Stage 2, 20000 nodes...
python gcn_profile_npu.py --nodes 20000 --stage 2

echo [18/32] Stage 3, 20000 nodes...
python gcn_profile_npu.py --nodes 20000 --stage 3

echo [19/32] Stage 5, 20000 nodes...
python gcn_profile_npu.py --nodes 20000 --stage 5

echo [20/32] Stage 6, 20000 nodes...
python gcn_profile_npu.py --nodes 20000 --stage 6

REM Test 50000 nodes
echo [21/32] Stage 2, 50000 nodes...
python gcn_profile_npu.py --nodes 50000 --stage 2

echo [22/32] Stage 3, 50000 nodes...
python gcn_profile_npu.py --nodes 50000 --stage 3

echo [23/32] Stage 5, 50000 nodes...
python gcn_profile_npu.py --nodes 50000 --stage 5

echo [24/32] Stage 6, 50000 nodes...
python gcn_profile_npu.py --nodes 50000 --stage 6

REM Test 80000 nodes
echo [25/32] Stage 2, 80000 nodes...
python gcn_profile_npu.py --nodes 80000 --stage 2

echo [26/32] Stage 3, 80000 nodes...
python gcn_profile_npu.py --nodes 80000 --stage 3

echo [27/32] Stage 5, 80000 nodes...
python gcn_profile_npu.py --nodes 80000 --stage 5

echo [28/32] Stage 6, 80000 nodes...
python gcn_profile_npu.py --nodes 80000 --stage 6

REM Test 100000 nodes
echo [29/32] Stage 2, 100000 nodes...
python gcn_profile_npu.py --nodes 100000 --stage 2

echo [30/32] Stage 3, 100000 nodes...
python gcn_profile_npu.py --nodes 100000 --stage 3

echo [31/32] Stage 5, 100000 nodes...
python gcn_profile_npu.py --nodes 100000 --stage 5

echo [32/32] Stage 6, 100000 nodes...
python gcn_profile_npu.py --nodes 100000 --stage 6

echo.
echo ========================================================================
echo Merging results...
echo ========================================================================
python gcn_profile_stages.py --merge-npu --platform 265V

echo.
echo Done!
echo.

pause
