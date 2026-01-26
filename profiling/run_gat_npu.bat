@echo off
REM ========================================================================
REM GAT NPU Simple - No for loops (like profiling_fused)
REM ========================================================================

setlocal EnableDelayedExpansion

echo ========================================================================
echo GAT NPU Simple Test
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
echo GAT NPU stages: 1, 2, 3, 5, 7
echo.

REM Check models
if not exist "gat_exported_models\stage1_npu_n1000_e2000.xml" (
    echo Exporting NPU models...
    python gat_profile_stages.py --export-npu
)

echo.
echo Starting NPU tests...
echo.

REM Test 1000 nodes - all stages
echo [1/40] Stage 1, 1000 nodes...
python gat_profile_npu.py --nodes 1000 --stage 1

echo [2/40] Stage 2, 1000 nodes...
python gat_profile_npu.py --nodes 1000 --stage 2

echo [3/40] Stage 3, 1000 nodes...
python gat_profile_npu.py --nodes 1000 --stage 3

echo [4/40] Stage 5, 1000 nodes...
python gat_profile_npu.py --nodes 1000 --stage 5

echo [5/40] Stage 7, 1000 nodes...
python gat_profile_npu.py --nodes 1000 --stage 7

REM Test 2000 nodes
echo [6/40] Stage 1, 2000 nodes...
python gat_profile_npu.py --nodes 2000 --stage 1

echo [7/40] Stage 2, 2000 nodes...
python gat_profile_npu.py --nodes 2000 --stage 2

echo [8/40] Stage 3, 2000 nodes...
python gat_profile_npu.py --nodes 2000 --stage 3

echo [9/40] Stage 5, 2000 nodes...
python gat_profile_npu.py --nodes 2000 --stage 5

echo [10/40] Stage 7, 2000 nodes...
python gat_profile_npu.py --nodes 2000 --stage 7

REM Test 5000 nodes
echo [11/40] Stage 1, 5000 nodes...
python gat_profile_npu.py --nodes 5000 --stage 1

echo [12/40] Stage 2, 5000 nodes...
python gat_profile_npu.py --nodes 5000 --stage 2

echo [13/40] Stage 3, 5000 nodes...
python gat_profile_npu.py --nodes 5000 --stage 3

echo [14/40] Stage 5, 5000 nodes...
python gat_profile_npu.py --nodes 5000 --stage 5

echo [15/40] Stage 7, 5000 nodes...
python gat_profile_npu.py --nodes 5000 --stage 7

REM Test 10000 nodes
echo [16/40] Stage 1, 10000 nodes...
python gat_profile_npu.py --nodes 10000 --stage 1

echo [17/40] Stage 2, 10000 nodes...
python gat_profile_npu.py --nodes 10000 --stage 2

echo [18/40] Stage 3, 10000 nodes...
python gat_profile_npu.py --nodes 10000 --stage 3

echo [19/40] Stage 5, 10000 nodes...
python gat_profile_npu.py --nodes 10000 --stage 5

echo [20/40] Stage 7, 10000 nodes...
python gat_profile_npu.py --nodes 10000 --stage 7

REM Test 20000 nodes
echo [21/40] Stage 1, 20000 nodes...
python gat_profile_npu.py --nodes 20000 --stage 1

echo [22/40] Stage 2, 20000 nodes...
python gat_profile_npu.py --nodes 20000 --stage 2

echo [23/40] Stage 3, 20000 nodes...
python gat_profile_npu.py --nodes 20000 --stage 3

echo [24/40] Stage 5, 20000 nodes...
python gat_profile_npu.py --nodes 20000 --stage 5

echo [25/40] Stage 7, 20000 nodes...
python gat_profile_npu.py --nodes 20000 --stage 7

REM Test 50000 nodes
echo [26/40] Stage 1, 50000 nodes...
python gat_profile_npu.py --nodes 50000 --stage 1

echo [27/40] Stage 2, 50000 nodes...
python gat_profile_npu.py --nodes 50000 --stage 2

echo [28/40] Stage 3, 50000 nodes...
python gat_profile_npu.py --nodes 50000 --stage 3

echo [29/40] Stage 5, 50000 nodes...
python gat_profile_npu.py --nodes 50000 --stage 5

echo [30/40] Stage 7, 50000 nodes...
python gat_profile_npu.py --nodes 50000 --stage 7

REM Test 80000 nodes
echo [31/40] Stage 1, 80000 nodes...
python gat_profile_npu.py --nodes 80000 --stage 1

echo [32/40] Stage 2, 80000 nodes...
python gat_profile_npu.py --nodes 80000 --stage 2

echo [33/40] Stage 3, 80000 nodes...
python gat_profile_npu.py --nodes 80000 --stage 3

echo [34/40] Stage 5, 80000 nodes...
python gat_profile_npu.py --nodes 80000 --stage 5

echo [35/40] Stage 7, 80000 nodes...
python gat_profile_npu.py --nodes 80000 --stage 7

REM Test 100000 nodes
echo [36/40] Stage 1, 100000 nodes...
python gat_profile_npu.py --nodes 100000 --stage 1

echo [37/40] Stage 2, 100000 nodes...
python gat_profile_npu.py --nodes 100000 --stage 2

echo [38/40] Stage 3, 100000 nodes...
python gat_profile_npu.py --nodes 100000 --stage 3

echo [39/40] Stage 5, 100000 nodes...
python gat_profile_npu.py --nodes 100000 --stage 5

echo [40/40] Stage 7, 100000 nodes...
python gat_profile_npu.py --nodes 100000 --stage 7

echo.
echo ========================================================================
echo Merging results...
echo ========================================================================
python gat_profile_stages.py --merge-npu --platform 265V

echo.
echo Done!
echo.

pause
