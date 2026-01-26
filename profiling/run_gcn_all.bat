@echo off
REM ========================================================================
REM GCN Full Profiling - CPU + GPU + NPU
REM ========================================================================

setlocal EnableDelayedExpansion

echo ========================================================================
echo GCN Full Profiling - CPU + GPU + NPU
echo ========================================================================
echo.

CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
cd /d "%~dp0"

echo Stages: 1-6 (COMPUTE_NORM, GATHER, MESSAGE, REDUCE_SUM, TRANSFORM, ACTIVATE)
echo NPU skips: Stage 1, 4 (scatter operations)
echo.

REM ========================================================================
REM Phase 1: Export Models
REM ========================================================================
echo [Phase 1] Exporting models...

if not exist "gcn_exported_models\stage1_cpu.xml" (
    echo Exporting CPU/GPU models...
    python gcn_profile_stages.py --export-cpugpu
)

if not exist "gcn_exported_models\stage2_npu_n1000_e2000.xml" (
    echo Exporting NPU models...
    python gcn_profile_stages.py --export-npu
)

echo.

REM ========================================================================
REM Phase 2: CPU Measurement
REM ========================================================================
echo ========================================================================
echo [Phase 2] CPU Measurement
echo ========================================================================
python gcn_profile_stages.py --measure-cpu --platform 185H
echo.

REM ========================================================================
REM Phase 3: GPU Measurement
REM ========================================================================
echo ========================================================================
echo [Phase 3] GPU Measurement
echo ========================================================================
python gcn_profile_stages.py --measure-gpu --platform 185H
echo.

REM ========================================================================
REM Phase 4: NPU Measurement (no for loops)
REM ========================================================================
echo ========================================================================
echo [Phase 4] NPU Measurement
echo ========================================================================
echo.

REM 1000 nodes - stages 2,3,5,6
echo [1/32] Stage 2, 1000 nodes...
python gcn_profile_npu.py --nodes 1000 --stage 2
echo [2/32] Stage 3, 1000 nodes...
python gcn_profile_npu.py --nodes 1000 --stage 3
echo [3/32] Stage 5, 1000 nodes...
python gcn_profile_npu.py --nodes 1000 --stage 5
echo [4/32] Stage 6, 1000 nodes...
python gcn_profile_npu.py --nodes 1000 --stage 6

REM 2000 nodes
echo [5/32] Stage 2, 2000 nodes...
python gcn_profile_npu.py --nodes 2000 --stage 2
echo [6/32] Stage 3, 2000 nodes...
python gcn_profile_npu.py --nodes 2000 --stage 3
echo [7/32] Stage 5, 2000 nodes...
python gcn_profile_npu.py --nodes 2000 --stage 5
echo [8/32] Stage 6, 2000 nodes...
python gcn_profile_npu.py --nodes 2000 --stage 6

REM 5000 nodes
echo [9/32] Stage 2, 5000 nodes...
python gcn_profile_npu.py --nodes 5000 --stage 2
echo [10/32] Stage 3, 5000 nodes...
python gcn_profile_npu.py --nodes 5000 --stage 3
echo [11/32] Stage 5, 5000 nodes...
python gcn_profile_npu.py --nodes 5000 --stage 5
echo [12/32] Stage 6, 5000 nodes...
python gcn_profile_npu.py --nodes 5000 --stage 6

REM 10000 nodes
echo [13/32] Stage 2, 10000 nodes...
python gcn_profile_npu.py --nodes 10000 --stage 2
echo [14/32] Stage 3, 10000 nodes...
python gcn_profile_npu.py --nodes 10000 --stage 3
echo [15/32] Stage 5, 10000 nodes...
python gcn_profile_npu.py --nodes 10000 --stage 5
echo [16/32] Stage 6, 10000 nodes...
python gcn_profile_npu.py --nodes 10000 --stage 6

REM 20000 nodes
echo [17/32] Stage 2, 20000 nodes...
python gcn_profile_npu.py --nodes 20000 --stage 2
echo [18/32] Stage 3, 20000 nodes...
python gcn_profile_npu.py --nodes 20000 --stage 3
echo [19/32] Stage 5, 20000 nodes...
python gcn_profile_npu.py --nodes 20000 --stage 5
echo [20/32] Stage 6, 20000 nodes...
python gcn_profile_npu.py --nodes 20000 --stage 6

REM 50000 nodes
echo [21/32] Stage 2, 50000 nodes...
python gcn_profile_npu.py --nodes 50000 --stage 2
echo [22/32] Stage 3, 50000 nodes...
python gcn_profile_npu.py --nodes 50000 --stage 3
echo [23/32] Stage 5, 50000 nodes...
python gcn_profile_npu.py --nodes 50000 --stage 5
echo [24/32] Stage 6, 50000 nodes...
python gcn_profile_npu.py --nodes 50000 --stage 6

REM 80000 nodes
echo [25/32] Stage 2, 80000 nodes...
python gcn_profile_npu.py --nodes 80000 --stage 2
echo [26/32] Stage 3, 80000 nodes...
python gcn_profile_npu.py --nodes 80000 --stage 3
echo [27/32] Stage 5, 80000 nodes...
python gcn_profile_npu.py --nodes 80000 --stage 5
echo [28/32] Stage 6, 80000 nodes...
python gcn_profile_npu.py --nodes 80000 --stage 6

REM 100000 nodes
echo [29/32] Stage 2, 100000 nodes...
python gcn_profile_npu.py --nodes 100000 --stage 2
echo [30/32] Stage 3, 100000 nodes...
python gcn_profile_npu.py --nodes 100000 --stage 3
echo [31/32] Stage 5, 100000 nodes...
python gcn_profile_npu.py --nodes 100000 --stage 5
echo [32/32] Stage 6, 100000 nodes...
python gcn_profile_npu.py --nodes 100000 --stage 6

REM ========================================================================
REM Phase 5: Merge and Analyze
REM ========================================================================
echo.
echo ========================================================================
echo [Phase 5] Merging NPU results and analyzing...
echo ========================================================================
python gcn_profile_stages.py --merge-npu --platform 185H
python gcn_profile_stages.py --analyze --platform 185H

echo.
echo ========================================================================
echo GCN Profiling Complete!
echo ========================================================================
echo Results: results\185H\gcn\
echo.

pause
