@echo off
REM ========================================================================
REM GAT-only GPU Profiling (stages 3/4/5 - not covered by SAGE)
REM
REM Stage 3: AttentionScore   (dot product + leaky_relu)
REM Stage 4: AttentionSoftmax (scatter_reduce amax + exp + div)
REM Stage 5: MessageWeighted  (element-wise mul [E,1]x[E,F])
REM ========================================================================
setlocal EnableDelayedExpansion

echo ========================================================================
echo GAT GPU Profiling - 3 Unique Stages (3/4/5)
echo ========================================================================
echo.

REM --- Activate conda environment ---
if exist "C:\Env\Anaconda\Scripts\activate.bat" (
    CALL "C:\Env\Anaconda\Scripts\activate.bat" MIX
) else (
    CALL "C:\Users\29067\anaconda3\Scripts\activate.bat" MIX
)
cd /d "%~dp0"

REM --- Export models if needed ---
if not exist "gat_exported_models\stage3_gpu.xml" (
    echo Exporting GAT CPU/GPU models...
    python gat_profile_stages.py --export-cpugpu
) else (
    echo GAT models already exported, skipping.
)

echo.
echo Running GAT GPU measurements (stages 3/4/5 only)...
echo ----------------------------------------------------------------
python gat_profile_stages.py --measure-gpu --platform 265V --stages 3,4,5

echo.
echo ========================================================================
echo Done! Results: results\265V\gat\checkpoint_gpu.json
echo ========================================================================
pause
