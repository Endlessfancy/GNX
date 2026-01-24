@echo off
REM CPU Thread Configuration Test for OpenVINO
REM Tests different thread settings to find optimal CPU utilization
REM
REM Usage:
REM   run_cpu_thread_test.bat           - Test all configurations
REM   run_cpu_thread_test.bat --threads 8  - Test specific thread count

echo ============================================================
echo CPU Thread Configuration Test (Stage 1-4 Fused Block)
echo ============================================================
echo.

REM Check if model exists
if not exist "%~dp0exported_models\block0_fused_cpu.xml" (
    echo ERROR: Model not found: exported_models\block0_fused_cpu.xml
    echo Please run model export first: python profile_pep3.py --export
    pause
    exit /b 1
)

REM Run the test
cd /d "%~dp0"

if "%1"=="" (
    echo Running all configurations...
    python test_cpu_threads.py --all
) else (
    echo Running with arguments: %*
    python test_cpu_threads.py %*
)

echo.
echo ============================================================
echo Test completed!
echo ============================================================
pause
