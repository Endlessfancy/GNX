#!/usr/bin/env python3
"""
Complete Pipeline Runner - Python Version (Cross-platform)
完整Pipeline运行脚本（Python版本，跨平台）
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime


class Colors:
    """Terminal colors"""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color


def print_header(text):
    """Print colored header"""
    print(f"{Colors.BLUE}{'=' * 80}{Colors.NC}")
    print(f"{Colors.BLUE}{text.center(80)}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 80}{Colors.NC}")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.YELLOW}{text}{Colors.NC}")


def run_command(cmd, cwd=None, log_file=None):
    """Run shell command and capture output"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )

        if log_file:
            with open(log_file, 'w') as f:
                f.write(result.stdout)
                f.write(result.stderr)

        return True, result.stdout, result.stderr

    except subprocess.CalledProcessError as e:
        if log_file:
            with open(log_file, 'w') as f:
                f.write(e.stdout)
                f.write(e.stderr)

        return False, e.stdout, e.stderr


def main():
    """Main pipeline execution"""
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    # Record start time
    pipeline_start = time.time()

    print_header("GNN Complete Pipeline - Compiler → Executor → Verification")
    print()

    # ========================================================================
    # Phase 1: Compiler
    # ========================================================================
    print_info("[Phase 1/3] Running Compiler...")
    print("  - Graph partitioning with METIS")
    print("  - PEP generation and optimization")
    print("  - Execution plan generation")
    print()

    compiler_start = time.time()
    compiler_dir = script_dir / "compiler"
    compiler_log = script_dir / "logs" / "compiler_output.log"
    compiler_log.parent.mkdir(exist_ok=True)

    # Clean old outputs
    print("  Cleaning old results...")
    output_dir = compiler_dir / "output"
    if (output_dir / "compilation_result.json").exists():
        (output_dir / "compilation_result.json").unlink()

    models_dir = output_dir / "models"
    if models_dir.exists():
        for f in models_dir.glob("*.onnx"):
            f.unlink()
        for f in models_dir.glob("*.ir"):
            f.unlink()

    # Run compiler
    print("  Running compiler...")
    success, stdout, stderr = run_command(
        "python test_compiler_flickr.py",
        cwd=compiler_dir,
        log_file=compiler_log
    )

    if not success:
        print_error("Compiler failed!")
        print(f"  See log: {compiler_log}")
        print(stderr)
        return 1

    compiler_time = time.time() - compiler_start

    # Verify compilation result
    result_file = compiler_dir / "output" / "compilation_result.json"
    if not result_file.exists():
        print_error("Compilation result not found!")
        return 1

    # Load compiler results
    with open(result_file, 'r') as f:
        compilation_result = json.load(f)

    num_subgraphs = compilation_result['partition_config']['num_subgraphs']
    num_models = compilation_result['statistics']['num_unique_models']
    estimated_makespan = compilation_result['statistics']['makespan']

    print_success(f"Compiler completed in {compiler_time:.1f}s")
    print()
    print("  Compilation Summary:")
    print(f"    - Subgraphs: {num_subgraphs}")
    print(f"    - Unique models: {num_models}")
    print(f"    - Estimated makespan: {estimated_makespan:.2f}ms")
    print()

    # ========================================================================
    # Phase 2: Model Export (auto)
    # ========================================================================
    print_info("[Phase 2/3] Model Export...")
    print("  - Will be handled automatically by executor")
    print()

    # ========================================================================
    # Phase 3: Executor
    # ========================================================================
    print_info("[Phase 3/3] Running Executor...")
    print("  - Loading graph data and partitions")
    print("  - Collecting ghost node features")
    print("  - Exporting real ONNX/IR models (if needed)")
    print("  - Executing inference on all subgraphs")
    print()

    executor_start = time.time()
    executor_dir = script_dir / "executer"
    executor_log = script_dir / "logs" / "executor_output.log"

    # Run executor
    success, stdout, stderr = run_command(
        "python test_executor.py",
        cwd=executor_dir,
        log_file=executor_log
    )

    if not success:
        print_error("Executor failed!")
        print(f"  See log: {executor_log}")
        print(stderr)
        return 1

    executor_time = time.time() - executor_start

    print_success(f"Executor completed in {executor_time:.1f}s")
    print()

    # ========================================================================
    # Results Summary
    # ========================================================================
    pipeline_time = time.time() - pipeline_start

    print_header("PIPELINE SUMMARY")
    print()

    print_info("Execution Time Breakdown:")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ Phase 1: Compiler                    {compiler_time:6.1f}s         │")
    print(f"  │ Phase 2: Model Export                (auto)         │")
    print(f"  │ Phase 3: Executor                    {executor_time:6.1f}s         │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print(f"  │ Total Pipeline Time:                 {pipeline_time:6.1f}s         │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()

    # Extract actual performance from executor log
    actual_latency = "N/A"
    error_pct = "N/A"

    try:
        with open(executor_log, 'r') as f:
            for line in f:
                if "Actual latency:" in line:
                    actual_latency = line.split("Actual latency:")[1].strip()
                if "Estimation error:" in line:
                    error_pct = line.split("Estimation error:")[1].strip()
    except:
        pass

    print_info("Performance Results:")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ Compiler Estimated Makespan:         {estimated_makespan:>6.2f}ms       │")
    print(f"  │ Actual Measured Latency:             {actual_latency:>10}       │")
    print(f"  │ Estimation Error:                    {error_pct:>10}       │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()

    print_info("Output Files:")
    print(f"  - Compilation result: compiler/output/compilation_result.json")
    print(f"  - ONNX models: compiler/output/models/*.onnx")
    print(f"  - Compiler log: {compiler_log}")
    print(f"  - Executor log: {executor_log}")
    print()

    # Check accuracy
    if error_pct != "N/A":
        try:
            error_value = float(error_pct.replace('%', '').replace('+', ''))
            if abs(error_value) < 20:
                print_success("Estimation is accurate (within 20%)")
            else:
                print_info("Estimation deviates significantly (>20%)")
        except:
            pass

    print()
    print_header("Pipeline completed successfully!")
    print()

    # Save summary
    summary_file = script_dir / "pipeline_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("GNN Pipeline Execution Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write("TIMING BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        f.write(f"Compiler:                 {compiler_time:.1f}s\n")
        f.write(f"Executor:                 {executor_time:.1f}s\n")
        f.write(f"Total:                    {pipeline_time:.1f}s\n\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Estimated Makespan:       {estimated_makespan:.2f}ms\n")
        f.write(f"Actual Latency:           {actual_latency}\n")
        f.write(f"Estimation Error:         {error_pct}\n\n")
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Dataset:                  Flickr (89,250 nodes, 899,756 edges)\n")
        f.write(f"Subgraphs:                {num_subgraphs}\n")
        f.write(f"Unique Models:            {num_models}\n\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 80 + "\n")
        f.write("- compiler/output/compilation_result.json\n")
        f.write("- compiler/output/models/*.onnx\n")
        f.write(f"- {compiler_log}\n")
        f.write(f"- {executor_log}\n\n")
        f.write("=" * 80 + "\n")

    print(f"Summary saved to: {summary_file}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
