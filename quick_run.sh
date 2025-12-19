#!/bin/bash
################################################################################
# Quick Run Script - Minimal Output Version
# 快速运行脚本（简洁版）
################################################################################

set -e
cd "$(dirname "$0")"

echo "=== GNN Pipeline Quick Run ==="
echo ""

# Phase 1: Compiler
echo "[1/3] Compiler..."
cd compiler
python test_compiler_flickr.py 2>&1 | grep -E "(✓|Makespan|Best Configuration)" || true
cd ..

# Phase 2: Executor
echo ""
echo "[2/3] Executor..."
cd executer
python test_executor.py 2>&1 | grep -E "(✓|Actual latency|Estimation error|Shape)" || true
cd ..

# Phase 3: Summary
echo ""
echo "[3/3] Done!"
echo ""
echo "Full logs saved to:"
echo "  - /tmp/compiler_output.log"
echo "  - /tmp/executor_output.log"
