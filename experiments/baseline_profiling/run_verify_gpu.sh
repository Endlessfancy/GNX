#!/bin/bash
# GPU Latency Verification Script
# Compares different measurement methods to verify GPU profiling accuracy

echo "============================================================"
echo "GPU Latency Verification"
echo "============================================================"
echo

cd "$(dirname "$0")"

echo "Running GPU verification test..."
echo

python verify_gpu_latency.py

echo
echo "============================================================"
echo "Verification complete"
echo "============================================================"
echo "Results saved to: results/gpu_verification_results.json"
