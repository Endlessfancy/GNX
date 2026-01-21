#!/bin/bash
# ========================================================================
# GPU + NPU Testing - Fused Blocks (Linux version)
# ========================================================================
#
# This script tests GPU and NPU only (skips CPU for faster testing):
#   Block 0: GPU only (FusedBlock0 - Stages 1-4: GATHER + REDUCE)
#   Block 1: NPU (FusedBlock1 - Stages 5-7: NORMALIZE + TRANSFORM + ACTIVATE)
#
# NPU uses PROCESS ISOLATION to handle potential crashes gracefully.

set -e  # Exit on error

echo "========================================================================"
echo "GPU + NPU Testing - Fused Blocks"
echo "========================================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

echo "Test Configuration:"
echo "  Block 0 (GPU only): GATHER + REDUCE (aggregation phase)"
echo "  Block 1 (NPU): NORMALIZE + TRANSFORM + ACTIVATE (update phase)"
echo ""
echo "Node sizes: 5k, 10k, 20k, 50k, 80k, 100k"
echo "Edge ratios: 10, 25, 40, 50, 60, 75, 100"
echo "Total: 42 test cases per device"
echo ""

# Create results directory
mkdir -p results

# ========================================================================
# Step 1: Export models (if needed)
# ========================================================================
echo ""
echo "[Step 1/4] Checking and exporting models..."

# Check GPU model
if [ ! -f "exported_models/block0_fused_gpu.xml" ]; then
    echo "  Exporting CPU/GPU fused model..."
    python profile_pep3.py --export-cpugpu
else
    echo "  CPU/GPU model already exists, skipping export."
fi

# Check NPU models
if [ ! -f "exported_models/block1_fused_npu_n5000_e50000.xml" ]; then
    echo "  Exporting NPU fused models (42 static models)..."
    python profile_pep3.py --export-npu
else
    echo "  NPU models already exist, skipping export."
fi

# ========================================================================
# Step 2: Measure GPU
# ========================================================================
echo ""
echo "[Step 2/4] Measuring GPU latencies..."
python profile_pep3.py --measure-gpu || echo "WARNING: Some GPU measurements may have failed"

echo ""
echo "GPU results saved to: results/block0_gpu.json"

# ========================================================================
# Step 3: Measure NPU with PROCESS ISOLATION
# ========================================================================
echo ""
echo "[Step 3/4] Measuring NPU latencies (PROCESS ISOLATION mode)..."
echo "Each node size runs in a separate Python process."
echo ""

for nodes in 5000 10000 20000 50000 80000 100000; do
    echo "  Testing ${nodes} nodes..."
    python profile_npu_isolated.py --nodes ${nodes} && echo "    > All tests PASSED for ${nodes} nodes" || echo "    > Some tests failed for ${nodes} nodes"
done

echo ""
echo "Merging NPU results..."
python merge_npu_results.py

# ========================================================================
# Step 4: Generate analysis
# ========================================================================
echo ""
echo "[Step 4/4] Generating analysis..."
python profile_pep3.py --analyze

echo ""
echo "========================================================================"
echo "GPU + NPU Testing Complete!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  - GPU:  results/block0_gpu.json"
echo "  - NPU:  results/block1_npu.json (merged)"
echo "  - NPU individual: results/npu_n*.json"
echo "  - Summary CSV: results/pep3_latency.csv"
echo ""
