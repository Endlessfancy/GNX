#!/bin/bash
# ========================================================================
# NPU Only Profiling - Linux Version
# ========================================================================
#
# This script ONLY runs NPU tests (skips CPU/GPU).
# Use this when you already have CPU/GPU results and just need to re-run NPU.
#
# Prerequisites:
#   - NPU models must be exported: python profile_stages.py --export-npu
#   - Or run full export first: python profile_stages.py --export

echo "========================================================================"
echo "NPU Only Profiling"
echo "========================================================================"
echo

# Configuration
NODE_SIZES=(1000 5000 10000 20000 30000 40000 50000 60000 80000 100000 150000)
NPU_STAGES=(1 2 5 6 7)

echo "NPU stages: 1, 2, 5, 6, 7 (skip Stage 3/4 - no scatter_add support)"
echo "NPU tests: ${#NODE_SIZES[@]} node sizes x ${#NPU_STAGES[@]} stages = $((${#NODE_SIZES[@]} * ${#NPU_STAGES[@]})) isolated processes"
echo

# Check if NPU models exist
if [ ! -f "exported_models/stage1_npu_n1000_e2000.xml" ]; then
    echo "WARNING: NPU models not found!"
    echo "Run first: python profile_stages.py --export-npu"
    read -p "Export NPU models now? (y/n): " EXPORT_NOW
    if [ "$EXPORT_NOW" = "y" ] || [ "$EXPORT_NOW" = "Y" ]; then
        python profile_stages.py --export-npu
        if [ $? -ne 0 ]; then
            echo "ERROR: NPU model export failed!"
            exit 1
        fi
    else
        echo "Aborted."
        exit 1
    fi
fi

# ========================================================================
# NPU Testing (isolated per nodes/stage)
# ========================================================================
echo
echo "========================================================================"
echo "Running NPU Tests (Isolated Processes)"
echo "========================================================================"
echo
echo "Each (nodes, stage) combination runs in its own Python process."
echo "If DEVICE_LOST occurs, that test fails but others continue."
echo

TOTAL_TESTS=$((${#NODE_SIZES[@]} * ${#NPU_STAGES[@]}))
echo "Total NPU tests: $TOTAL_TESTS"
echo

CURRENT_TEST=0
PASSED=0
FAILED=0
DEVICE_LOST_COUNT=0

for nodes in "${NODE_SIZES[@]}"; do
    for stage in "${NPU_STAGES[@]}"; do
        CURRENT_TEST=$((CURRENT_TEST + 1))
        echo
        echo "[$CURRENT_TEST/$TOTAL_TESTS] Testing Stage $stage with $nodes nodes..."

        set +e
        python profile_npu.py --nodes "$nodes" --stage "$stage"
        EXIT_CODE=$?
        set -e

        if [ $EXIT_CODE -eq 0 ]; then
            echo "  Result: PASSED"
            PASSED=$((PASSED + 1))
        elif [ $EXIT_CODE -eq 1 ]; then
            echo "  Result: PARTIAL FAILURE (some edges failed)"
            FAILED=$((FAILED + 1))
        elif [ $EXIT_CODE -eq 2 ]; then
            echo "  Result: DEVICE_LOST - NPU in bad state"
            DEVICE_LOST_COUNT=$((DEVICE_LOST_COUNT + 1))
            echo "  Continuing to next test..."
        else
            echo "  Result: UNKNOWN ERROR (code=$EXIT_CODE)"
            FAILED=$((FAILED + 1))
        fi
    done
done

echo
echo "========================================================================"
echo "NPU Testing Summary"
echo "========================================================================"
echo "  Passed:      $PASSED"
echo "  Failed:      $FAILED"
echo "  Device Lost: $DEVICE_LOST_COUNT"
echo "  Total:       $TOTAL_TESTS"
echo

# ========================================================================
# Merge NPU results
# ========================================================================
echo "========================================================================"
echo "Merging NPU Results"
echo "========================================================================"
echo

set +e
python profile_stages.py --merge-npu
set -e

echo
echo "========================================================================"
echo "NPU Profiling Complete!"
echo "========================================================================"
echo
echo "Results saved in: profiling/results/"
echo "  - checkpoint_npu.json    (merged NPU data)"
echo "  - npu_stage*_n*.json     (individual NPU test results)"
echo
echo "Next steps:"
echo "  1. Run analysis: python profile_stages.py --analyze"
echo "  2. Or run full workflow: ./run_profiling.sh"
echo
echo "NPU Test Summary:"
echo "  Passed: $PASSED / $TOTAL_TESTS"
echo "  Failed: $FAILED"
echo "  Device Lost: $DEVICE_LOST_COUNT"
echo
