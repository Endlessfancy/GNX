#!/bin/bash
# ========================================================================
# GNN Stage Profiling - Linux Version
# ========================================================================
#
# New architecture: NPU tests run in isolated processes per (nodes, stage)
# This prevents DEVICE_LOST errors from cascading and allows finding NPU limits.
#
# Workflow:
#   Phase 1: Export all models
#   Phase 2: Measure CPU/GPU latencies
#   Phase 3: Measure NPU latencies (isolated per nodes/stage)
#   Phase 4: Merge results and analyze

set -e

echo "========================================================================"
echo "GNN Stage Profiling - Linux"
echo "========================================================================"
echo

# Configuration
NODE_SIZES=(1000 5000 10000 20000 30000 40000 50000 60000 80000 100000 150000)
NPU_STAGES=(1 2 5 6 7)

echo "Test cases: 55 combinations (1k-150k nodes, 5 edge ratios each)"
echo "NPU stages: 1, 2, 5, 6, 7 (skip Stage 3/4 - no scatter_add support)"
echo "NPU tests: ${#NODE_SIZES[@]} node sizes x ${#NPU_STAGES[@]} stages = $((${#NODE_SIZES[@]} * ${#NPU_STAGES[@]})) isolated processes"
echo

# ========================================================================
# PHASE 1: Export all models
# ========================================================================
echo "========================================================================"
echo "PHASE 1: Exporting All Models"
echo "========================================================================"
echo

python profile_stages.py --export

echo
echo "Phase 1 complete: All models exported."
echo

# ========================================================================
# PHASE 2: Measure CPU/GPU latencies
# ========================================================================
echo "========================================================================"
echo "PHASE 2: Measuring CPU/GPU Latencies"
echo "========================================================================"
echo

python profile_stages.py --measure

echo
echo "Phase 2 complete: CPU/GPU measurements saved."
echo

# ========================================================================
# PHASE 3: Measure NPU latencies (isolated per nodes/stage)
# ========================================================================
echo "========================================================================"
echo "PHASE 3: Measuring NPU Latencies (Isolated Processes)"
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
echo "Phase 3 Summary"
echo "========================================================================"
echo "  Passed:      $PASSED"
echo "  Failed:      $FAILED"
echo "  Device Lost: $DEVICE_LOST_COUNT"
echo "  Total:       $TOTAL_TESTS"
echo

# ========================================================================
# PHASE 4: Merge NPU results and analyze
# ========================================================================
echo "========================================================================"
echo "PHASE 4: Merging Results and Analyzing"
echo "========================================================================"
echo

echo "Merging NPU checkpoint files..."
set +e
python profile_stages.py --merge-npu
set -e

echo
echo "Generating final analysis..."
python profile_stages.py --analyze

echo
echo "========================================================================"
echo "Profiling Complete!"
echo "========================================================================"
echo
echo "Results saved in: profiling/results/"
echo "  - lookup_table.json      (compute times)"
echo "  - bandwidth_table.json   (bandwidth estimates)"
echo "  - profiling_report.txt   (summary report)"
echo
echo "Checkpoints:"
echo "  - checkpoint_cpugpu.json (CPU/GPU data)"
echo "  - checkpoint_npu.json    (merged NPU data)"
echo "  - npu_stage*_n*.json     (individual NPU test results)"
echo
echo "NPU Test Summary:"
echo "  Passed: $PASSED / $TOTAL_TESTS"
echo "  Failed: $FAILED"
echo "  Device Lost: $DEVICE_LOST_COUNT"
echo
