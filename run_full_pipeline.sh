#!/bin/bash
################################################################################
# GNN Complete Pipeline - Full Workflow
# 完整流程：编译 → 模型导出 → 执行推理
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Record start time
PIPELINE_START=$(date +%s)

echo -e "${BLUE}################################################################################${NC}"
echo -e "${BLUE}#                                                                              #${NC}"
echo -e "${BLUE}#        GNN Complete Pipeline - Compiler → Executor → Verification          #${NC}"
echo -e "${BLUE}#                                                                              #${NC}"
echo -e "${BLUE}################################################################################${NC}"
echo ""

################################################################################
# Phase 1: Compiler - Graph Partitioning and Optimization
################################################################################

echo -e "${GREEN}[Phase 1/3] Running Compiler...${NC}"
echo "  - Graph partitioning with METIS"
echo "  - PEP generation and optimization"
echo "  - Execution plan generation"
echo ""

COMPILER_START=$(date +%s)

cd compiler

# Clean old outputs
echo "  Cleaning old compilation results..."
rm -f output/compilation_result.json
rm -f output/models/*.onnx 2>/dev/null || true
rm -f output/models/*.ir 2>/dev/null || true

echo "  Running compiler..."
python test_compiler_flickr.py > /tmp/compiler_output.log 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}  ✗ Compiler failed!${NC}"
    echo "  See log: /tmp/compiler_output.log"
    cat /tmp/compiler_output.log
    exit 1
fi

COMPILER_END=$(date +%s)
COMPILER_TIME=$((COMPILER_END - COMPILER_START))

# Verify compilation result exists
if [ ! -f "output/compilation_result.json" ]; then
    echo -e "${RED}  ✗ Compilation result not found!${NC}"
    exit 1
fi

echo -e "${GREEN}  ✓ Compiler completed in ${COMPILER_TIME}s${NC}"
echo ""

# Extract compiler statistics
ESTIMATED_MAKESPAN=$(python3 -c "
import json
with open('output/compilation_result.json', 'r') as f:
    data = json.load(f)
    print(f\"{data['statistics']['makespan']:.2f}\")
")

NUM_SUBGRAPHS=$(python3 -c "
import json
with open('output/compilation_result.json', 'r') as f:
    data = json.load(f)
    print(data['partition_config']['num_subgraphs'])
")

NUM_MODELS=$(python3 -c "
import json
with open('output/compilation_result.json', 'r') as f:
    data = json.load(f)
    print(data['statistics']['num_unique_models'])
")

echo "  Compilation Summary:"
echo "    - Subgraphs: $NUM_SUBGRAPHS"
echo "    - Unique models: $NUM_MODELS"
echo "    - Estimated makespan: ${ESTIMATED_MAKESPAN}ms"
echo ""

cd ..

################################################################################
# Phase 2: Model Export (handled by executor automatically)
################################################################################

echo -e "${GREEN}[Phase 2/3] Model Export...${NC}"
echo "  - Model export will be handled by executor automatically"
echo "  - Placeholder models will be replaced with real ONNX models"
echo ""

################################################################################
# Phase 3: Executor - Pipeline Inference
################################################################################

echo -e "${GREEN}[Phase 3/3] Running Executor...${NC}"
echo "  - Loading graph data and partitions"
echo "  - Collecting ghost node features"
echo "  - Exporting real ONNX/IR models (if needed)"
echo "  - Executing inference on all subgraphs"
echo ""

EXECUTOR_START=$(date +%s)

cd executer

python test_executor.py > /tmp/executor_output.log 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}  ✗ Executor failed!${NC}"
    echo "  See log: /tmp/executor_output.log"
    cat /tmp/executor_output.log
    exit 1
fi

EXECUTOR_END=$(date +%s)
EXECUTOR_TIME=$((EXECUTOR_END - EXECUTOR_START))

echo -e "${GREEN}  ✓ Executor completed in ${EXECUTOR_TIME}s${NC}"
echo ""

cd ..

################################################################################
# Results Summary
################################################################################

PIPELINE_END=$(date +%s)
TOTAL_TIME=$((PIPELINE_END - PIPELINE_START))

echo -e "${BLUE}################################################################################${NC}"
echo -e "${BLUE}#                                                                              #${NC}"
echo -e "${BLUE}#                          PIPELINE SUMMARY                                   #${NC}"
echo -e "${BLUE}#                                                                              #${NC}"
echo -e "${BLUE}################################################################################${NC}"
echo ""

echo -e "${YELLOW}Execution Time Breakdown:${NC}"
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │ Phase 1: Compiler                    ${COMPILER_TIME}s          │"
echo "  │ Phase 2: Model Export                (auto)        │"
echo "  │ Phase 3: Executor                    ${EXECUTOR_TIME}s          │"
echo "  ├─────────────────────────────────────────────────────────┤"
echo "  │ Total Pipeline Time:                 ${TOTAL_TIME}s          │"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""

# Extract actual inference time from executor log
ACTUAL_LATENCY=$(grep "Actual latency:" /tmp/executor_output.log | awk '{print $3}' || echo "N/A")
ERROR_PCT=$(grep "Estimation error:" /tmp/executor_output.log | awk '{print $3}' || echo "N/A")

echo -e "${YELLOW}Performance Results:${NC}"
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │ Compiler Estimated Makespan:         ${ESTIMATED_MAKESPAN}ms        │"
echo "  │ Actual Measured Latency:             ${ACTUAL_LATENCY}        │"
echo "  │ Estimation Error:                    ${ERROR_PCT}        │"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""

echo -e "${YELLOW}Output Files:${NC}"
echo "  - Compilation result: compiler/output/compilation_result.json"
echo "  - ONNX models: compiler/output/models/*.onnx"
echo "  - Compiler log: /tmp/compiler_output.log"
echo "  - Executor log: /tmp/executor_output.log"
echo ""

# Check if estimation is accurate
if [ "$ERROR_PCT" != "N/A" ]; then
    ERROR_NUM=$(echo $ERROR_PCT | sed 's/%//' | sed 's/+//')
    if (( $(echo "$ERROR_NUM < 20" | bc -l 2>/dev/null || echo "1") )); then
        echo -e "${GREEN}✓ Estimation is accurate (within 20%)${NC}"
    else
        echo -e "${YELLOW}⚠ Estimation deviates significantly (>20%)${NC}"
    fi
fi

echo ""
echo -e "${BLUE}################################################################################${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${BLUE}################################################################################${NC}"
echo ""

# Save summary to file
SUMMARY_FILE="pipeline_summary.txt"
cat > "$SUMMARY_FILE" << EOF
GNN Pipeline Execution Summary
Generated: $(date)
================================================================================

TIMING BREAKDOWN
----------------
Compiler:                 ${COMPILER_TIME}s
Executor:                 ${EXECUTOR_TIME}s
Total:                    ${TOTAL_TIME}s

PERFORMANCE METRICS
-------------------
Estimated Makespan:       ${ESTIMATED_MAKESPAN}ms
Actual Latency:           ${ACTUAL_LATENCY}
Estimation Error:         ${ERROR_PCT}

CONFIGURATION
-------------
Dataset:                  Flickr (89,250 nodes, 899,756 edges)
Subgraphs:                $NUM_SUBGRAPHS
Unique Models:            $NUM_MODELS

OUTPUT FILES
------------
- compiler/output/compilation_result.json
- compiler/output/models/*.onnx
- /tmp/compiler_output.log
- /tmp/executor_output.log

================================================================================
EOF

echo "Summary saved to: $SUMMARY_FILE"
