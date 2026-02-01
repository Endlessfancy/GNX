#!/usr/bin/env python3
"""
Phase 4: Model Code Generation

Export and cache ONNX/IR models based on PEP assignment.
- CPU/GPU: Dynamic shape ONNX models
- NPU: Static shape IR models (padded)

Usage:
    python phase4_model_codegen.py --input output/phase3_optimization.json
    python phase4_model_codegen.py --input output/phase3_optimization.json --output output/phase4_models.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import CompilerConfig
from core import ModelCodegen
from core.graph_partitioner import Subgraph
from core.pep_generator import PEP, PEPBlock


# ============================================================================
# Serialization Helpers
# ============================================================================

def dict_to_subgraph(d):
    """dict -> Subgraph"""
    return Subgraph(
        id=d['id'], n=d['n'], m=d['m'],
        n_pad=d['n_pad'], m_pad=d['m_pad'],
        cut_edges=d.get('cut_edges', 0)
    )


def dict_to_pep(d):
    """dict -> PEP"""
    blocks = []
    for block_d in d['blocks']:
        blocks.append(PEPBlock(
            devices=block_d['devices'],
            stages=block_d['stages'],
            ratios=block_d['ratios']
        ))
    return PEP(blocks=blocks)


# ============================================================================
# Phase 4 Main
# ============================================================================

def run_phase4(input_path: Path, output_path: Path, config: CompilerConfig = None):
    """
    Run Phase 4: Model Code Generation

    Args:
        input_path: Phase 3 output JSON path
        output_path: Output JSON path
        config: Compiler config (optional)

    Returns:
        Phase 4 result dict
    """
    print("=" * 60)
    print("Phase 4: Model Code Generation")
    print("=" * 60)

    # Load Phase 3 output
    print(f"\nLoading Phase 3 output: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        phase3_data = json.load(f)

    config = config or CompilerConfig()

    # Reconstruct subgraphs
    subgraphs = [dict_to_subgraph(d) for d in phase3_data['subgraphs']]

    # Reconstruct assignment: {sg_id: (pep, cost)}
    assignment = {}
    for sg_id_str, entry in phase3_data['assignment'].items():
        sg_id = int(sg_id_str)
        pep = dict_to_pep(entry['pep'])
        cost = entry['cost']
        assignment[sg_id] = (pep, cost)

    # Reconstruct clusters: {cluster_key: [Subgraph]}
    clusters = {}
    for cluster_key, sg_dicts in phase3_data['clusters'].items():
        clusters[cluster_key] = [dict_to_subgraph(d) for d in sg_dicts]

    # Initialize model codegen
    model_codegen = ModelCodegen(config)

    # Generate models
    print(f"\nGenerating models...")
    t0 = time.perf_counter()
    model_index = model_codegen.generate_models(assignment, clusters, subgraphs)
    codegen_time = time.perf_counter() - t0

    # Convert Path objects to strings for JSON
    model_index_serialized = {}
    for key, path in model_index.items():
        model_index_serialized[key] = str(path)

    print(f"\n  Generated {len(model_index_serialized)} unique models")
    for key, path in model_index_serialized.items():
        print(f"    {key} -> {path}")
    print(f"  Codegen time: {codegen_time:.2f}s")

    # Build output (carry forward phase3 data for phase5)
    result = {
        'phase': 4,
        'dataset': phase3_data['dataset'],
        'num_nodes': phase3_data['num_nodes'],
        'num_edges': phase3_data['num_edges'],
        'best_k': phase3_data['best_k'],
        'best_makespan': phase3_data['best_makespan'],
        'codegen_time_sec': codegen_time,
        'model_index': model_index_serialized,
        'num_unique_models': len(model_index_serialized),
        # Carry forward for Phase 5
        'assignment': phase3_data['assignment'],
        'clusters': phase3_data['clusters'],
        'subgraphs': phase3_data['subgraphs']
    }

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"\nPhase 4 output saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Model Code Generation")
    parser.add_argument('--input', type=str, default='output/phase3_optimization.json',
                        help='Phase 3 output JSON path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else \
        Path(__file__).parent / 'output' / 'phase4_models.json'

    run_phase4(input_path, output_path)


if __name__ == '__main__':
    main()
