#!/usr/bin/env python3
"""
Phase 5: Execution Plan Generation

Assemble the final compilation_result.json from optimization results
and model index.

Usage:
    python phase5_execution_plan.py --input output/phase4_models.json
    python phase5_execution_plan.py --input output/phase4_models.json --output output/compilation_result.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.pep_generator import PEP, PEPBlock


# ============================================================================
# Serialization Helpers
# ============================================================================

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


def _generate_dynamic_model_key(device, stages):
    """Generate dynamic model key (CPU/GPU)"""
    stages_str = '_'.join(map(str, stages))
    return f"{device}_stages_{stages_str}"


def _generate_static_model_key(device, stages, n_pad, m_pad):
    """Generate static model key (NPU)"""
    stages_str = '_'.join(map(str, stages))
    return f"{device}_stages_{stages_str}_n{n_pad}_m{m_pad}"


# ============================================================================
# Phase 5 Main
# ============================================================================

def run_phase5(input_path: Path, output_path: Path):
    """
    Run Phase 5: Execution Plan Generation

    Args:
        input_path: Phase 4 output JSON path
        output_path: Output JSON path (compilation_result.json)

    Returns:
        Final compilation result dict
    """
    print("=" * 60)
    print("Phase 5: Execution Plan Generation")
    print("=" * 60)

    # Load Phase 4 output
    print(f"\nLoading Phase 4 output: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        phase4_data = json.load(f)

    k = phase4_data['best_k']
    makespan = phase4_data['best_makespan']
    model_index = phase4_data['model_index']
    subgraphs = phase4_data['subgraphs']
    assignment = phase4_data['assignment']
    clusters_data = phase4_data['clusters']

    print(f"  Dataset: {phase4_data['dataset']}")
    print(f"  Best K: {k}")
    print(f"  Makespan: {makespan:.2f}ms")
    print(f"  Unique models: {len(model_index)}")

    # Build cluster execution plans
    cluster_plans = []

    for cluster_key, sg_dicts in clusters_data.items():
        # Get the PEP for this cluster (from first subgraph)
        sample_sg = sg_dicts[0]
        sample_sg_id = str(sample_sg['id'])
        pep_dict = assignment[sample_sg_id]['pep']
        pep = dict_to_pep(pep_dict)

        # Build model references
        model_refs = {}
        for block_idx, block in enumerate(pep.blocks):
            for device in block.devices:
                if device in ['CPU', 'GPU']:
                    model_key = _generate_dynamic_model_key(device, block.stages)
                else:  # NPU
                    model_key = _generate_static_model_key(
                        device, block.stages,
                        sample_sg['n_pad'], sample_sg['m_pad']
                    )

                ref_key = f"block_{block_idx}_{device}"
                abs_path = model_index.get(model_key, "")
                if abs_path:
                    rel_path = f"models/{Path(abs_path).name}"
                    model_refs[ref_key] = rel_path
                else:
                    model_refs[ref_key] = ""

        cluster_plans.append({
            'pep_key': cluster_key,
            'pep': pep.to_executor_format(),
            'subgraph_ids': [sg['id'] for sg in sg_dicts],
            'model_refs': model_refs,
            'num_subgraphs': len(sg_dicts)
        })

    # Assemble final execution plan
    execution_plan = {
        'partition_config': {
            'k': k,
            'num_subgraphs': len(subgraphs),
            'subgraphs': [
                {
                    'id': sg['id'],
                    'n': sg['n'],
                    'm': sg['m'],
                    'n_pad': sg['n_pad'],
                    'm_pad': sg['m_pad'],
                    'cut_edges': sg.get('cut_edges', 0)
                }
                for sg in subgraphs
            ]
        },
        'execution_plan': {
            'clusters': cluster_plans,
            'num_clusters': len(cluster_plans)
        },
        'statistics': {
            'makespan': makespan,
            'num_unique_models': len(model_index),
            'num_subgraphs': len(subgraphs),
            'num_clusters': len(cluster_plans)
        }
    }

    # Print summary
    print(f"\nExecution Plan Summary:")
    print(f"  Clusters: {len(cluster_plans)}")
    for i, cp in enumerate(cluster_plans):
        print(f"    Cluster {i}: {cp['num_subgraphs']} subgraphs, PEP={cp['pep_key']}")
        for ref_key, ref_path in cp['model_refs'].items():
            print(f"      {ref_key} -> {ref_path}")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(execution_plan, f, indent=2)
    print(f"\nFinal execution plan saved to: {output_path}")

    return execution_plan


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Execution Plan Generation")
    parser.add_argument('--input', type=str, default='output/phase4_models.json',
                        help='Phase 4 output JSON path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else \
        Path(__file__).parent / 'output' / 'compilation_result.json'

    run_phase5(input_path, output_path)


if __name__ == '__main__':
    main()
