#!/usr/bin/env python3
"""
Phase 3: Pipeline-Aware Global Optimization

Optimize PEP assignment for all subgraphs to minimize total makespan.
Uses clustering, Johnson's Rule sorting, and iterative bottleneck reduction.

Usage:
    python phase3_optimization.py --input output/phase2_pep_candidates.json
    python phase3_optimization.py --input output/phase2_pep_candidates.json --output output/phase3_optimization.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import CompilerConfig, ProfilingLoader
from core import CostEstimator, GlobalOptimizer
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


def pep_to_dict(pep):
    """PEP -> dict"""
    return {
        'blocks': [
            {
                'devices': block.devices,
                'stages': block.stages,
                'ratios': block.ratios
            }
            for block in pep.blocks
        ]
    }


def subgraph_to_dict(sg):
    """Subgraph -> dict"""
    return {
        'id': sg.id, 'n': sg.n, 'm': sg.m,
        'n_pad': sg.n_pad, 'm_pad': sg.m_pad,
        'cut_edges': sg.cut_edges
    }


# ============================================================================
# Phase 3 Main
# ============================================================================

def run_phase3(input_path: Path, output_path: Path, config: CompilerConfig = None):
    """
    Run Phase 3: Pipeline-Aware Global Optimization

    Args:
        input_path: Phase 2 output JSON path
        output_path: Output JSON path
        config: Compiler config (optional)

    Returns:
        Phase 3 result dict
    """
    print("=" * 60)
    print("Phase 3: Pipeline-Aware Global Optimization")
    print("=" * 60)

    # Load Phase 2 output
    print(f"\nLoading Phase 2 output: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        phase2_data = json.load(f)

    config = config or CompilerConfig()

    # Initialize modules
    profiling = ProfilingLoader(config.profiling_dir)
    cost_estimator = CostEstimator(profiling, config)
    global_optimizer = GlobalOptimizer(cost_estimator, config)

    best_k = None
    best_makespan = float('inf')
    best_result = None

    t0_total = time.perf_counter()

    for k_str, k_data in phase2_data['k_results'].items():
        k = int(k_str)
        print(f"\n--- Optimizing K = {k} ---")

        # Reconstruct subgraphs
        subgraphs = [dict_to_subgraph(d) for d in k_data['subgraphs']]

        # Reconstruct top_k_peps: {sg_id: [(pep, cost), ...]}
        top_k_peps = {}
        for sg_id_str, pep_list in k_data['top_k_peps'].items():
            sg_id = int(sg_id_str)
            top_k_peps[sg_id] = [
                (dict_to_pep(entry['pep']), entry['cost'])
                for entry in pep_list
            ]

        # Run optimization
        try:
            assignment, clusters, makespan = global_optimizer.optimize(
                subgraphs, top_k_peps
            )
        except Exception as e:
            print(f"  Optimization failed for K={k}: {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"  Makespan: {makespan:.2f}ms")

        if makespan < best_makespan:
            best_makespan = makespan
            best_k = k

            # Serialize assignment and clusters
            assignment_serialized = {}
            for sg_id, (pep, cost) in assignment.items():
                assignment_serialized[str(sg_id)] = {
                    'pep': pep_to_dict(pep),
                    'cost': cost
                }

            clusters_serialized = {}
            for cluster_key, sg_list in clusters.items():
                clusters_serialized[cluster_key] = [subgraph_to_dict(sg) for sg in sg_list]

            best_result = {
                'k': best_k,
                'makespan': best_makespan,
                'assignment': assignment_serialized,
                'clusters': clusters_serialized,
                'subgraphs': [subgraph_to_dict(sg) for sg in subgraphs]
            }

    total_time = time.perf_counter() - t0_total

    if best_result is None:
        print("\nAll optimization attempts failed!")
        return None

    # Build final output
    result = {
        'phase': 3,
        'dataset': phase2_data['dataset'],
        'num_nodes': phase2_data['num_nodes'],
        'num_edges': phase2_data['num_edges'],
        'optimization_time_sec': total_time,
        'best_k': best_k,
        'best_makespan': best_makespan,
        **best_result
    }

    print(f"\n{'='*60}")
    print(f"Best K = {best_k}, Makespan = {best_makespan:.2f}ms")
    print(f"Optimization time: {total_time:.2f}s")
    print(f"{'='*60}")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"Phase 3 output saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Global Optimization")
    parser.add_argument('--input', type=str, default='output/phase2_pep_candidates.json',
                        help='Phase 2 output JSON path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--max-iterations', type=int, default=20,
                        help='Max optimization iterations')
    parser.add_argument('--profiling-dir', type=str, default=None,
                        help='Profiling results directory')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else \
        Path(__file__).parent / 'output' / 'phase3_optimization.json'

    config = CompilerConfig()
    config.max_iterations = args.max_iterations
    if args.profiling_dir:
        config.profiling_dir = Path(args.profiling_dir)

    run_phase3(input_path, output_path, config)


if __name__ == '__main__':
    main()
