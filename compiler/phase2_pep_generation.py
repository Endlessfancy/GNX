#!/usr/bin/env python3
"""
Phase 2: PEP Generation & Cost Estimation

Generate candidate Parallel Execution Plans (PEPs) for each subgraph
and estimate their execution cost using profiling data.

Usage:
    python phase2_pep_generation.py --input output/phase1_partition.json
    python phase2_pep_generation.py --input output/phase1_partition.json --output output/phase2_pep_candidates.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import CompilerConfig, ProfilingLoader
from core import PEPGenerator, CostEstimator
from core.graph_partitioner import Subgraph
from core.pep_generator import PEP, PEPBlock


# ============================================================================
# Serialization Helpers
# ============================================================================

def dict_to_subgraph(d):
    """dict -> Subgraph"""
    return Subgraph(
        id=d['id'],
        n=d['n'],
        m=d['m'],
        n_pad=d['n_pad'],
        m_pad=d['m_pad'],
        cut_edges=d.get('cut_edges', 0)
    )


def pep_to_dict(pep):
    """PEP -> dict for JSON serialization"""
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


def subgraph_to_dict(sg):
    """Subgraph -> dict"""
    return {
        'id': sg.id,
        'n': sg.n,
        'm': sg.m,
        'n_pad': sg.n_pad,
        'm_pad': sg.m_pad,
        'cut_edges': sg.cut_edges
    }


# ============================================================================
# Phase 2 Main
# ============================================================================

def run_phase2(input_path: Path, output_path: Path, config: CompilerConfig = None):
    """
    Run Phase 2: PEP Generation & Cost Estimation

    Args:
        input_path: Phase 1 output JSON path
        output_path: Output JSON path
        config: Compiler config (optional)

    Returns:
        Phase 2 result dict
    """
    print("=" * 60)
    print("Phase 2: PEP Generation & Cost Estimation")
    print("=" * 60)

    # Load Phase 1 output
    print(f"\nLoading Phase 1 output: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        phase1_data = json.load(f)

    config = config or CompilerConfig()

    # Initialize modules
    profiling = ProfilingLoader(config.profiling_dir)
    pep_generator = PEPGenerator(config)
    cost_estimator = CostEstimator(profiling, config)

    result = {
        'phase': 2,
        'dataset': phase1_data['dataset'],
        'num_nodes': phase1_data['num_nodes'],
        'num_edges': phase1_data['num_edges'],
        'top_k_peps_count': config.top_k_peps,
        'k_results': {}
    }

    t0_total = time.perf_counter()

    for k_str, sg_dicts in phase1_data['k_results'].items():
        k = int(k_str)
        print(f"\n--- K = {k} ({len(sg_dicts)} subgraphs) ---")

        subgraphs = [dict_to_subgraph(d) for d in sg_dicts]
        k_result = {
            'subgraphs': sg_dicts,
            'top_k_peps': {}
        }

        for sg in subgraphs:
            # Generate all candidate PEPs
            candidates = pep_generator.generate_candidates(sg)

            if not candidates:
                print(f"  Warning: No valid PEP for subgraph {sg.id}")
                continue

            # Estimate cost for each candidate
            pep_costs = []
            for pep in candidates:
                cost_info = cost_estimator.estimate_pep_cost(pep, sg)
                pep_costs.append((pep, cost_info['total_time'], cost_info))

            # Sort by cost
            pep_costs.sort(key=lambda x: x[1])

            # Keep Top-K
            top_k = pep_costs[:config.top_k_peps]

            k_result['top_k_peps'][str(sg.id)] = [
                {
                    'pep': pep_to_dict(pep),
                    'cost': cost,
                    'breakdown': {
                        'compute': info['breakdown']['compute'],
                        'transfer': info['breakdown']['transfer'],
                        'block_times': info['block_times'],
                        'transfer_times': info['transfer_times']
                    }
                }
                for pep, cost, info in top_k
            ]

            if config.verbose:
                print(f"  SG {sg.id} (n={sg.n}, m={sg.m}): "
                      f"{len(candidates)} candidates -> Top-{len(top_k)}, "
                      f"best={top_k[0][1]:.2f}ms")

        result['k_results'][k_str] = k_result

    total_time = time.perf_counter() - t0_total
    result['generation_time_sec'] = total_time
    print(f"\nTotal PEP generation time: {total_time:.2f}s")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"Phase 2 output saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 2: PEP Generation & Cost Estimation")
    parser.add_argument('--input', type=str, default='output/phase1_partition.json',
                        help='Phase 1 output JSON path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top PEP candidates to keep')
    parser.add_argument('--profiling-dir', type=str, default=None,
                        help='Profiling results directory')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else \
        Path(__file__).parent / 'output' / 'phase2_pep_candidates.json'

    config = CompilerConfig()
    config.top_k_peps = args.top_k
    if args.profiling_dir:
        config.profiling_dir = Path(args.profiling_dir)

    run_phase2(input_path, output_path, config)


if __name__ == '__main__':
    main()
