#!/usr/bin/env python3
"""
Merge NPU isolated test results into a single file.

This script combines results from:
  results/npu_n5000.json
  results/npu_n10000.json
  ...
  results/npu_n100000.json

Into:
  results/block1_npu.json
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'results'

NODE_SIZES = [5000, 10000, 20000, 50000, 80000, 100000]


def merge_results():
    merged = {}
    total_success = 0
    total_failed = 0

    print("Merging NPU results...")
    print("-" * 50)

    for nodes in NODE_SIZES:
        result_file = RESULTS_DIR / f'npu_n{nodes}.json'

        if result_file.exists():
            with open(result_file, 'r') as f:
                node_results = json.load(f)

            success = sum(1 for r in node_results.values() if not r.get('failed', True))
            failed = sum(1 for r in node_results.values() if r.get('failed', True))

            print(f"  n{nodes}: {success} success, {failed} failed")

            merged.update(node_results)
            total_success += success
            total_failed += failed
        else:
            print(f"  n{nodes}: NOT FOUND (skipped)")

    print("-" * 50)
    print(f"Total: {total_success} success, {total_failed} failed")

    # Save merged results
    output_file = RESULTS_DIR / 'block1_npu.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged results saved to: {output_file}")

    # Generate boundary summary
    print("\n" + "=" * 50)
    print("NPU Boundary Analysis")
    print("=" * 50)

    for nodes in NODE_SIZES:
        # Find results for this node size
        node_results = {k: v for k, v in merged.items() if k.startswith(f"{nodes},")}

        if not node_results:
            print(f"  {nodes:>6} nodes: No data")
            continue

        # Sort by edge count
        sorted_results = sorted(node_results.items(), key=lambda x: int(x[0].split(',')[1]))

        success_edges = []
        failed_edges = []

        for key, result in sorted_results:
            edges = int(key.split(',')[1])
            if result.get('failed', True):
                failed_edges.append(edges)
            else:
                success_edges.append(edges)

        if success_edges and failed_edges:
            boundary = max(success_edges)
            print(f"  {nodes:>6} nodes: OK up to {boundary:>8} edges, FAIL at {min(failed_edges):>8}+ edges")
        elif success_edges:
            print(f"  {nodes:>6} nodes: All OK (max tested: {max(success_edges)} edges)")
        else:
            print(f"  {nodes:>6} nodes: All FAILED")

    return merged


if __name__ == '__main__':
    merge_results()
