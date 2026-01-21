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

Note: Block 1 (stages 5-7) is edge-independent, so we only have
      one result per node size (6 total), not per node×edge combination.
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

            # New format: key is "{nodes},NPU,block1"
            key = f"{nodes},NPU,block1"

            if key in node_results:
                result = node_results[key]
                merged[key] = result

                if result.get('failed', True):
                    print(f"  n{nodes}: FAILED - {result.get('error', 'unknown')[:30]}")
                    total_failed += 1
                else:
                    print(f"  n{nodes}: {result['mean']:.2f}ms")
                    total_success += 1
            else:
                # Try old format for backward compatibility
                old_results = [v for k, v in node_results.items() if k.startswith(f"{nodes},")]
                if old_results:
                    # Use first successful result
                    for r in old_results:
                        if not r.get('failed', True):
                            merged[key] = r
                            print(f"  n{nodes}: {r['mean']:.2f}ms (migrated)")
                            total_success += 1
                            break
                    else:
                        merged[key] = old_results[0]
                        print(f"  n{nodes}: FAILED (migrated)")
                        total_failed += 1
                else:
                    print(f"  n{nodes}: No valid data found")
                    total_failed += 1
        else:
            print(f"  n{nodes}: NOT FOUND (skipped)")

    print("-" * 50)
    print(f"Total: {total_success} success, {total_failed} failed")

    # Save merged results
    output_file = RESULTS_DIR / 'block1_npu.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged results saved to: {output_file}")

    # Generate summary
    print("\n" + "=" * 50)
    print("NPU Latency Summary (by node size)")
    print("=" * 50)

    for nodes in NODE_SIZES:
        key = f"{nodes},NPU,block1"
        result = merged.get(key, {})

        if result.get('failed', True):
            print(f"  {nodes:>6} nodes: FAILED")
        else:
            print(f"  {nodes:>6} nodes: {result['mean']:>8.2f}ms (std={result.get('std', 0):.2f})")

    return merged


if __name__ == '__main__':
    merge_results()
