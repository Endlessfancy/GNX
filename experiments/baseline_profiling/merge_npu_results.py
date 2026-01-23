#!/usr/bin/env python3
"""
Merge NPU isolated test results into a single file.

This script combines results from individual NPU tests:
  results/npu_graphsage_n5000_e50000.json
  results/npu_gcn_n5000_e50000.json
  ...

Into:
  results/npu_results.json
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'results'
TEST_CASES_FILE = SCRIPT_DIR / 'test_cases.json'

MODEL_NAMES = ['graphsage', 'gcn', 'gat']


def load_config():
    with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_results():
    config = load_config()
    test_cases = config['test_cases']

    merged = {}
    total_success = 0
    total_failed = 0
    total_missing = 0

    print("Merging NPU results...")
    print("=" * 60)

    for model_name in MODEL_NAMES:
        print(f"\n--- {model_name.upper()} ---")

        for case in test_cases:
            nodes, edges = case['nodes'], case['edges']
            result_file = RESULTS_DIR / f'npu_{model_name}_n{nodes}_e{edges}.json'
            key = f"{model_name},{nodes},{edges},NPU"

            if result_file.exists():
                with open(result_file, 'r') as f:
                    node_results = json.load(f)

                if key in node_results:
                    result = node_results[key]
                    merged[key] = result

                    if result.get('failed', True):
                        print(f"  [{nodes}n, {edges}e]: FAILED - {result.get('error', 'unknown')[:30]}")
                        total_failed += 1
                    else:
                        print(f"  [{nodes}n, {edges}e]: {result['mean']:.2f}ms")
                        total_success += 1
                else:
                    print(f"  [{nodes}n, {edges}e]: Invalid data")
                    total_missing += 1
            else:
                print(f"  [{nodes}n, {edges}e]: NOT FOUND")
                total_missing += 1

    print("\n" + "=" * 60)
    print(f"Total: {total_success} success, {total_failed} failed, {total_missing} missing")

    # Save merged results
    output_file = RESULTS_DIR / 'npu_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged results saved to: {output_file}")

    # Generate summary by model
    print("\n" + "=" * 60)
    print("NPU Latency Summary (by model)")
    print("=" * 60)

    for model_name in MODEL_NAMES:
        model_results = [v['mean'] for k, v in merged.items()
                        if k.startswith(f"{model_name},") and not v.get('failed', True)]

        if model_results:
            import numpy as np
            print(f"  {model_name.upper():>10}: mean={np.mean(model_results):.2f}ms, "
                  f"min={np.min(model_results):.2f}ms, max={np.max(model_results):.2f}ms "
                  f"({len(model_results)}/{len(test_cases)} tests)")
        else:
            print(f"  {model_name.upper():>10}: No successful tests")

    return merged


if __name__ == '__main__':
    merge_results()
