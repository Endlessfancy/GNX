"""
GNX Primitive Profiling Script

This script profiles the 7 universal primitives defined in the common primitive library,
replacing the need to profile ~20 individual stages across SAGE, GAT, and GCN models.

Primitives to Profile (7 total):
    - P1_MATMUL: Matrix multiplication (100% reuse)
    - P2_GATHER: Index selection (100% reuse)
    - P3_SCATTER_ADD: Scatter aggregation (100% reuse)
    - P4_ELEWISE_MUL: Element-wise multiplication (100% reuse)
    - P5_ELEWISE_ACT: Element-wise activation (100% reuse)
    - P6_GAT_EDGE_ATT: GAT attention score (GAT only)
    - P7_GAT_SOFTMAX: GAT edge softmax (GAT only)

Profiling Efficiency:
    - Original (stage-based): 7 + 7 + 6 = 20 stages
    - New (primitive-based): 7 primitives
    - Reduction: ~65% less profiling work

Usage:
    python profile_primitives.py --all              # Run full profiling
    python profile_primitives.py --export           # Export primitive models only
    python profile_primitives.py --measure          # Measure latencies
    python profile_primitives.py --analyze          # Generate results
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

# Import primitives
from models.primitives import (
    P1_Matmul, P1_Matmul_Dual,
    P2_Gather,
    P3_ScatterAdd, P3_ScatterAdd_Count,
    P4_ElewiseMul, P4_ElewiseDiv,
    P5_ElewiseAct,
    P6_GATEdgeAtt,
    P7_GATSoftmax,
    PRIMITIVE_REGISTRY,
    PROFILING_PRIMITIVES,
)

# ============================================================================
# Configuration
# ============================================================================

PROFILING_DIR = Path(__file__).parent
MODELS_DIR = PROFILING_DIR / 'exported_primitives'
RESULTS_DIR = PROFILING_DIR / 'results'
TEST_CASES_FILE = PROFILING_DIR / 'test_cases.json'

# Feature dimensions to test
FEATURE_DIMS = [64, 128, 256, 500]
DEFAULT_FEATURE_DIM = 500

# ============================================================================
# Primitive Model Factory
# ============================================================================

def get_primitive_model(primitive_id: str, **kwargs):
    """
    Get a primitive model instance.

    Args:
        primitive_id: One of P1-P7 identifiers
        **kwargs: Primitive-specific parameters

    Returns:
        nn.Module: The primitive model
    """
    feature_dim = kwargs.get('feature_dim', DEFAULT_FEATURE_DIM)
    num_nodes = kwargs.get('num_nodes', None)

    primitives = {
        'P1_MATMUL': lambda: P1_Matmul(feature_dim, feature_dim),
        'P1_MATMUL_DUAL': lambda: P1_Matmul_Dual(feature_dim, feature_dim),
        'P2_GATHER': lambda: P2_Gather(),
        'P3_SCATTER_ADD': lambda: P3_ScatterAdd(num_nodes_static=num_nodes),
        'P3_SCATTER_ADD_COUNT': lambda: P3_ScatterAdd_Count(),
        'P4_ELEWISE_MUL': lambda: P4_ElewiseMul(),
        'P4_ELEWISE_DIV': lambda: P4_ElewiseDiv(),
        'P5_ELEWISE_ACT': lambda: P5_ElewiseAct(activation='relu'),
        'P6_GAT_EDGE_ATT': lambda: P6_GATEdgeAtt(feature_dim),
        'P7_GAT_SOFTMAX': lambda: P7_GATSoftmax(num_nodes_static=num_nodes),
    }

    if primitive_id not in primitives:
        raise ValueError(f"Unknown primitive: {primitive_id}")

    return primitives[primitive_id]()


def generate_primitive_input(primitive_id: str, num_nodes: int, num_edges: int,
                             feature_dim: int = DEFAULT_FEATURE_DIM):
    """
    Generate dummy input for a primitive.

    Args:
        primitive_id: One of P1-P7 identifiers
        num_nodes: Number of nodes
        num_edges: Number of edges
        feature_dim: Feature dimension

    Returns:
        tuple or tensor: Input data for the primitive
    """
    torch.manual_seed(42)

    if primitive_id == 'P1_MATMUL':
        # Input: x [N, F]
        return (torch.randn(num_nodes, feature_dim),)

    elif primitive_id == 'P1_MATMUL_DUAL':
        # Input: agg [N, F], x [N, F]
        return (torch.randn(num_nodes, feature_dim),
                torch.randn(num_nodes, feature_dim))

    elif primitive_id == 'P2_GATHER':
        # Input: x [N, F], indices [E]
        x = torch.randn(num_nodes, feature_dim)
        indices = torch.randint(0, num_nodes, (num_edges,))
        return (x, indices)

    elif primitive_id == 'P3_SCATTER_ADD':
        # Input: src [E, F], indices [E], num_nodes
        src = torch.randn(num_edges, feature_dim)
        indices = torch.randint(0, num_nodes, (num_edges,))
        return (src, indices, num_nodes)

    elif primitive_id == 'P3_SCATTER_ADD_COUNT':
        # Input: indices [E], num_nodes, num_edges
        indices = torch.randint(0, num_nodes, (num_edges,))
        return (indices, num_nodes, num_edges)

    elif primitive_id == 'P4_ELEWISE_MUL':
        # Input: scalar [E], tensor [E, F]
        scalar = torch.rand(num_edges)
        tensor = torch.randn(num_edges, feature_dim)
        return (scalar, tensor)

    elif primitive_id == 'P4_ELEWISE_DIV':
        # Input: tensor [N, F], divisor [N]
        tensor = torch.randn(num_nodes, feature_dim)
        divisor = torch.rand(num_nodes) + 1.0
        return (tensor, divisor)

    elif primitive_id == 'P5_ELEWISE_ACT':
        # Input: x [N, F]
        return (torch.randn(num_nodes, feature_dim),)

    elif primitive_id == 'P6_GAT_EDGE_ATT':
        # Input: Wx_i [E, F], Wx_j [E, F]
        Wx_i = torch.randn(num_edges, feature_dim)
        Wx_j = torch.randn(num_edges, feature_dim)
        return (Wx_i, Wx_j)

    elif primitive_id == 'P7_GAT_SOFTMAX':
        # Input: e [E], edge_index [2, E], num_nodes
        e = torch.randn(num_edges)
        edge_index = torch.stack([
            torch.randint(0, num_nodes, (num_edges,)),  # src
            torch.randint(0, num_nodes, (num_edges,)),  # dst
        ])
        return (e, edge_index, num_nodes)

    else:
        raise ValueError(f"Unknown primitive: {primitive_id}")


def estimate_primitive_data_size(primitive_id: str, num_nodes: int, num_edges: int,
                                  feature_dim: int = DEFAULT_FEATURE_DIM) -> int:
    """Estimate input + output data size in bytes."""
    bytes_per_float = 4
    bytes_per_int = 8

    if primitive_id == 'P1_MATMUL':
        # Input: [N, F], Output: [N, F]
        return 2 * num_nodes * feature_dim * bytes_per_float

    elif primitive_id == 'P1_MATMUL_DUAL':
        # Input: 2 × [N, F], Output: [N, F]
        return 3 * num_nodes * feature_dim * bytes_per_float

    elif primitive_id == 'P2_GATHER':
        # Input: [N, F] + [E], Output: [E, F]
        return (num_nodes * feature_dim + num_edges * feature_dim) * bytes_per_float + num_edges * bytes_per_int

    elif primitive_id == 'P3_SCATTER_ADD':
        # Input: [E, F] + [E], Output: [N, F]
        return (num_edges * feature_dim + num_nodes * feature_dim) * bytes_per_float + num_edges * bytes_per_int

    elif primitive_id == 'P3_SCATTER_ADD_COUNT':
        # Input: [E], Output: [N]
        return num_edges * bytes_per_int + num_nodes * bytes_per_float

    elif primitive_id == 'P4_ELEWISE_MUL':
        # Input: [E] + [E, F], Output: [E, F]
        return (num_edges + 2 * num_edges * feature_dim) * bytes_per_float

    elif primitive_id == 'P4_ELEWISE_DIV':
        # Input: [N, F] + [N], Output: [N, F]
        return (2 * num_nodes * feature_dim + num_nodes) * bytes_per_float

    elif primitive_id == 'P5_ELEWISE_ACT':
        # Input: [N, F], Output: [N, F]
        return 2 * num_nodes * feature_dim * bytes_per_float

    elif primitive_id == 'P6_GAT_EDGE_ATT':
        # Input: 2 × [E, F], Output: [E]
        return (2 * num_edges * feature_dim + num_edges) * bytes_per_float

    elif primitive_id == 'P7_GAT_SOFTMAX':
        # Input: [E] + [2, E], Output: [E]
        return 2 * num_edges * bytes_per_float + 2 * num_edges * bytes_per_int

    else:
        return 0


# ============================================================================
# Latency Measurement
# ============================================================================

def measure_latency_pytorch(model, dummy_input, num_warmup=10, num_iterations=50):
    """Measure latency using PyTorch."""
    model.eval()

    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            if isinstance(dummy_input, tuple):
                _ = model(*dummy_input)
            else:
                _ = model(dummy_input)

        # Measure
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            if isinstance(dummy_input, tuple):
                _ = model(*dummy_input)
            else:
                _ = model(dummy_input)
            latencies.append((time.perf_counter() - start) * 1000)

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
    }


# ============================================================================
# Main Profiling Functions
# ============================================================================

def profile_all_primitives(test_cases, feature_dim=DEFAULT_FEATURE_DIM,
                           num_warmup=10, num_iterations=50):
    """
    Profile all primitives across test cases.

    Args:
        test_cases: List of {nodes, edges} dicts
        feature_dim: Feature dimension
        num_warmup: Warmup iterations
        num_iterations: Measurement iterations

    Returns:
        dict: Profiling results
    """
    print("=" * 70)
    print("Primitive Profiling")
    print("=" * 70)
    print(f"Primitives: {len(PROFILING_PRIMITIVES)}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Feature dim: {feature_dim}")
    print()

    results = {}
    total = len(PROFILING_PRIMITIVES) * len(test_cases)
    count = 0

    for primitive_id in PROFILING_PRIMITIVES:
        props = PRIMITIVE_REGISTRY.get(primitive_id, {})
        npu_ok = '✅' if props.get('npu_compatible', True) else '❌'

        print(f"\n{primitive_id} (NPU: {npu_ok})")
        print("-" * 40)

        for case in test_cases:
            count += 1
            nodes, edges = case['nodes'], case['edges']
            print(f"  [{count}/{total}] n={nodes}, e={edges}... ", end='', flush=True)

            try:
                # Get model and input
                model = get_primitive_model(
                    primitive_id,
                    feature_dim=feature_dim,
                    num_nodes=nodes
                )
                dummy_input = generate_primitive_input(
                    primitive_id, nodes, edges, feature_dim
                )

                # Measure latency
                result = measure_latency_pytorch(
                    model, dummy_input, num_warmup, num_iterations
                )

                # Calculate data size
                data_size = estimate_primitive_data_size(
                    primitive_id, nodes, edges, feature_dim
                )
                result['data_size_bytes'] = data_size

                # Store result
                key = f"{primitive_id},{nodes},{edges},{feature_dim}"
                results[key] = result

                print(f"{result['mean']:.3f}ms ±{result['std']:.3f}")

            except Exception as e:
                print(f"Failed: {e}")
                key = f"{primitive_id},{nodes},{edges},{feature_dim}"
                results[key] = {'failed': True, 'error': str(e)}

    print(f"\n✓ Profiled {len(results)} configurations")
    return results


def save_primitive_results(results, output_file=None):
    """Save profiling results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_file is None:
        output_file = RESULTS_DIR / 'primitive_lookup_table.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved results to: {output_file}")
    return output_file


def generate_primitive_report(results):
    """Generate a summary report."""
    report = []
    report.append("=" * 80)
    report.append("GNX Primitive Profiling Report")
    report.append("=" * 80)
    report.append("")

    # Summary by primitive
    report.append("Average Latency by Primitive (ms):")
    report.append("-" * 80)

    for primitive_id in PROFILING_PRIMITIVES:
        latencies = [v['mean'] for k, v in results.items()
                    if k.startswith(primitive_id) and 'mean' in v]
        if latencies:
            avg = np.mean(latencies)
            props = PRIMITIVE_REGISTRY.get(primitive_id, {})
            npu = '✅' if props.get('npu_compatible', True) else '❌'
            report.append(f"  {primitive_id:<20} {avg:>8.3f}ms  NPU: {npu}")

    report.append("")
    report.append("=" * 80)

    # Print and save
    report_text = '\n'.join(report)
    print(report_text)

    report_file = RESULTS_DIR / 'primitive_profiling_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n✓ Report saved to: {report_file}")


def build_model_cost_table(primitive_results):
    """
    Build per-model cost estimation tables from primitive results.

    This allows estimating SAGE/GAT/GCN costs by combining primitive costs.
    """
    # Extract unique configurations
    configs = {}
    for key, value in primitive_results.items():
        if 'failed' in value:
            continue
        parts = key.split(',')
        primitive, nodes, edges, feat = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
        config_key = (nodes, edges, feat)
        if config_key not in configs:
            configs[config_key] = {}
        configs[config_key][primitive] = value['mean']

    # Build model cost tables
    model_costs = {
        'SAGE': {},
        'GAT': {},
        'GCN': {},
    }

    for (nodes, edges, feat), primitives in configs.items():
        key = f"{nodes},{edges},{feat}"

        # SAGE cost: P2 + P3 + P3_COUNT + P4_DIV + P1_DUAL + P5
        sage_cost = (
            primitives.get('P2_GATHER', 0) +
            primitives.get('P3_SCATTER_ADD', 0) * 1.1 +  # approximate count variant
            primitives.get('P4_ELEWISE_MUL', 0) +  # approximate div
            primitives.get('P1_MATMUL', 0) * 2 +  # dual linear
            primitives.get('P5_ELEWISE_ACT', 0)
        )
        model_costs['SAGE'][key] = sage_cost

        # GAT cost: P1 + 2×P2 + P6 + P7 + P4 + P3 + P5
        gat_cost = (
            primitives.get('P1_MATMUL', 0) +
            primitives.get('P2_GATHER', 0) * 2 +  # GATHER_BOTH = 2× GATHER
            primitives.get('P6_GAT_EDGE_ATT', 0) +
            primitives.get('P7_GAT_SOFTMAX', 0) +
            primitives.get('P4_ELEWISE_MUL', 0) +
            primitives.get('P3_SCATTER_ADD', 0) +
            primitives.get('P5_ELEWISE_ACT', 0)
        )
        model_costs['GAT'][key] = gat_cost

        # GCN cost: P2 + P4 + P3 + P1 + P5 (COMPUTE_NORM is preprocessing)
        gcn_cost = (
            primitives.get('P2_GATHER', 0) +
            primitives.get('P4_ELEWISE_MUL', 0) +
            primitives.get('P3_SCATTER_ADD', 0) +
            primitives.get('P1_MATMUL', 0) +
            primitives.get('P5_ELEWISE_ACT', 0)
        )
        model_costs['GCN'][key] = gcn_cost

    # Save model cost tables
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / 'model_cost_from_primitives.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(model_costs, f, indent=2)

    print(f"✓ Model cost tables saved to: {output_file}")
    return model_costs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='GNX Primitive Profiling Script')
    parser.add_argument('--all', action='store_true', help='Run full profiling')
    parser.add_argument('--measure', action='store_true', help='Measure latencies')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing results')
    parser.add_argument('--feature-dim', type=int, default=DEFAULT_FEATURE_DIM,
                       help='Feature dimension')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=50, help='Measurement iterations')

    args = parser.parse_args()

    # Load test cases
    if TEST_CASES_FILE.exists():
        with open(TEST_CASES_FILE, 'r') as f:
            config = json.load(f)
            test_cases = config['test_cases']
    else:
        # Default test cases
        test_cases = [
            {'nodes': 1000, 'edges': 5000},
            {'nodes': 2000, 'edges': 10000},
            {'nodes': 5000, 'edges': 25000},
            {'nodes': 10000, 'edges': 50000},
        ]

    print("=" * 70)
    print("GNX Primitive Profiling")
    print("=" * 70)
    print(f"Primitives to profile: {len(PROFILING_PRIMITIVES)}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Feature dimension: {args.feature_dim}")
    print()

    if args.all or args.measure:
        # Run profiling
        results = profile_all_primitives(
            test_cases,
            feature_dim=args.feature_dim,
            num_warmup=args.warmup,
            num_iterations=args.iterations
        )

        # Save results
        save_primitive_results(results)

        # Generate report
        generate_primitive_report(results)

        # Build model cost tables
        build_model_cost_table(results)

    elif args.analyze:
        # Load existing results
        results_file = RESULTS_DIR / 'primitive_lookup_table.json'
        if not results_file.exists():
            print(f"ERROR: Results file not found: {results_file}")
            print("Run: python profile_primitives.py --measure")
            return

        with open(results_file, 'r') as f:
            results = json.load(f)

        generate_primitive_report(results)
        build_model_cost_table(results)

    else:
        parser.print_help()
        print("\n" + "-" * 70)
        print("Primitive Library Summary:")
        print("-" * 70)
        for prim_id in PROFILING_PRIMITIVES:
            props = PRIMITIVE_REGISTRY.get(prim_id, {})
            npu = '✅' if props.get('npu_compatible', True) else '❌'
            used_by = ', '.join(props.get('used_by', []))
            print(f"  {prim_id:<20} NPU: {npu}  Used by: {used_by}")


if __name__ == '__main__':
    main()
