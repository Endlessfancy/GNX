"""
PEP Configuration for Latency Testing

PEP (Parallel Execution Plan) defines how stages are distributed across devices.

PEP Format:
    PEP = [ Block1, Block2, ... ]
    Block = [ [devices], [stages], [ratios] ]
        - devices: List of devices to use (e.g., ['CPU', 'GPU'])
        - stages: List of stage IDs to execute (1-7)
        - ratios: Data parallel split ratios (optional, default [1.0])

Stage Definitions (GraphSAGE 7-stage decomposition):
    Stage 1: GATHER - Neighbor feature gathering
    Stage 2: MESSAGE - Message computation (identity)
    Stage 3: REDUCE_SUM - Sum aggregation
    Stage 4: REDUCE_COUNT - Count neighbors
    Stage 5: NORMALIZE - Compute mean
    Stage 6: TRANSFORM - Linear transformation
    Stage 7: ACTIVATE - ReLU activation
"""

from typing import List, Dict, Any


# ============================================================================
# PEP Definitions (from executer/test_helper.py)
# ============================================================================

# PEP1: CPU+GPU data parallel for stages 1-5, NPU for stages 6-7
PEP1 = [
    [['CPU', 'GPU'], [1, 2, 3, 4, 5], [0.3, 0.7]],  # Block 0: CPU 30% + GPU 70%
    [['NPU'], [6, 7]]                                # Block 1: NPU 100%
]

# PEP2: CPU for stages 1-4, GPU+NPU data parallel for stages 5-7
PEP2 = [
    [['CPU'], [1, 2, 3, 4]],                         # Block 0: CPU 100%
    [['GPU', 'NPU'], [5, 6, 7], [0.7, 0.3]]          # Block 1: GPU 70% + NPU 30%
]

# PEP3: All stages on single device (baseline)
PEP_CPU_ONLY = [
    [['CPU'], [1, 2, 3, 4, 5, 6, 7]]
]

PEP_GPU_ONLY = [
    [['GPU'], [1, 2, 3, 4, 5, 6, 7]]
]

# PEP4: 3-block split
PEP_3BLOCK = [
    [['CPU'], [1, 2]],                               # Block 0: CPU - Gather+Message
    [['GPU'], [3, 4, 5]],                            # Block 1: GPU - Reduce+Normalize
    [['NPU'], [6, 7]]                                # Block 2: NPU - Transform+Activate
]

# PEP5: Fine-grained data parallel
PEP_FINE_DP = [
    [['CPU', 'GPU', 'NPU'], [1, 2, 3, 4, 5], [0.3, 0.4, 0.3]],  # 3-way data parallel
    [['GPU'], [6, 7]]
]


# ============================================================================
# Cluster/Execution Plan Definitions
# ============================================================================

def create_execution_plan(pep_configs: List[Dict]) -> Dict:
    """
    Create execution plan from PEP configurations.

    Args:
        pep_configs: List of PEP configuration dictionaries
            [
                {
                    'pep': [...],
                    'subgraph_ids': [0, 1, 2, ...]
                },
                ...
            ]

    Returns:
        execution_plan: Dict with 'clusters' and 'num_clusters'
    """
    clusters = []

    for i, config in enumerate(pep_configs):
        cluster = {
            'pep_key': f'pep_{i}',
            'pep': config['pep'],
            'subgraph_ids': config.get('subgraph_ids', [i]),
            'model_refs': {}
        }
        clusters.append(cluster)

    return {
        'clusters': clusters,
        'num_clusters': len(clusters)
    }


def get_two_pep_test_plan() -> Dict:
    """
    Get the standard two-PEP test plan (from executer).

    PEP1 (subgraphs 0-7):
        Block 0: CPU+GPU (50-50) execute stages 1-5
        Block 1: NPU execute stages 6-7

    PEP2 (subgraphs 8-15):
        Block 0: CPU execute stages 1-4
        Block 1: GPU+NPU (70-30) execute stages 5-7
    """
    return create_execution_plan([
        {
            'pep': PEP1,
            'subgraph_ids': [0, 1, 2, 3, 4, 5, 6, 7]
        },
        {
            'pep': PEP2,
            'subgraph_ids': [8, 9, 10, 11, 12, 13, 14, 15]
        }
    ])


def get_single_pep_test_plan(pep: List, num_subgraphs: int = 8) -> Dict:
    """
    Get a single-PEP test plan.

    Args:
        pep: PEP definition
        num_subgraphs: Number of subgraphs

    Returns:
        execution_plan
    """
    return create_execution_plan([
        {
            'pep': pep,
            'subgraph_ids': list(range(num_subgraphs))
        }
    ])


# ============================================================================
# PEP Analysis Utilities
# ============================================================================

def analyze_pep(pep: List) -> Dict:
    """
    Analyze a PEP configuration.

    Returns:
        Analysis dict with block info, devices used, stages per block, etc.
    """
    analysis = {
        'num_blocks': len(pep),
        'blocks': [],
        'all_devices': set(),
        'all_stages': set(),
        'has_data_parallel': False,
    }

    for i, block in enumerate(pep):
        devices = block[0]
        stages = block[1]
        ratios = block[2] if len(block) > 2 else [1.0]

        block_info = {
            'block_id': i,
            'devices': devices,
            'stages': stages,
            'ratios': ratios,
            'is_data_parallel': len(devices) > 1,
        }

        analysis['blocks'].append(block_info)
        analysis['all_devices'].update(devices)
        analysis['all_stages'].update(stages)

        if len(devices) > 1:
            analysis['has_data_parallel'] = True

    analysis['all_devices'] = list(analysis['all_devices'])
    analysis['all_stages'] = sorted(list(analysis['all_stages']))

    return analysis


def print_pep(pep: List, name: str = "PEP"):
    """Pretty print a PEP configuration."""
    print(f"\n{name}:")
    print("-" * 60)

    for i, block in enumerate(pep):
        devices = block[0]
        stages = block[1]
        ratios = block[2] if len(block) > 2 else [1.0]

        devices_str = '+'.join(devices)
        stages_str = ','.join(map(str, stages))

        if len(devices) > 1:
            ratios_str = ' (' + ', '.join(f"{d}:{r*100:.0f}%" for d, r in zip(devices, ratios)) + ')'
        else:
            ratios_str = ''

        print(f"  Block {i}: [{devices_str}] Stages [{stages_str}]{ratios_str}")

    print("-" * 60)


def print_execution_plan(plan: Dict):
    """Pretty print an execution plan."""
    print(f"\nExecution Plan: {plan['num_clusters']} cluster(s)")
    print("=" * 70)

    for cluster in plan['clusters']:
        print(f"\nCluster '{cluster['pep_key']}':")
        print(f"  Subgraphs: {cluster['subgraph_ids']}")
        print_pep(cluster['pep'], name="  PEP")


# ============================================================================
# Predefined Test Configurations
# ============================================================================

# All available PEP configurations for testing
ALL_PEPS = {
    'pep1': PEP1,
    'pep2': PEP2,
    'cpu_only': PEP_CPU_ONLY,
    'gpu_only': PEP_GPU_ONLY,
    '3block': PEP_3BLOCK,
    'fine_dp': PEP_FINE_DP,
}


if __name__ == "__main__":
    # Demo: print all PEP configurations
    print("=" * 70)
    print("PEP Configurations for Latency Testing")
    print("=" * 70)

    for name, pep in ALL_PEPS.items():
        print_pep(pep, name=name.upper())
        analysis = analyze_pep(pep)
        print(f"  Devices: {analysis['all_devices']}")
        print(f"  Stages: {analysis['all_stages']}")
        print(f"  Data Parallel: {analysis['has_data_parallel']}")
        print()

    print("\n" + "=" * 70)
    print("Standard Two-PEP Test Plan")
    print("=" * 70)
    plan = get_two_pep_test_plan()
    print_execution_plan(plan)
