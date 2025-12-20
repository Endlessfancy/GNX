"""
Test Helper Functions

Helper functions to create custom execution plans for testing pipeline execution.
"""

from typing import List, Dict


def create_custom_execution_plan(pep_configs: List[Dict]) -> Dict:
    """
    Create custom execution plan from PEP configurations.

    Args:
        pep_configs: List of PEP configuration dictionaries
        Example:
            [
                {
                    'pep': [
                        [['CPU', 'GPU'], [1, 2, 3, 4, 5], [0.5, 0.5]],
                        [['NPU'], [6, 7]]
                    ],
                    'subgraph_ids': [0, 1, 2, 3, 4, 5, 6, 7]
                },
                ...
            ]

    Returns:
        execution_plan: Dict with 'clusters' and 'num_clusters'
    """
    clusters = []

    for i, config in enumerate(pep_configs):
        cluster = {
            'pep_key': f'custom_pep_{i}',
            'pep': config['pep'],
            'subgraph_ids': config['subgraph_ids'],
            'model_refs': {}  # Will be filled by ModelManager
        }
        clusters.append(cluster)

    return {
        'clusters': clusters,
        'num_clusters': len(clusters)
    }


def create_two_pep_test_plan() -> Dict:
    """
    Create the specific test plan with two custom PEPs as requested.

    PEP1 (subgraphs 0-7):
        Block 0: CPU+GPU (50-50) execute stages 1-5
        Block 1: NPU execute stages 6-7

    PEP2 (subgraphs 8-15):
        Block 0: CPU execute stages 1-4
        Block 1: GPU+NPU (70-30) execute stages 5-7

    Returns:
        execution_plan: Custom execution plan
    """
    pep_configs = [
        {
            'pep': [
                [['CPU', 'GPU'], [1, 2, 3, 4, 5], [0.5, 0.5]],
                [['NPU'], [6, 7]]
            ],
            'subgraph_ids': [0, 1, 2, 3, 4, 5, 6, 7]
        },
        {
            'pep': [
                [['CPU'], [1, 2, 3, 4]],
                [['GPU', 'NPU'], [5, 6, 7], [0.7, 0.3]]
            ],
            'subgraph_ids': [8, 9, 10, 11, 12, 13, 14, 15]
        }
    ]

    return create_custom_execution_plan(pep_configs)
