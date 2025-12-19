"""
Test Compiler with Real Flickr Dataset
测试真实Flickr数据集的编译
"""

from compiler import GNNCompiler
from utils import CompilerConfig


def test_flickr_compilation():
    """
    使用真实Flickr数据集测试编译器（带METIS）

    Flickr Dataset:
    - Nodes: 89,250
    - Edges: 899,756
    - Feature dim: 500

    Constraints:
    - NPU Memory: 8 GB
    - NPU Padding Multiple: 1000
    - METIS: Minimize ghost nodes and cut edges
    """

    print("=" * 80)
    print("Flickr Dataset Compilation Test with METIS")
    print("=" * 80)

    # 配置参数
    # k值范围: [8, 12, 16, 20, 24, 30]
    # - k=8: 每个subgraph约11K节点，padding浪费小
    # - k=30: 每个subgraph约3K节点，平衡ghost node
    config = CompilerConfig(
        k_set=[8, 12, 16, 20, 24, 30],
        max_pipeline_blocks=2,
        top_k_peps=5,           # 保留更多候选PEP
        max_iterations=20,       # 允许更多迭代优化
        patience=5,
        convergence_threshold=0.5,
        use_metis=True,         # 使用METIS
        metis_ufactor=30,       # 3%不平衡度
        dataset_name='flickr',  # Flickr数据集
        verbose=True
    )

    print(f"\nCompiler Configuration:")
    print(f"  k values to test: {config.k_set}")
    print(f"  Using METIS: {config.use_metis} (ufactor={config.metis_ufactor})")
    print(f"  Max pipeline blocks: {config.max_pipeline_blocks}")
    print(f"  Top-K PEPs per subgraph: {config.top_k_peps}")
    print(f"  Max optimization iterations: {config.max_iterations}")

    # 创建编译器（自动加载Flickr数据集）
    compiler = GNNCompiler(config)

    # 执行编译（不再需要传入节点/边数）
    print(f"\n{'=' * 80}")
    print("Starting Compilation...")
    print(f"{'=' * 80}\n")

    result = compiler.compile()

    # 分析结果
    print(f"\n{'=' * 80}")
    print("Compilation Results Analysis")
    print(f"{'=' * 80}")

    partition_config = result['partition_config']
    execution_plan = result['execution_plan']
    statistics = result['statistics']

    best_k = partition_config['k']
    num_subgraphs = partition_config['num_subgraphs']

    print(f"\n✓ Best Configuration: k={best_k}")
    print(f"  Number of subgraphs: {num_subgraphs}")
    print(f"  Estimated Makespan: {statistics['makespan']:.2f} ms")
    print(f"  Number of clusters: {statistics['num_clusters']}")
    print(f"  Unique models generated: {statistics['num_unique_models']}")

    # 分析每个subgraph
    print(f"\n{'=' * 80}")
    print("Subgraph Analysis (Memory & Padding)")
    print(f"{'=' * 80}")

    subgraphs = partition_config['subgraphs']

    # 统计信息
    total_n = 0
    total_n_pad = 0
    total_m = 0
    total_m_pad = 0

    max_n_pad = 0
    max_m_pad = 0

    print(f"\n{'ID':<4} {'Nodes':<8} {'n_pad':<8} {'Edges':<8} {'m_pad':<8} "
          f"{'Cut':<8} {'n_waste%':<10} {'m_waste%':<10} {'Mem(MB)':<10}")
    print("-" * 90)

    total_cut_edges = 0

    for sg in subgraphs:
        sg_id = sg['id']
        n = sg['n']
        m = sg['m']
        n_pad = sg['n_pad']
        m_pad = sg['m_pad']
        cut_edges = sg.get('cut_edges', 0)  # METIS提供的cut edges

        # 计算浪费率
        n_waste = (n_pad - n) / n * 100 if n > 0 else 0
        m_waste = (m_pad - m) / m * 100 if m > 0 else 0

        # 估算NPU内存使用（简化模型）
        # 特征: n_pad × 500 × 4 bytes × 6 (input + output + 中间激活)
        memory_mb = (n_pad * 500 * 4 * 6) / (1024 * 1024)

        print(f"{sg_id:<4} {n:<8} {n_pad:<8} {m:<8} {m_pad:<8} {cut_edges:<8} "
              f"{n_waste:<10.2f} {m_waste:<10.2f} {memory_mb:<10.2f}")

        total_cut_edges += cut_edges

        total_n += n
        total_n_pad += n_pad
        total_m += m
        total_m_pad += m_pad

        max_n_pad = max(max_n_pad, n_pad)
        max_m_pad = max(max_m_pad, m_pad)

    print("-" * 90)

    # 总体统计
    avg_n = total_n / num_subgraphs
    avg_n_pad = total_n_pad / num_subgraphs
    avg_m = total_m / num_subgraphs
    avg_m_pad = total_m_pad / num_subgraphs

    overall_n_waste = (total_n_pad - total_n) / total_n * 100
    overall_m_waste = (total_m_pad - total_m) / total_m * 100

    print(f"\nStatistics:")
    print(f"  Average nodes per subgraph: {avg_n:.0f} (padding: {avg_n_pad:.0f})")
    print(f"  Average edges per subgraph: {avg_m:.0f} (padding: {avg_m_pad:.0f})")
    print(f"  Max padded nodes: {max_n_pad}")
    print(f"  Max padded edges: {max_m_pad}")
    print(f"  Overall node padding waste: {overall_n_waste:.2f}%")
    print(f"  Overall edge padding waste: {overall_m_waste:.2f}%")

    # Cut edges分析
    total_edges_in_graph = sum(sg['m'] for sg in subgraphs) + total_cut_edges // 2  # 近似

    print(f"\nCut Edges Analysis (METIS):")
    print(f"  Total cut edges: {total_cut_edges:,}")
    if total_edges_in_graph > 0:
        print(f"  Cut edge ratio: {total_cut_edges / total_edges_in_graph * 100:.2f}%")
    print(f"  Avg cut edges per subgraph: {total_cut_edges / num_subgraphs:.0f}")

    # 内存验证
    max_memory_mb = (max_n_pad * 500 * 4 * 6) / (1024 * 1024)
    npu_memory_gb = 8

    print(f"\nMemory Validation:")
    print(f"  Max NPU memory per subgraph: {max_memory_mb:.2f} MB")
    print(f"  NPU memory limit: {npu_memory_gb * 1024:.2f} MB")
    print(f"  Memory utilization: {max_memory_mb / (npu_memory_gb * 1024) * 100:.2f}%")

    if max_memory_mb < npu_memory_gb * 1024:
        print(f"  ✓ Memory constraint satisfied!")
    else:
        print(f"  ✗ WARNING: Memory constraint violated!")

    # Ghost node估算
    # Ghost nodes are represented by cut edges - nodes from other partitions
    original_total_nodes = 89250  # Flickr节点总数
    original_total_edges = 899756  # Flickr边总数

    print(f"\nGhost Node Analysis (METIS-based):")
    print(f"  Original total nodes: {original_total_nodes:,}")
    print(f"  Original total edges: {original_total_edges:,}")
    print(f"  Total nodes in subgraphs: {total_n:,}")
    print(f"  Note: Ghost nodes estimated from {total_cut_edges:,} cut edges")

    # Cluster分析
    print(f"\n{'=' * 80}")
    print("Cluster & PEP Analysis")
    print(f"{'=' * 80}")

    clusters = execution_plan['clusters']

    for i, cluster in enumerate(clusters):
        pep = cluster['pep']
        num_sgs = cluster['num_subgraphs']
        sg_ids = cluster['subgraph_ids']

        print(f"\nCluster {i}:")
        print(f"  Subgraphs: {num_sgs} (IDs: {sg_ids})")
        print(f"  PEP Structure:")

        for block_idx, block in enumerate(pep):
            devices, stages, ratios = block
            print(f"    Block {block_idx}: {devices} | Stages {stages} | Ratios {ratios}")

    # 性能指标
    print(f"\n{'=' * 80}")
    print("Performance Metrics")
    print(f"{'=' * 80}")

    print(f"\n  Estimated Makespan: {statistics['makespan']:.2f} ms")
    print(f"  Throughput: {original_total_nodes / statistics['makespan'] * 1000:.0f} nodes/sec")
    print(f"  Edges per second: {original_total_edges / statistics['makespan'] * 1000:.0f} edges/sec")

    # 验证
    print(f"\n{'=' * 80}")
    print("Validation Summary")
    print(f"{'=' * 80}")

    checks = []

    # Check 1: k值合理性
    if 4 <= best_k <= 50:
        checks.append(("✓", f"k={best_k} within reasonable range [4, 50]"))
    else:
        checks.append(("✗", f"k={best_k} outside recommended range"))

    # Check 2: Padding浪费率
    if overall_n_waste < 30:
        checks.append(("✓", f"Node padding waste {overall_n_waste:.1f}% < 30%"))
    else:
        checks.append(("⚠", f"Node padding waste {overall_n_waste:.1f}% high"))

    # Check 3: 内存约束
    if max_memory_mb < npu_memory_gb * 1024:
        checks.append(("✓", f"Memory usage {max_memory_mb:.0f}MB fits in {npu_memory_gb}GB NPU"))
    else:
        checks.append(("✗", f"Memory constraint violated"))

    # Check 4: Cluster数量
    if statistics['num_clusters'] <= best_k:
        checks.append(("✓", f"{statistics['num_clusters']} clusters ≤ {best_k} subgraphs"))
    else:
        checks.append(("⚠", f"More clusters than subgraphs"))

    # Check 5: Makespan合理性
    if statistics['makespan'] > 0:
        checks.append(("✓", f"Makespan {statistics['makespan']:.2f}ms > 0"))
    else:
        checks.append(("✗", f"Invalid makespan"))

    for symbol, msg in checks:
        print(f"  {symbol} {msg}")

    print(f"\n{'=' * 80}")
    print("Test Complete!")
    print(f"{'=' * 80}\n")

    return result


if __name__ == '__main__':
    result = test_flickr_compilation()
