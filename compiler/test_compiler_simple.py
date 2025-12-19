"""
Simple Test for GNN Compiler
"""

from compiler import GNNCompiler
from utils import CompilerConfig


def test_basic_compilation():
    """基本编译测试"""
    print("="*60)
    print("GNN Compiler - Basic Test")
    print("="*60)

    # 创建配置
    config = CompilerConfig(
        k_set=[8, 10, 12],  # 测试3个k值
        max_pipeline_blocks=2,  # 最多2段pipeline
        top_k_peps=3,  # 每个sg保留3个候选
        max_iterations=10,  # 最多10次迭代
        verbose=True
    )

    # 创建编译器
    compiler = GNNCompiler(config)

    # 执行编译
    # 测试用例：中等规模图
    total_nodes = 30000
    total_edges = 60000

    print(f"\nCompiling graph: {total_nodes} nodes, {total_edges} edges")
    print(f"Testing k in {config.k_set}")

    try:
        result = compiler.compile(total_nodes, total_edges)

        # 验证结果
        print("\n" + "="*60)
        print("Test Results")
        print("="*60)

        assert 'partition_config' in result
        assert 'execution_plan' in result
        assert 'statistics' in result

        stats = result['statistics']
        print(f"✓ Partition: k={result['partition_config']['k']}")
        print(f"✓ Clusters: {stats['num_clusters']}")
        print(f"✓ Unique Models: {stats['num_unique_models']}")
        print(f"✓ Makespan: {stats['makespan']:.2f}ms")

        # 检查cluster结构
        clusters = result['execution_plan']['clusters']
        print(f"\nCluster Details:")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i}: {cluster['num_subgraphs']} subgraphs")
            print(f"    PEP: {cluster['pep_key']}")

        print("\n✓ All tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_partition_only():
    """测试图划分功能"""
    from core import GraphPartitioner

    print("\n" + "="*60)
    print("Testing Graph Partitioner")
    print("="*60)

    partitioner = GraphPartitioner(padding_multiple=1000)

    # 测试不同的k值
    total_nodes = 20000
    total_edges = 40000

    for k in [8, 10, 12]:
        print(f"\nPartition with k={k}:")
        subgraphs = partitioner.partition(total_nodes, total_edges, k)

        for sg in subgraphs:
            print(f"  {sg}")

    print("✓ Partition test passed!")


def test_pep_generation():
    """测试PEP生成功能"""
    from core import GraphPartitioner, PEPGenerator
    from utils import CompilerConfig

    print("\n" + "="*60)
    print("Testing PEP Generator")
    print("="*60)

    config = CompilerConfig()
    partitioner = GraphPartitioner()
    pep_generator = PEPGenerator(config)

    # 创建一个测试subgraph
    subgraphs = partitioner.partition(10000, 20000, k=5)
    sg = subgraphs[0]

    print(f"\nGenerating PEPs for {sg}")

    candidates = pep_generator.generate_candidates(sg)

    print(f"Generated {len(candidates)} valid PEPs:")
    for i, pep in enumerate(candidates[:5]):  # 只显示前5个
        print(f"  PEP {i+1}: {pep}")

    print(f"✓ PEP generation test passed! ({len(candidates)} candidates)")


if __name__ == '__main__':
    # 运行所有测试
    print("Starting GNN Compiler Tests...\n")

    # Test 1: Graph Partition
    test_partition_only()

    # Test 2: PEP Generation
    test_pep_generation()

    # Test 3: Full Compilation
    success = test_basic_compilation()

    if success:
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED ✗")
        print("="*60)
