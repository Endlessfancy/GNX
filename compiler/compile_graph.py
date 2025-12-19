"""
GNN Compiler CLI Tool
"""

import argparse
from pathlib import Path
from compiler import GNNCompiler
from utils import CompilerConfig


def main():
    parser = argparse.ArgumentParser(description='GNN Graph Compiler')

    # Graph configuration
    parser.add_argument('--nodes', type=int, default=50000,
                       help='Total number of nodes in the graph')
    parser.add_argument('--edges', type=int, default=100000,
                       help='Total number of edges in the graph')

    # Compiler configuration
    parser.add_argument('--k-min', type=int, default=8,
                       help='Minimum partition count')
    parser.add_argument('--k-max', type=int, default=15,
                       help='Maximum partition count')
    parser.add_argument('--max-blocks', type=int, default=2,
                       help='Maximum pipeline blocks (1-3)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top PEPs to keep per subgraph')

    # Optimization configuration
    parser.add_argument('--max-iter', type=int, default=20,
                       help='Maximum optimization iterations')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Convergence threshold (ms)')

    # Output configuration
    parser.add_argument('--output-dir', type=Path,
                       default=Path('/home/haoyang/private/GNX_final/compiler/output'),
                       help='Output directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # 创建配置
    config = CompilerConfig(
        k_set=list(range(args.k_min, args.k_max + 1)),
        max_pipeline_blocks=args.max_blocks,
        top_k_peps=args.top_k,
        max_iterations=args.max_iter,
        convergence_threshold=args.threshold,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    # 创建编译器
    compiler = GNNCompiler(config)

    # 执行编译
    result = compiler.compile(args.nodes, args.edges)

    # 打印结果摘要
    print("\n" + "="*60)
    print("Compilation Summary")
    print("="*60)
    print(f"Graph: {args.nodes} nodes, {args.edges} edges")
    print(f"Partition: k={result['partition_config']['k']}")
    print(f"Clusters: {result['statistics']['num_clusters']}")
    print(f"Unique Models: {result['statistics']['num_unique_models']}")
    print(f"Estimated Makespan: {result['statistics']['makespan']:.2f}ms")
    print("="*60)


if __name__ == '__main__':
    main()
