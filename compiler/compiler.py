"""
GNN Compiler - Main Class
整合所有编译阶段
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils import CompilerConfig, ProfilingLoader, GraphLoader
from core import (
    GraphPartitioner,
    PEPGenerator,
    CostEstimator,
    GlobalOptimizer,
    ModelCodegen
)


class GNNCompiler:
    """
    GNN编译器主类

    完整的编译流程：
    1. Graph Partition: 划分大图为多个subgraph
    2. PEP Generation: 为每个subgraph生成候选PEP
    3. Global Optimization: Pipeline-aware全局优化
    4. Model Codegen: 导出和缓存模型
    5. Execution Plan: 生成最终执行计划
    """

    def __init__(self, config: Optional[CompilerConfig] = None, dataset_name: Optional[str] = None):
        """
        Args:
            config: 编译器配置（可选，默认使用默认配置）
            dataset_name: 数据集名称（可选，覆盖config中的dataset_name）
        """
        self.config = config or CompilerConfig()

        # 覆盖数据集名称（如果提供）
        if dataset_name is not None:
            self.config.dataset_name = dataset_name

        # 加载图数据
        self.graph_loader = GraphLoader()
        self.graph_data = self.graph_loader.load_dataset(self.config.dataset_name)

        print(f"\n{'='*60}")
        print(f"Dataset: {self.config.dataset_name.upper()}")
        print(f"  Nodes: {self.graph_data.num_nodes:,}")
        print(f"  Edges: {self.graph_data.num_edges:,}")
        print(f"{'='*60}")

        # 加载profiling数据
        self.profiling = ProfilingLoader(self.config.profiling_dir)

        # 初始化各个模块
        self.partitioner = GraphPartitioner(
            self.config.npu_padding_multiple,
            use_metis=self.config.use_metis
        )
        self.pep_generator = PEPGenerator(self.config)
        self.cost_estimator = CostEstimator(self.profiling, self.config)
        self.global_optimizer = GlobalOptimizer(self.cost_estimator, self.config)
        self.model_codegen = ModelCodegen(self.config)

    def compile(self) -> Dict:
        """
        编译主函数

        使用已加载的图数据进行编译

        Returns:
            执行计划字典
        """
        print("\n" + "="*60)
        print("GNN Compiler - Pipeline-Aware Optimization")
        print("="*60)

        # Phase 1: Graph Partition
        print("\n[Phase 1] Graph Partition...")
        try:
            partition_results = self.partitioner.partition_multiple_k(
                self.graph_data.edge_index,  # 传递真实图结构
                self.config.k_set
            )
        except Exception as e:
            print(f"✗ Graph partitioning failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        if not partition_results:
            print("✗ No valid partitions generated!")
            return None

        # 对每个k进行编译，选择最优的
        best_k = None
        best_makespan = float('inf')
        best_result = None

        for k, subgraphs in partition_results.items():
            print(f"\n{'='*60}")
            print(f"Compiling for k={k} ({len(subgraphs)} subgraphs)")
            print(f"{'='*60}")

            try:
                # Phase 2: PEP Generation
                result = self._compile_for_k(k, subgraphs)

                # 选择最优k
                if result and result['statistics']['makespan'] < best_makespan:
                    best_makespan = result['statistics']['makespan']
                    best_k = k
                    best_result = result
            except Exception as e:
                print(f"✗ Compilation failed for k={k}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if best_result is None:
            print("\n✗ All compilation attempts failed!")
            return None

        # 输出最优结果
        print(f"\n{'='*60}")
        print(f"Best Configuration: k={best_k}")
        print(f"Estimated Makespan: {best_makespan:.2f}ms")
        print(f"{'='*60}")

        # 保存编译结果
        self._save_compilation_result(best_result)

        return best_result

    def _compile_for_k(self, k: int, subgraphs: List) -> Dict:
        """
        对给定的k值进行完整编译

        Args:
            k: 划分数量
            subgraphs: Subgraph列表

        Returns:
            编译结果字典
        """
        # Phase 2: PEP Generation
        print(f"\n[Phase 2] Generating candidate PEPs...")
        top_k_peps = {}

        for sg in subgraphs:
            # 生成所有候选PEP
            candidates = self.pep_generator.generate_candidates(sg)

            if not candidates:
                print(f"  Warning: No valid PEP for subgraph {sg.id}")
                continue

            # 估算代价并排序
            pep_costs = []
            for pep in candidates:
                cost_info = self.cost_estimator.estimate_pep_cost(pep, sg)
                pep_costs.append((pep, cost_info['total_time']))

            # 按代价排序
            pep_costs.sort(key=lambda x: x[1])

            # 保留Top-K
            top_k_peps[sg.id] = pep_costs[:self.config.top_k_peps]

            if self.config.verbose:
                print(f"  Subgraph {sg.id}: {len(candidates)} candidates → Top-{len(top_k_peps[sg.id])}")
                print(f"    Best PEP: {top_k_peps[sg.id][0][0]}, cost={top_k_peps[sg.id][0][1]:.2f}ms")

        # Phase 3: Global Optimization
        print(f"\n[Phase 3] Pipeline-aware global optimization...")
        assignment, clusters, makespan = self.global_optimizer.optimize(subgraphs, top_k_peps)

        # Phase 4: Model Codegen
        print(f"\n[Phase 4] Model code generation...")
        model_index = self.model_codegen.generate_models(assignment, clusters, subgraphs)

        # Phase 5: Generate Execution Plan
        print(f"\n[Phase 5] Generating execution plan...")
        execution_plan = self._generate_execution_plan(
            k,
            subgraphs,
            assignment,
            clusters,
            makespan,
            model_index
        )

        return execution_plan

    def _generate_execution_plan(self, k: int, subgraphs: List, assignment: Dict,
                                 clusters: Dict, makespan: float, model_index: Dict) -> Dict:
        """
        生成最终执行计划

        Args:
            k: 划分数量
            subgraphs: Subgraph列表
            assignment: PEP分配
            clusters: Cluster分组
            makespan: 预估makespan
            model_index: 模型索引

        Returns:
            执行计划字典
        """
        # 构建cluster执行计划
        cluster_plans = []

        for cluster_key, sg_list in clusters.items():
            # 获取该cluster的PEP
            sample_sg = sg_list[0]
            pep, _ = assignment[sample_sg.id]

            # 构建model references（使用相对路径以支持跨平台）
            model_refs = {}
            for block_idx, block in enumerate(pep.blocks):
                for device in block.devices:
                    if device in ['CPU', 'GPU']:
                        model_key = self.model_codegen._generate_dynamic_model_key(device, block.stages)
                    else:  # NPU
                        model_key = self.model_codegen._generate_static_model_key(
                            device, block.stages, sample_sg.n_pad, sample_sg.m_pad
                        )

                    ref_key = f"block_{block_idx}_{device}"
                    # Convert to relative path for cross-platform compatibility
                    abs_path = model_index.get(model_key, "")
                    if abs_path:
                        # Store relative path: models/MODEL_NAME.onnx
                        rel_path = f"models/{Path(abs_path).name}"
                        model_refs[ref_key] = rel_path
                    else:
                        model_refs[ref_key] = ""

            cluster_plans.append({
                'pep_key': cluster_key,
                'pep': pep.to_executor_format(),
                'subgraph_ids': [sg.id for sg in sg_list],
                'model_refs': model_refs,
                'num_subgraphs': len(sg_list)
            })

        # 构建完整执行计划
        execution_plan = {
            'partition_config': {
                'k': k,
                'num_subgraphs': len(subgraphs),
                'subgraphs': [
                    {
                        'id': sg.id,
                        'n': sg.n,
                        'm': sg.m,
                        'n_pad': sg.n_pad,
                        'm_pad': sg.m_pad,
                        'cut_edges': sg.cut_edges
                    }
                    for sg in subgraphs
                ]
            },
            'execution_plan': {
                'clusters': cluster_plans,
                'num_clusters': len(cluster_plans)
            },
            'statistics': {
                'makespan': makespan,
                'num_unique_models': len(model_index),
                'num_subgraphs': len(subgraphs),
                'num_clusters': len(cluster_plans)
            }
        }

        return execution_plan

    def _save_compilation_result(self, result: Dict):
        """
        保存编译结果到文件

        Args:
            result: 编译结果字典
        """
        output_path = self.config.output_dir / 'compilation_result.json'

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n✓ Compilation result saved to: {output_path}")
