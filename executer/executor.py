"""
Pipeline Executor - Main Orchestrator
主执行器：整合所有组件，执行完整pipeline
"""

import json
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional

# Support both relative and absolute imports
try:
    from .data_loader import GraphDataLoader
    from .ghost_node_handler import GhostNodeHandler
    from .model_manager import ModelManager
    from .subgraph_executor import SubgraphExecutor
except ImportError:
    # Fallback to absolute imports when run as script
    from data_loader import GraphDataLoader
    from ghost_node_handler import GhostNodeHandler
    from model_manager import ModelManager
    from subgraph_executor import SubgraphExecutor


class PipelineExecutor:
    """
    Pipeline执行器

    整合所有组件，执行完整的推理pipeline:
    1. 加载图数据并分区
    2. 收集ghost node特征
    3. 导出/加载模型
    4. 顺序执行所有subgraph
    5. 合并结果
    """

    def __init__(self, compilation_result_path: Optional[str] = None,
                 dataset_name: str = 'flickr',
                 custom_execution_plan: Optional[Dict] = None,
                 custom_partition_config: Optional[Dict] = None):
        """
        Args:
            compilation_result_path: Compiler输出的JSON文件路径 (可选，如果提供custom_execution_plan则不需要)
            dataset_name: 数据集名称
            custom_execution_plan: 自定义执行计划（用于测试）
            custom_partition_config: 自定义分区配置（用于测试）
        """
        self.compilation_result_path = compilation_result_path
        self.dataset_name = dataset_name

        # 加载编译结果或使用自定义配置
        if custom_execution_plan is not None:
            # 使用自定义配置（测试模式）
            self.execution_plan = custom_execution_plan

            # partition_config 仍需从 compilation_result.json 加载（包含图分区信息）
            if custom_partition_config is not None:
                self.partition_config = custom_partition_config
            else:
                # 从默认路径加载 partition_config
                default_result_path = Path(__file__).parent.parent / 'compiler' / 'output' / 'compilation_result.json'
                if default_result_path.exists():
                    with open(default_result_path, 'r') as f:
                        compilation_result = json.load(f)
                    self.partition_config = compilation_result['partition_config']
                else:
                    raise FileNotFoundError(
                        f"partition_config not provided and default compilation_result.json not found at {default_result_path}. "
                        "Please provide custom_partition_config or ensure compilation_result.json exists."
                    )

            self.statistics = {'makespan': 0}  # Placeholder
        else:
            # 正常模式：加载JSON
            with open(compilation_result_path, 'r') as f:
                self.compilation_result = json.load(f)

            self.partition_config = self.compilation_result['partition_config']
            self.execution_plan = self.compilation_result['execution_plan']
            self.statistics = self.compilation_result['statistics']

        self.num_subgraphs = self.partition_config['num_subgraphs']
        self.num_clusters = self.execution_plan['num_clusters']
        self.estimated_makespan = self.statistics.get('makespan', 0)

        # 初始化组件（延迟到prepare()）
        self.data_loader = None
        self.ghost_handler = None
        self.model_manager = None
        self.subgraph_executors = {}

    def prepare(self):
        """
        准备阶段：加载数据、导出模型、初始化执行器

        这一步比较耗时（模型导出），但只需执行一次
        """
        print(f"\nPreparing executor...")

        # Step 1: 加载图数据
        print(f"\n[Step 1/4] Loading graph data...")
        self.data_loader = GraphDataLoader(
            self.dataset_name,
            self.partition_config
        )

        # Step 2: 收集ghost node特征
        print(f"\n[Step 2/4] Collecting ghost node features...")
        self.ghost_handler = GhostNodeHandler(self.data_loader)

        # Step 3: 导出和加载模型
        print(f"\n[Step 3/4] Exporting and loading models...")
        # Pass compilation_result directory for resolving relative model paths
        if self.compilation_result_path is not None:
            compilation_result_dir = Path(self.compilation_result_path).parent
        else:
            # 使用默认路径（用于自定义 execution_plan）
            compilation_result_dir = Path(__file__).parent.parent / 'compiler' / 'output'
        self.model_manager = ModelManager(self.execution_plan, compilation_result_dir)
        self.model_manager.ensure_models_exist()
        self.model_manager.load_models()

        # Step 4: 创建subgraph执行器
        print(f"\n[Step 4/4] Creating subgraph executors...")
        self._create_subgraph_executors()

        print(f"\n✓ Preparation complete!\n")

    def _create_subgraph_executors(self):
        """
        为每个subgraph创建执行器（支持多cluster）
        """
        # 为每个cluster的subgraphs创建executor
        for cluster_id, cluster in enumerate(self.execution_plan['clusters']):
            pep = cluster['pep']
            models = self.model_manager.get_cluster_models(cluster_id=cluster_id)

            for sg_id in cluster['subgraph_ids']:
                # 找到对应的subgraph配置
                sg_config = self.partition_config['subgraphs'][sg_id]

                self.subgraph_executors[sg_id] = SubgraphExecutor(
                    subgraph_id=sg_id,
                    subgraph_config=sg_config,
                    pep=pep,
                    models=models
                )

        print(f"  ✓ Created {len(self.subgraph_executors)} subgraph executors")

    def execute(self, use_pipeline_parallelism: bool = False) -> Dict:
        """
        执行推理（支持多cluster）

        Args:
            use_pipeline_parallelism: 是否启用流水线并行
                - False: 顺序执行（默认，向后兼容）
                - True: 使用流水线并行执行（block级别并行 + 数据并行）

        Returns:
            {
                'embeddings': [total_nodes, hidden_dim] 所有节点的输出
                'per_subgraph_times': [sg0_time, sg1_time, ...] 每个subgraph的时间
                'per_cluster_times': [cluster0_time, cluster1_time, ...] 每个cluster的时间
                'total_time': 总时间（ms）
            }
        """
        print(f"\n{'='*70}")
        print(f"Execution Mode: {'Pipeline Parallel' if use_pipeline_parallelism else 'Sequential'}")
        print(f"{'='*70}\n")

        # 准备输出
        total_nodes = self.data_loader.full_data.num_nodes
        hidden_dim = 256  # GraphSAGE默认hidden dimension
        all_embeddings = torch.zeros(total_nodes, hidden_dim)

        per_subgraph_times = []
        per_cluster_times = []
        start_time = time.time()

        if use_pipeline_parallelism:
            # 使用流水线并行执行
            try:
                from .pipeline_executor import PipelineExecutor as PipelineExec
            except ImportError:
                from pipeline_executor import PipelineExecutor as PipelineExec

            for cluster_id, cluster in enumerate(self.execution_plan['clusters']):
                print(f"\n{'='*70}")
                print(f"Cluster {cluster_id}: {cluster['pep_key']} (Pipeline Mode)")
                print(f"  PEP: {cluster['pep']}")
                print(f"  Subgraphs: {cluster['subgraph_ids']}")
                print(f"  Blocks: {len(cluster['pep'])}")
                print(f"{'='*70}\n")

                cluster_start = time.time()

                # 创建流水线执行器
                pipeline_exec = PipelineExec(
                    cluster_id, cluster, self.data_loader, self.subgraph_executors
                )

                # 保存最后一个pipeline_exec实例（用于详细分析）
                self.last_pipeline_exec = pipeline_exec

                # 执行流水线
                result = pipeline_exec.execute_pipeline()

                # 合并embeddings
                all_embeddings += result['embeddings']

                cluster_time = (time.time() - cluster_start) * 1000
                per_cluster_times.append(cluster_time)
                print(f"\n✓ Cluster {cluster_id} completed in {cluster_time:.2f}ms\n")

        else:
            # 原有的顺序执行逻辑
            for cluster_id, cluster in enumerate(self.execution_plan['clusters']):
                print(f"\n{'='*70}")
                print(f"Cluster {cluster_id}: {cluster['pep_key']}")
                print(f"  PEP: {cluster['pep']}")
                print(f"  Subgraphs: {cluster['subgraph_ids']}")
                print(f"{'='*70}\n")

                cluster_start = time.time()

                # 执行该cluster的所有subgraph
                for sg_id in cluster['subgraph_ids']:
                    print(f"  Subgraph {sg_id}...", end=" ", flush=True)

                    # 获取subgraph数据
                    sg_data = self.data_loader.get_subgraph_data(sg_id)
                    edge_index = sg_data['edge_index']
                    x = sg_data['x']
                    owned_nodes = sg_data['owned_nodes']
                    global_owned_nodes = sg_data['global_owned_nodes']

                    # 执行推理
                    executor = self.subgraph_executors[sg_id]
                    embeddings, sg_time = executor.execute(edge_index, x, owned_nodes)

                    # 存储结果（映射回全局节点ID）
                    all_embeddings[global_owned_nodes] = embeddings

                    per_subgraph_times.append(sg_time)
                    print(f"{sg_time:.2f}ms")

                cluster_time = (time.time() - cluster_start) * 1000
                per_cluster_times.append(cluster_time)
                print(f"\n✓ Cluster {cluster_id} completed in {cluster_time:.2f}ms\n")

        total_time = (time.time() - start_time) * 1000  # ms

        print(f"\n✓ All clusters executed")
        print(f"  Total time: {total_time:.2f}ms")

        return {
            'embeddings': all_embeddings,
            'per_subgraph_times': per_subgraph_times,
            'per_cluster_times': per_cluster_times,
            'total_time': total_time
        }
