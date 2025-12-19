"""
Pipeline Executor - Main Orchestrator
主执行器：整合所有组件，执行完整pipeline
"""

import json
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional

from .data_loader import GraphDataLoader
from .ghost_node_handler import GhostNodeHandler
from .model_manager import ModelManager
from .subgraph_executor import SubgraphExecutor


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

    def __init__(self, compilation_result_path: str, dataset_name: str = 'flickr'):
        """
        Args:
            compilation_result_path: Compiler输出的JSON文件路径
            dataset_name: 数据集名称
        """
        self.compilation_result_path = compilation_result_path
        self.dataset_name = dataset_name

        # 加载编译结果
        with open(compilation_result_path, 'r') as f:
            self.compilation_result = json.load(f)

        self.partition_config = self.compilation_result['partition_config']
        self.execution_plan = self.compilation_result['execution_plan']
        self.statistics = self.compilation_result['statistics']

        self.num_subgraphs = self.partition_config['num_subgraphs']
        self.num_clusters = self.execution_plan['num_clusters']
        self.estimated_makespan = self.statistics['makespan']

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
        self.model_manager = ModelManager(self.execution_plan)
        self.model_manager.ensure_models_exist()
        self.model_manager.load_models()

        # Step 4: 创建subgraph执行器
        print(f"\n[Step 4/4] Creating subgraph executors...")
        self._create_subgraph_executors()

        print(f"\n✓ Preparation complete!\n")

    def _create_subgraph_executors(self):
        """
        为每个subgraph创建执行器
        """
        # 目前所有subgraph用同一个cluster（同一个PEP）
        cluster = self.execution_plan['clusters'][0]
        pep = cluster['pep']
        models = self.model_manager.get_cluster_models(cluster_id=0)

        for sg_config in self.partition_config['subgraphs']:
            sg_id = sg_config['id']

            self.subgraph_executors[sg_id] = SubgraphExecutor(
                subgraph_id=sg_id,
                subgraph_config=sg_config,
                pep=pep,
                models=models
            )

        print(f"  ✓ Created {len(self.subgraph_executors)} subgraph executors")

    def execute(self) -> Dict:
        """
        执行推理

        Returns:
            {
                'embeddings': [total_nodes, hidden_dim] 所有节点的输出
                'per_subgraph_times': [sg0_time, sg1_time, ...] 每个subgraph的时间
                'total_time': 总时间（ms）
            }
        """
        print(f"Executing {self.num_subgraphs} subgraphs sequentially...")

        # 准备输出
        total_nodes = self.data_loader.full_data.num_nodes
        hidden_dim = 256  # GraphSAGE默认hidden dimension
        all_embeddings = torch.zeros(total_nodes, hidden_dim)

        per_subgraph_times = []
        start_time = time.time()

        # 顺序执行每个subgraph
        for sg_id in range(self.num_subgraphs):
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

        total_time = (time.time() - start_time) * 1000  # ms

        print(f"\n✓ All subgraphs executed")
        print(f"  Total time: {total_time:.2f}ms")

        return {
            'embeddings': all_embeddings,
            'per_subgraph_times': per_subgraph_times,
            'total_time': total_time
        }
