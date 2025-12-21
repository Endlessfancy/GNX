"""
Pipeline Executor - Block-level Pipeline Parallelism
Block级别流水线并行执行器
"""

import time
import torch
import threading
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor


class PipelineExecutor:
    """
    流水线执行器：实现block级别的流水线并行

    工作原理:
    - 每个block运行在独立的线程中
    - Block间通过字典传递中间结果
    - 使用Event实现同步：Block N等待Block N-1完成对应subgraph

    示例:
        PEP: [[['CPU', 'GPU'], [1,2,3,4,5], [0.5, 0.5]], [['NPU'], [6,7]]]
        Subgraphs: [0, 1, 2, 3, 4, 5, 6, 7]

        执行流程:
        Block0-Thread: SG0 -> SG1 -> SG2 -> ... (CPU+GPU并发处理每个SG)
        Block1-Thread:   └─> SG0 -> SG1 -> ... (NPU等Block0完成SG0后开始)
    """

    def __init__(self, cluster_id: int, cluster_config: Dict,
                 data_loader, subgraph_executors: Dict):
        """
        Args:
            cluster_id: Cluster ID
            cluster_config: Cluster配置 {'pep': ..., 'subgraph_ids': ...}
            data_loader: DataLoader实例
            subgraph_executors: {sg_id: SubgraphExecutor} 字典
        """
        self.cluster_id = cluster_id
        self.pep = cluster_config['pep']
        self.subgraph_ids = cluster_config['subgraph_ids']
        self.num_blocks = len(self.pep)
        self.data_loader = data_loader
        self.subgraph_executors = subgraph_executors

        # 中间结果存储: {(sg_id, block_id): output_data}
        self.intermediate_results = {}
        self.results_lock = threading.Lock()

        # 同步事件: {(sg_id, block_id): Event}
        # 用于通知下一个block："我已经完成了这个subgraph"
        self.completion_events = {}
        for sg_id in self.subgraph_ids:
            for block_id in range(self.num_blocks):
                self.completion_events[(sg_id, block_id)] = threading.Event()

        # 存储原始subgraph数据（避免重复加载）
        self.raw_data_cache = {}
        self.cache_lock = threading.Lock()

    def execute_pipeline(self) -> Dict:
        """
        执行流水线并行

        Returns:
            {
                'embeddings': torch.Tensor [total_nodes, embed_dim],
                'per_subgraph_times': List[float],  # 每个subgraph的总时间
                'total_time': float  # 总执行时间（ms）
            }
        """
        print(f"  Starting pipeline execution with {self.num_blocks} blocks...")
        start_time = time.time()

        # 创建线程池，每个block一个worker
        with ThreadPoolExecutor(max_workers=self.num_blocks) as executor:
            futures = []
            for block_id in range(self.num_blocks):
                future = executor.submit(self._block_worker, block_id)
                futures.append(future)

            # 等待所有block完成
            for future in futures:
                future.result()

        # 收集最终结果（从最后一个block的输出中）
        final_embeddings = self._collect_final_embeddings()

        total_time = (time.time() - start_time) * 1000  # ms

        return {
            'embeddings': final_embeddings,
            'total_time': total_time
        }

    def _block_worker(self, block_id: int):
        """
        Block worker线程：负责执行所有subgraph的特定block

        工作流程：
        1. 遍历所有subgraph
        2. 对于每个subgraph:
           - 如果是Block 0: 直接获取原始数据
           - 如果是Block N: 等待Block N-1完成，然后获取中间结果
           - 执行当前block
           - 保存结果并发送完成信号

        Args:
            block_id: 当前block的ID (0 to num_blocks-1)
        """
        for sg_id in self.subgraph_ids:
            sg_start_time = time.time()

            # 获取输入数据
            if block_id == 0:
                # Block 0: 使用原始subgraph数据
                input_data = self._get_raw_subgraph_data(sg_id)
            else:
                # Block N: 等待Block N-1完成此subgraph
                prev_block_id = block_id - 1
                self.completion_events[(sg_id, prev_block_id)].wait()

                # 获取前一个block的输出
                input_data = self._get_intermediate_result(sg_id, prev_block_id)

            # 执行当前block（会调用SubgraphExecutor执行单个block）
            output_data = self._execute_single_block(sg_id, block_id, input_data)

            # 保存结果
            self._save_intermediate_result(sg_id, block_id, output_data)

            # 发送完成信号，通知下一个block
            self.completion_events[(sg_id, block_id)].set()

            sg_time = (time.time() - sg_start_time) * 1000

            # 只有最后一个block打印时间（避免多线程打印混乱）
            if block_id == self.num_blocks - 1:
                print(f"  Subgraph {sg_id}... {sg_time:.2f}ms")

    def _get_raw_subgraph_data(self, sg_id: int) -> Dict:
        """
        获取subgraph的原始数据（特征、边）
        线程安全，带缓存

        Args:
            sg_id: Subgraph ID

        Returns:
            {
                'x': torch.Tensor,
                'edge_index': torch.Tensor,
                'owned_nodes': torch.Tensor,
                'global_owned_nodes': torch.Tensor,
                'num_nodes': int
            }
        """
        # 检查缓存
        with self.cache_lock:
            if sg_id in self.raw_data_cache:
                return self.raw_data_cache[sg_id]

        # 加载数据
        sg_data = self.data_loader.get_subgraph_data(sg_id)

        data_dict = {
            'x': sg_data['x'],
            'edge_index': sg_data['edge_index'],
            'owned_nodes': sg_data['owned_nodes'],
            'global_owned_nodes': sg_data['global_owned_nodes'],
            'num_nodes': sg_data['x'].size(0)
        }

        # 缓存数据
        with self.cache_lock:
            self.raw_data_cache[sg_id] = data_dict

        return data_dict

    def _execute_single_block(self, sg_id: int, block_id: int, input_data: Dict) -> Dict:
        """
        执行单个block（可能包含数据并行）

        调用SubgraphExecutor的_execute_block方法

        Args:
            sg_id: Subgraph ID
            block_id: Block ID
            input_data: 输入数据字典

        Returns:
            output_data: 输出数据字典
        """
        executor = self.subgraph_executors[sg_id]
        block = executor.pep[block_id]

        # 确保input_data包含owned_nodes
        if 'owned_nodes' not in input_data:
            # 从raw data获取
            raw_data = self._get_raw_subgraph_data(sg_id)
            input_data['owned_nodes'] = raw_data['owned_nodes']

        # 调用SubgraphExecutor的_execute_block（内部会处理数据并行）
        output_data = executor._execute_block(
            block_id,
            block,
            input_data,
            input_data['owned_nodes']
        )

        return output_data

    def _save_intermediate_result(self, sg_id: int, block_id: int, output_data: Dict):
        """
        线程安全地保存中间结果

        Args:
            sg_id: Subgraph ID
            block_id: Block ID
            output_data: 输出数据字典
        """
        with self.results_lock:
            self.intermediate_results[(sg_id, block_id)] = output_data

    def _get_intermediate_result(self, sg_id: int, block_id: int) -> Dict:
        """
        线程安全地获取中间结果

        Args:
            sg_id: Subgraph ID
            block_id: Block ID

        Returns:
            output_data: 之前保存的输出数据字典
        """
        with self.results_lock:
            return self.intermediate_results[(sg_id, block_id)]

    def _collect_final_embeddings(self) -> torch.Tensor:
        """
        收集所有subgraph的最终embeddings并合并

        Returns:
            final_embeddings: [total_nodes, embed_dim] 全局embeddings tensor
        """
        # 从最后一个block收集结果
        final_block_id = self.num_blocks - 1

        # 准备存储所有subgraph的embeddings
        all_embeddings_list = []

        for sg_id in self.subgraph_ids:
            # 获取最后一个block的输出
            result = self.intermediate_results[(sg_id, final_block_id)]

            # 获取global节点ID映射
            raw_data = self._get_raw_subgraph_data(sg_id)
            global_owned_nodes = raw_data['global_owned_nodes']
            owned_nodes = raw_data['owned_nodes']

            # 提取embeddings
            embeddings = result.get('output', result.get('x'))

            # 确保提取owned nodes的embeddings
            if embeddings.size(0) > len(owned_nodes):
                # Full-sized output，需要提取
                embeddings = embeddings[owned_nodes]

            all_embeddings_list.append((global_owned_nodes, embeddings))

        # 合并为全局tensor
        total_nodes = self.data_loader.full_data.num_nodes
        embed_dim = all_embeddings_list[0][1].size(1)
        final_tensor = torch.zeros(total_nodes, embed_dim)

        for global_ids, embeds in all_embeddings_list:
            final_tensor[global_ids] = embeds

        return final_tensor
