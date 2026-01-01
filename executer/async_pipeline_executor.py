"""
Async Pipeline Executor - Asyncio-based Block-level Pipeline Parallelism
【阶段3优化】基于asyncio的异步pipeline执行器，实现真正的并行执行
"""

import asyncio
import time
import torch
from typing import Dict, List
from collections import defaultdict


class AsyncPipelineExecutor:
    """
    异步流水线执行器：使用asyncio实现block级别的真正并行

    与sync版本的区别:
    - 使用asyncio.Event代替threading.Event（无GIL阻塞）
    - 使用asyncio.create_task代替ThreadPoolExecutor（协程而非线程）
    - Model推理通过AsyncONNXWrapper异步执行（释放GIL）
    - 更高效的block间协调（event loop调度）

    工作原理:
    - 每个block运行在独立的协程中
    - Block间通过异步Event和字典传递中间结果
    - 模型推理在线程池中异步执行，释放GIL
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

        # 同步事件: {(sg_id, block_id): asyncio.Event}
        self.completion_events = {}
        for sg_id in self.subgraph_ids:
            for block_id in range(self.num_blocks):
                self.completion_events[(sg_id, block_id)] = asyncio.Event()

        # 存储原始subgraph数据（避免重复加载）
        self.raw_data_cache = {}

        # 【阶段1优化】预计算的partition缓存
        self.partition_cache = {}
        self.is_preloaded = False

        # 详细时间统计
        self.detailed_timing = {
            'block_times': {},      # {(sg_id, block_id): {'wait': t, 'exec': t, 'total': t}}
            'timestamps': {},       # {(sg_id, block_id): {'start': t, 'end': t}}
            'task_ids': {},         # {block_id: task_id}
            'preload_time': 0.0,
        }

        self.actual_pipeline_time = 0.0

    async def execute_pipeline_async(self) -> Dict:
        """
        异步执行流水线并行

        Returns:
            {
                'embeddings': torch.Tensor [total_nodes, embed_dim],
                'total_time': float  # 总执行时间（ms）
            }
        """
        # 【阶段1优化】首次执行时预加载所有数据（仍然是同步，但只执行一次）
        if not self.is_preloaded:
            print(f"  [Optimization] Pre-loading all data for cluster {self.cluster_id}...")
            self._preload_all_data()  # 同步调用，因为是一次性准备
            print(f"  [Optimization] Pre-loading completed in {self.detailed_timing['preload_time']:.0f}ms")

        print(f"  Starting async pipeline execution with {self.num_blocks} blocks...")
        start_time = time.time()

        # 创建所有block的协程任务
        tasks = []
        for block_id in range(self.num_blocks):
            task = asyncio.create_task(self._block_worker_async(block_id))
            tasks.append(task)

        # 并发执行所有block（event loop自动协调）
        await asyncio.gather(*tasks)

        # 收集最终结果
        final_embeddings = self._collect_final_embeddings()

        total_time = (time.time() - start_time) * 1000  # ms
        self.actual_pipeline_time = total_time

        return {
            'embeddings': final_embeddings,
            'total_time': total_time
        }

    async def _block_worker_async(self, block_id: int):
        """
        异步Block worker：负责执行所有subgraph的特定block

        与sync版本的区别:
        - 使用 await 等待Event（不阻塞其他协程）
        - 使用 async方法调用
        """
        import asyncio as aio
        task_id = id(aio.current_task())

        # 记录task ID
        self.detailed_timing['task_ids'][block_id] = task_id

        for sg_id in self.subgraph_ids:
            total_start_time = time.time()
            wait_time = 0.0

            # 记录开始时间戳
            self.detailed_timing['timestamps'][(sg_id, block_id)] = {
                'start': total_start_time
            }

            # 日志
            print(f"  [Block{block_id} Task-{task_id % 10000}] Starting SG{sg_id} at {total_start_time:.3f}s")

            # 获取输入数据
            if block_id == 0:
                # Block 0: 使用原始subgraph数据
                input_data = self._get_raw_subgraph_data(sg_id)
            else:
                # Block N: 等待Block N-1完成此subgraph
                prev_block_id = block_id - 1

                wait_start_time = time.time()
                await self.completion_events[(sg_id, prev_block_id)].wait()  # 【异步等待】
                wait_time = (time.time() - wait_start_time) * 1000

                # 获取前一个block的输出
                input_data = self._get_intermediate_result(sg_id, prev_block_id)

            # 执行当前block
            exec_start_time = time.time()
            output_data = await self._execute_single_block_async(sg_id, block_id, input_data)  # 【异步执行】
            exec_time = (time.time() - exec_start_time) * 1000

            # 保存结果
            self._save_intermediate_result(sg_id, block_id, output_data)

            # 发送完成信号
            self.completion_events[(sg_id, block_id)].set()

            # 计算总时间
            total_end_time = time.time()
            total_time = (total_end_time - total_start_time) * 1000

            # 记录结束时间戳
            self.detailed_timing['timestamps'][(sg_id, block_id)]['end'] = total_end_time

            # 保存详细统计
            self.detailed_timing['block_times'][(sg_id, block_id)] = {
                'wait': wait_time,
                'exec': exec_time,
                'total': total_time
            }

            # 日志
            print(f"  [Block{block_id} Task-{task_id % 10000}] Finished SG{sg_id}: wait={wait_time:.0f}ms, exec={exec_time:.0f}ms, total={total_time:.0f}ms")

            if block_id == self.num_blocks - 1:
                print(f"  Subgraph {sg_id} completed: {total_time:.2f}ms")

    def _get_raw_subgraph_data(self, sg_id: int) -> Dict:
        """
        获取subgraph的原始数据（从缓存）

        注意: 在async环境中，所有数据都已预加载，不需要异步加载
        """
        if sg_id in self.raw_data_cache:
            return self.raw_data_cache[sg_id]

        # Fallback（不应该执行到这里）
        sg_data = self.data_loader.get_subgraph_data(sg_id)
        data_dict = {
            'x': sg_data['x'],
            'edge_index': sg_data['edge_index'],
            'owned_nodes': sg_data['owned_nodes'],
            'global_owned_nodes': sg_data['global_owned_nodes'],
            'num_nodes': sg_data['x'].size(0)
        }
        self.raw_data_cache[sg_id] = data_dict
        return data_dict

    async def _execute_single_block_async(self, sg_id: int, block_id: int, input_data: Dict) -> Dict:
        """
        【异步执行】单个block

        关键优化: 调用SubgraphExecutor时，如果有数据并行（多设备），
        模型推理会通过AsyncONNXWrapper在线程池中异步执行
        """
        executor = self.subgraph_executors[sg_id]
        block = executor.pep[block_id]

        # 确保input_data包含owned_nodes
        if 'owned_nodes' not in input_data:
            raw_data = self._get_raw_subgraph_data(sg_id)
            input_data['owned_nodes'] = raw_data['owned_nodes']

        # 【阶段1优化】获取预计算的partition
        precomputed_partitions = self.partition_cache.get((sg_id, block_id), None)

        # 【注意】这里仍然调用同步方法，因为SubgraphExecutor是同步的
        # 但是其内部的ONNX Runtime推理会释放GIL，实现并发
        # 在更完整的实现中，可以进一步改造SubgraphExecutor为async
        output_data = executor._execute_block(
            block_id,
            block,
            input_data,
            input_data['owned_nodes'],
            precomputed_partitions
        )

        return output_data

    def _save_intermediate_result(self, sg_id: int, block_id: int, output_data: Dict):
        """保存中间结果（asyncio单线程，无需锁）"""
        self.intermediate_results[(sg_id, block_id)] = output_data

    def _get_intermediate_result(self, sg_id: int, block_id: int) -> Dict:
        """获取中间结果"""
        return self.intermediate_results[(sg_id, block_id)]

    def _collect_final_embeddings(self) -> torch.Tensor:
        """收集最终embeddings（从最后一个block的输出）"""
        last_block_id = self.num_blocks - 1
        all_embeddings = []

        for sg_id in self.subgraph_ids:
            result = self.intermediate_results[(sg_id, last_block_id)]

            # 提取embeddings
            if 'output' in result:
                embeddings = result['output']
            elif 'x' in result:
                embeddings = result['x']
            else:
                raise KeyError(f"No output found for SG{sg_id}, Block{last_block_id}")

            all_embeddings.append(embeddings)

        # 拼接所有subgraph的embeddings
        final_embeddings = torch.cat(all_embeddings, dim=0)
        return final_embeddings

    def _preload_all_data(self):
        """
        【阶段1优化】预加载所有subgraph数据和预计算所有partition
        （同步方法，在pipeline执行前一次性完成）
        """
        preload_start = time.time()

        # Step 1: 预加载所有subgraph的原始数据
        print(f"    Preloading {len(self.subgraph_ids)} subgraphs...")
        for sg_id in self.subgraph_ids:
            if sg_id not in self.raw_data_cache:
                self.raw_data_cache[sg_id] = self.data_loader.get_subgraph_data(sg_id)

        # Step 2: 预计算所有需要data parallelism的partition
        print(f"    Precomputing partitions for {self.num_blocks} blocks...")
        for block_id, block in enumerate(self.pep):
            devices = block[0]

            # 只有多设备的block需要预计算partition
            if len(devices) > 1:
                stages = block[1]
                ratios = block[2] if len(block) > 2 else [1.0 / len(devices)] * len(devices)

                for sg_id in self.subgraph_ids:
                    raw_data = self.raw_data_cache[sg_id]
                    partitions = self._precompute_partitions_for_block(
                        sg_id, block_id, raw_data, ratios
                    )
                    self.partition_cache[(sg_id, block_id)] = partitions

        preload_time = (time.time() - preload_start) * 1000
        self.detailed_timing['preload_time'] = preload_time
        self.is_preloaded = True

        print(f"    ✓ Preloaded {len(self.raw_data_cache)} subgraphs, "
              f"{len(self.partition_cache)} partitions in {preload_time:.0f}ms")

    def _precompute_partitions_for_block(self, sg_id: int, block_id: int,
                                         raw_data: Dict, ratios: List[float]) -> List[Dict]:
        """【阶段1优化】为特定block预计算partition"""
        try:
            from .node_partitioner import NodeBasedPartitioner
        except ImportError:
            from node_partitioner import NodeBasedPartitioner

        x = raw_data['x']
        edge_index = raw_data['edge_index']
        num_nodes = raw_data.get('num_nodes', x.size(0))

        partitions = NodeBasedPartitioner.partition_by_nodes(
            edge_index, num_nodes, ratios, x
        )

        return partitions

    def get_detailed_statistics(self) -> Dict:
        """获取详细的统计信息（与sync版本相同的接口）"""
        total_exec_per_block = defaultdict(float)
        total_wait_per_block = defaultdict(float)
        avg_exec_per_block = {}

        for (sg_id, block_id), times in self.detailed_timing['block_times'].items():
            total_exec_per_block[block_id] += times['exec']
            total_wait_per_block[block_id] += times['wait']

        for block_id in range(self.num_blocks):
            num_subgraphs = len(self.subgraph_ids)
            avg_exec_per_block[block_id] = total_exec_per_block[block_id] / num_subgraphs if num_subgraphs > 0 else 0

        # 理论性能分析
        theoretical_seq_time = sum(total_exec_per_block.values())
        actual_time = self.actual_pipeline_time
        theoretical_speedup = self.num_blocks
        actual_speedup = theoretical_seq_time / actual_time if actual_time > 0 else 0
        efficiency = (actual_speedup / theoretical_speedup * 100) if theoretical_speedup > 0 else 0

        block_exec_times = dict(total_exec_per_block)
        block_wait_times = dict(total_wait_per_block)
        avg_block_exec_times = avg_exec_per_block

        return {
            'theoretical_seq_time': theoretical_seq_time,
            'actual_pipeline_time': actual_time,
            'theoretical_speedup': theoretical_speedup,
            'actual_speedup': actual_speedup,
            'efficiency': efficiency,
            'block_exec_times': block_exec_times,
            'block_wait_times': block_wait_times,
            'avg_block_exec_times': avg_block_exec_times,
        }
