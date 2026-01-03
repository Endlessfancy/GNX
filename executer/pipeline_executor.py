"""
Pipeline Executor - Block-level Pipeline Parallelism
Block级别流水线并行执行器

支持 OpenVINO 异步推理模式
"""

import time
import torch
import threading
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

# OpenVINO support
try:
    from openvino.runtime import CompiledModel, InferRequest
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False


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

        # 详细时间统计
        self.detailed_timing = {
            'block_times': {},      # {(sg_id, block_id): {'wait': t, 'exec': t, 'total': t}}
            'timestamps': {},       # {(sg_id, block_id): {'start': t, 'end': t}}
            'thread_ids': {},       # {block_id: thread_id}
        }
        self.timing_lock = threading.Lock()

        # 保存实际流水线时间（用于性能分析）
        self.actual_pipeline_time = 0.0

        # 异步推理模式标志
        self.use_async_inference = False

    def execute_pipeline(self, use_async: bool = False) -> Dict:
        """
        执行流水线并行

        Args:
            use_async: 是否使用 OpenVINO 异步推理模式

        Returns:
            {
                'embeddings': torch.Tensor [total_nodes, embed_dim],
                'per_subgraph_times': List[float],  # 每个subgraph的总时间
                'total_time': float  # 总执行时间（ms）
            }
        """
        self.use_async_inference = use_async and OPENVINO_AVAILABLE

        if self.use_async_inference:
            print(f"  Starting pipeline execution with {self.num_blocks} blocks (OpenVINO Async Mode)...")
        else:
            print(f"  Starting pipeline execution with {self.num_blocks} blocks...")

        start_time = time.time()

        # 创建线程池，每个block一个worker
        with ThreadPoolExecutor(max_workers=self.num_blocks) as executor:
            futures = []
            for block_id in range(self.num_blocks):
                if self.use_async_inference:
                    future = executor.submit(self._block_worker_async, block_id)
                else:
                    future = executor.submit(self._block_worker, block_id)
                futures.append(future)

            # 等待所有block完成
            for future in futures:
                future.result()

        # 收集最终结果（从最后一个block的输出中）
        final_embeddings = self._collect_final_embeddings()

        total_time = (time.time() - start_time) * 1000  # ms
        self.actual_pipeline_time = total_time  # 保存用于性能分析

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
        import threading as th
        thread_id = th.get_ident()

        # 记录线程ID
        with self.timing_lock:
            self.detailed_timing['thread_ids'][block_id] = thread_id

        for sg_id in self.subgraph_ids:
            total_start_time = time.time()
            wait_time = 0.0

            # 记录开始时间戳
            with self.timing_lock:
                self.detailed_timing['timestamps'][(sg_id, block_id)] = {
                    'start': total_start_time
                }

            # 日志：开始处理
            print(f"  [Block{block_id} Thread-{thread_id % 10000}] Starting SG{sg_id} at {total_start_time:.3f}s")

            # 获取输入数据
            if block_id == 0:
                # Block 0: 使用原始subgraph数据
                input_data = self._get_raw_subgraph_data(sg_id)
            else:
                # Block N: 等待Block N-1完成此subgraph
                prev_block_id = block_id - 1

                wait_start_time = time.time()
                self.completion_events[(sg_id, prev_block_id)].wait()
                wait_time = (time.time() - wait_start_time) * 1000  # ms

                # 获取前一个block的输出
                input_data = self._get_intermediate_result(sg_id, prev_block_id)

            # 执行当前block（会调用SubgraphExecutor执行单个block）
            exec_start_time = time.time()
            output_data = self._execute_single_block(sg_id, block_id, input_data)
            exec_time = (time.time() - exec_start_time) * 1000  # ms

            # 保存结果
            self._save_intermediate_result(sg_id, block_id, output_data)

            # 发送完成信号，通知下一个block
            self.completion_events[(sg_id, block_id)].set()

            # 计算总时间
            total_end_time = time.time()
            total_time = (total_end_time - total_start_time) * 1000  # ms

            # 记录结束时间戳
            with self.timing_lock:
                self.detailed_timing['timestamps'][(sg_id, block_id)]['end'] = total_end_time

            # 保存详细统计
            with self.timing_lock:
                self.detailed_timing['block_times'][(sg_id, block_id)] = {
                    'wait': wait_time,
                    'exec': exec_time,
                    'total': total_time
                }

            # 日志：完成处理
            print(f"  [Block{block_id} Thread-{thread_id % 10000}] Finished SG{sg_id}: wait={wait_time:.0f}ms, exec={exec_time:.0f}ms, total={total_time:.0f}ms")

            # 最后一个block额外打印汇总（兼容原有输出）
            if block_id == self.num_blocks - 1:
                print(f"  Subgraph {sg_id} completed: {total_time:.2f}ms")

    def _block_worker_async(self, block_id: int):
        """
        Block worker线程（OpenVINO 异步推理模式）

        使用 launch all → wait all 模式：
        1. 为所有 subgraph 启动异步推理
        2. 批量等待推理完成
        3. 通知下一个 block

        这种模式可以最大化硬件利用率，减少 GIL 竞争

        Args:
            block_id: 当前block的ID (0 to num_blocks-1)
        """
        import threading as th
        thread_id = th.get_ident()

        # 记录线程ID
        with self.timing_lock:
            self.detailed_timing['thread_ids'][block_id] = thread_id

        block = self.pep[block_id]
        devices = block[0]  # ['CPU', 'GPU'] or ['NPU']
        stages = block[1]

        # 对于单设备情况，使用异步推理
        if len(devices) == 1:
            device = devices[0]
            self._run_block_single_device_async(block_id, device, stages, thread_id)
        else:
            # 多设备数据并行，回退到同步模式
            # (后续可以扩展为异步数据并行)
            for sg_id in self.subgraph_ids:
                total_start_time = time.time()
                wait_time = 0.0

                with self.timing_lock:
                    self.detailed_timing['timestamps'][(sg_id, block_id)] = {
                        'start': total_start_time
                    }

                print(f"  [Block{block_id} Thread-{thread_id % 10000}] Starting SG{sg_id} (data parallel)")

                if block_id == 0:
                    input_data = self._get_raw_subgraph_data(sg_id)
                else:
                    prev_block_id = block_id - 1
                    wait_start_time = time.time()
                    self.completion_events[(sg_id, prev_block_id)].wait()
                    wait_time = (time.time() - wait_start_time) * 1000

                    input_data = self._get_intermediate_result(sg_id, prev_block_id)

                exec_start_time = time.time()
                output_data = self._execute_single_block(sg_id, block_id, input_data)
                exec_time = (time.time() - exec_start_time) * 1000

                self._save_intermediate_result(sg_id, block_id, output_data)
                self.completion_events[(sg_id, block_id)].set()

                total_end_time = time.time()
                total_time = (total_end_time - total_start_time) * 1000

                with self.timing_lock:
                    self.detailed_timing['timestamps'][(sg_id, block_id)]['end'] = total_end_time
                    self.detailed_timing['block_times'][(sg_id, block_id)] = {
                        'wait': wait_time,
                        'exec': exec_time,
                        'total': total_time
                    }

                print(f"  [Block{block_id} Thread-{thread_id % 10000}] Finished SG{sg_id}: wait={wait_time:.0f}ms, exec={exec_time:.0f}ms")

                if block_id == self.num_blocks - 1:
                    print(f"  Subgraph {sg_id} completed: {total_time:.2f}ms")

    def _run_block_single_device_async(self, block_id: int, device: str, stages: List[int], thread_id: int):
        """
        单设备异步推理模式

        Launch all → Wait all 模式：
        1. 遍历所有 subgraph，为每个启动异步推理
        2. 在需要等待前一个 block 完成时等待
        3. 批量收集结果并通知下一个 block

        Args:
            block_id: Block ID
            device: 设备名称
            stages: 执行的 stages
            thread_id: 线程 ID
        """
        # Phase 1: 准备所有推理请求
        pending_inferences = []  # [(sg_id, infer_request, input_data, start_time)]

        for sg_id in self.subgraph_ids:
            total_start_time = time.time()
            wait_time = 0.0

            with self.timing_lock:
                self.detailed_timing['timestamps'][(sg_id, block_id)] = {
                    'start': total_start_time
                }

            # 获取输入数据
            if block_id == 0:
                input_data = self._get_raw_subgraph_data(sg_id)
            else:
                prev_block_id = block_id - 1
                wait_start_time = time.time()
                self.completion_events[(sg_id, prev_block_id)].wait()
                wait_time = (time.time() - wait_start_time) * 1000

                input_data = self._get_intermediate_result(sg_id, prev_block_id)

            # 获取 executor 和创建异步推理请求
            executor = self.subgraph_executors[sg_id]
            infer_request = executor.create_async_infer_request(block_id, device)

            if infer_request is not None:
                # 启动异步推理
                print(f"  [Block{block_id} Thread-{thread_id % 10000}] Launching async SG{sg_id}")
                executor.start_async_inference(infer_request, input_data, stages)
                pending_inferences.append((sg_id, infer_request, input_data, total_start_time, wait_time))
            else:
                # Fallback 到同步模式
                print(f"  [Block{block_id} Thread-{thread_id % 10000}] Starting SG{sg_id} (sync fallback)")
                exec_start_time = time.time()
                output_data = self._execute_single_block(sg_id, block_id, input_data)
                exec_time = (time.time() - exec_start_time) * 1000

                self._save_intermediate_result(sg_id, block_id, output_data)
                self.completion_events[(sg_id, block_id)].set()

                total_end_time = time.time()
                total_time = (total_end_time - total_start_time) * 1000

                with self.timing_lock:
                    self.detailed_timing['timestamps'][(sg_id, block_id)]['end'] = total_end_time
                    self.detailed_timing['block_times'][(sg_id, block_id)] = {
                        'wait': wait_time,
                        'exec': exec_time,
                        'total': total_time
                    }

                print(f"  [Block{block_id} Thread-{thread_id % 10000}] Finished SG{sg_id}: wait={wait_time:.0f}ms, exec={exec_time:.0f}ms")

        # Phase 2: 等待所有异步推理完成并收集结果
        for sg_id, infer_request, input_data, start_time, wait_time in pending_inferences:
            exec_start_time = time.time()

            # 等待推理完成并获取输出
            executor = self.subgraph_executors[sg_id]
            output_data = executor.wait_and_get_output(infer_request, stages, input_data)

            exec_time = (time.time() - exec_start_time) * 1000

            # 保存结果并通知
            self._save_intermediate_result(sg_id, block_id, output_data)
            self.completion_events[(sg_id, block_id)].set()

            total_end_time = time.time()
            total_time = (total_end_time - start_time) * 1000

            with self.timing_lock:
                self.detailed_timing['timestamps'][(sg_id, block_id)]['end'] = total_end_time
                self.detailed_timing['block_times'][(sg_id, block_id)] = {
                    'wait': wait_time,
                    'exec': exec_time,
                    'total': total_time
                }

            print(f"  [Block{block_id} Thread-{thread_id % 10000}] Finished SG{sg_id}: wait={wait_time:.0f}ms, exec={exec_time:.0f}ms (async)")

            if block_id == self.num_blocks - 1:
                print(f"  Subgraph {sg_id} completed: {total_time:.2f}ms")

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

        # 合并为全局tensor（使用 FP16 以匹配模型输出精度）
        total_nodes = self.data_loader.full_data.num_nodes
        embed_dim = all_embeddings_list[0][1].size(1)
        final_tensor = torch.zeros(total_nodes, embed_dim, dtype=torch.float16)

        for global_ids, embeds in all_embeddings_list:
            final_tensor[global_ids] = embeds

        return final_tensor

    def analyze_performance(self) -> Dict:
        """
        分析流水线性能

        Returns:
            性能分析结果字典
        """
        # 计算每个block的总执行时间（不含等待）
        block_exec_times = {}
        block_wait_times = {}

        for block_id in range(self.num_blocks):
            exec_sum = 0.0
            wait_sum = 0.0

            for sg_id in self.subgraph_ids:
                timing = self.detailed_timing['block_times'].get((sg_id, block_id))
                if timing:
                    exec_sum += timing['exec']
                    wait_sum += timing['wait']

            block_exec_times[block_id] = exec_sum
            block_wait_times[block_id] = wait_sum

        # 理论顺序执行时间（所有block串行执行）
        sequential_time = sum(block_exec_times.values())

        # 实际流水线时间
        pipeline_time = self.actual_pipeline_time

        # 理论最大加速比（假设无overhead）
        theoretical_speedup = self.num_blocks

        # 实际加速比
        actual_speedup = sequential_time / pipeline_time if pipeline_time > 0 else 0.0

        # 流水线效率
        efficiency = actual_speedup / theoretical_speedup if theoretical_speedup > 0 else 0.0

        # 计算每个block的平均执行时间
        avg_block_exec_times = {
            block_id: exec_time / len(self.subgraph_ids)
            for block_id, exec_time in block_exec_times.items()
        }

        return {
            'sequential_time': sequential_time,
            'pipeline_time': pipeline_time,
            'theoretical_speedup': theoretical_speedup,
            'actual_speedup': actual_speedup,
            'efficiency': efficiency,
            'block_exec_times': block_exec_times,
            'block_wait_times': block_wait_times,
            'avg_block_exec_times': avg_block_exec_times,
        }
