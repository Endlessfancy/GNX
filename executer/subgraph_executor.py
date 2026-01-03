"""
Subgraph Executor - Execute One Subgraph with PEP
单个subgraph执行器（支持灵活的1-3个block + data parallelism + pipeline）

支持 OpenVINO 异步推理
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor

# Import partitioner and NPU utils
try:
    from .node_partitioner import NodeBasedPartitioner
    from .npu_utils import pad_graph_data_for_npu, unpad_tensor_from_npu, pad_tensor_for_npu
except ImportError:
    from node_partitioner import NodeBasedPartitioner
    from npu_utils import pad_graph_data_for_npu, unpad_tensor_from_npu, pad_tensor_for_npu

# OpenVINO support
try:
    from openvino.runtime import CompiledModel, InferRequest
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False


class SubgraphExecutor:
    """
    Subgraph执行器

    功能:
    1. 执行单个subgraph的推理
    2. 支持灵活的1-3个block
    3. 支持data parallelism（多设备按ratio分割）
    4. 支持block间正确的数据传递
    5. 返回owned nodes的embeddings
    """

    def __init__(self, subgraph_id: int, subgraph_config: Dict, pep: List, models: Dict,
                 npu_static_nodes: int = 0, npu_static_edges: int = 0):
        """
        Args:
            subgraph_id: Subgraph ID
            subgraph_config: Subgraph配置 (from partition_config)
            pep: PEP定义 [[devices, stages, ratios], ...]
            models: {(block_id, device): compiled_model}
            npu_static_nodes: NPU模型的静态节点数（用于padding）
            npu_static_edges: NPU模型的静态边数（用于padding）
        """
        self.subgraph_id = subgraph_id
        self.subgraph_config = subgraph_config
        self.pep = pep
        self.models = models

        self.num_blocks = len(pep)
        self.n = subgraph_config['n']  # owned nodes
        self.n_pad = subgraph_config.get('n_pad', 0)

        # NPU 静态 shape 信息（用于 padding）
        self.npu_static_nodes = npu_static_nodes
        self.npu_static_edges = npu_static_edges

    def execute(self, edge_index: torch.Tensor, x: torch.Tensor,
                owned_nodes: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        执行推理

        Args:
            edge_index: [2, m] 边索引（局部ID）
            x: [n + num_ghosts, feat_dim] 节点特征（包含ghost）
            owned_nodes: [n] owned节点的局部ID（通常是[0, n)）

        Returns:
            (embeddings, execution_time)
            - embeddings: [n, hidden_dim] owned节点的输出
            - execution_time: 执行时间（ms）
        """
        start_time = time.time()

        # 准备初始数据
        num_nodes = x.size(0)
        current_data = {'x': x, 'edge_index': edge_index, 'num_nodes': num_nodes}

        # 逐block执行，传递中间结果
        for block_id, block in enumerate(self.pep):
            current_data = self._execute_block(
                block_id,
                block,
                current_data,
                owned_nodes
            )

        # 从最后的输出中提取owned nodes
        final_output = current_data.get('output', current_data.get('x'))

        # 确保提取owned nodes
        if final_output.size(0) == num_nodes:
            # Full-sized output
            output = final_output[owned_nodes]
        else:
            # Already owned-sized
            output = final_output

        execution_time = (time.time() - start_time) * 1000  # ms

        return output, execution_time

    def _execute_block(self, block_id: int, block: List,
                      input_data: Dict, owned_nodes: torch.Tensor) -> Dict:
        """
        执行单个block

        Args:
            block_id: Block ID
            block: [devices, stages, ratios]
            input_data: 输入数据字典 {'x': ..., 'edge_index': ..., 其他中间结果}
            owned_nodes: [n] owned节点ID

        Returns:
            output_data: 输出数据字典（传递给下一个block）
        """
        devices = block[0]  # ['CPU', 'GPU'] or ['NPU']
        stages = block[1]   # [1, 2, 3, 4, 5] or [6, 7]
        ratios = block[2] if len(block) > 2 else [1.0]  # [0.5, 0.5] or [1.0]

        if len(devices) == 1:
            # 单设备执行
            output_data = self._execute_single_device(
                block_id,
                devices[0],
                input_data,
                stages
            )
        else:
            # 数据并行执行
            output_data = self._execute_data_parallel(
                block_id,
                devices,
                ratios,
                input_data,
                owned_nodes,
                stages
            )

        return output_data

    def _execute_single_device(self, block_id: int, device: str,
                               input_data: Dict, stages: List[int]) -> Dict:
        """
        单设备执行

        Args:
            block_id: Block ID
            device: 'CPU', 'GPU', or 'NPU'
            input_data: 输入数据
            stages: 该block执行的stages

        Returns:
            output_data: 输出数据
        """
        model = self.models[(block_id, device)]

        # 准备模型输入（根据stages确定）
        model_input = self._prepare_model_input(input_data, stages)

        # NPU 需要 padding 到静态大小
        original_num_nodes = None
        if device == 'NPU' and self.npu_static_nodes > 0:
            model_input, original_num_nodes = self._pad_input_for_npu(model_input, stages)

        # 检查模型类型
        if OPENVINO_AVAILABLE and hasattr(model, 'create_infer_request'):
            # OpenVINO CompiledModel
            output = self._run_openvino_inference(model, model_input)

            # NPU 输出需要 unpad
            if device == 'NPU' and original_num_nodes is not None:
                output = unpad_tensor_from_npu(output, original_num_nodes, dim=0)

        elif hasattr(model, 'run'):
            # ONNX Runtime model
            input_names = [inp.name for inp in model.get_inputs()]

            # 转换为numpy
            numpy_inputs = self._convert_to_numpy(model_input)

            # 构建输入字典
            if len(input_names) == len(numpy_inputs):
                input_dict = {name: data for name, data in zip(input_names, numpy_inputs)}
            else:
                # Fallback
                input_dict = {f'input_{i}': data for i, data in enumerate(numpy_inputs)}

            # 推理
            outputs = model.run(None, input_dict)

            # 返回第一个输出（通常是node embeddings）
            output = torch.from_numpy(outputs[0])
        else:
            # PyTorch model
            with torch.no_grad():
                # Move inputs to model's device
                if device == 'GPU' and torch.cuda.is_available():
                    model_input = [t.cuda() if isinstance(t, torch.Tensor) else t for t in model_input]

                output = model(*model_input)

                # Move back to CPU if needed
                if isinstance(output, torch.Tensor) and output.is_cuda:
                    output = output.cpu()

        # 准备输出数据（根据stages确定下一步需要什么）
        output_data = self._prepare_next_input(output, stages, input_data)

        return output_data

    def _run_openvino_inference(self, compiled_model: Any, model_input: List) -> torch.Tensor:
        """
        使用 OpenVINO 执行同步推理

        Args:
            compiled_model: OpenVINO CompiledModel
            model_input: 模型输入列表

        Returns:
            output: PyTorch Tensor
        """
        # 创建推理请求
        infer_request = compiled_model.create_infer_request()

        # 获取输入名称
        input_names = [inp.any_name for inp in compiled_model.inputs]

        # 转换为numpy
        numpy_inputs = self._convert_to_numpy(model_input)

        # 设置输入
        for i, (name, data) in enumerate(zip(input_names, numpy_inputs)):
            infer_request.set_tensor(name, self._to_ov_tensor(data))

        # 同步推理
        infer_request.infer()

        # 获取输出
        output_tensor = infer_request.get_output_tensor(0)
        output = torch.from_numpy(output_tensor.data.copy())

        return output

    def create_async_infer_request(self, block_id: int, device: str) -> Optional[Any]:
        """
        创建异步推理请求

        Args:
            block_id: Block ID
            device: 设备名称

        Returns:
            infer_request: OpenVINO InferRequest 或 None
        """
        model = self.models[(block_id, device)]

        if OPENVINO_AVAILABLE and hasattr(model, 'create_infer_request'):
            return model.create_infer_request()

        return None

    def start_async_inference(self, infer_request: Any, input_data: Dict, stages: List[int],
                               device: str = None):
        """
        启动异步推理（立即返回）

        Args:
            infer_request: OpenVINO InferRequest
            input_data: 输入数据
            stages: 执行的 stages
            device: 设备名称（用于 NPU padding）
        """
        # 准备输入
        model_input = self._prepare_model_input(input_data, stages)

        # NPU 需要 padding 到静态大小
        original_num_nodes = None
        if device == 'NPU' and self.npu_static_nodes > 0:
            model_input, original_num_nodes = self._pad_input_for_npu(model_input, stages)
            # 保存原始节点数，供 wait_and_get_output 使用
            self._async_original_nodes = original_num_nodes

        numpy_inputs = self._convert_to_numpy(model_input)

        # 获取编译模型的输入名称
        compiled_model = infer_request.get_compiled_model()
        input_names = [inp.any_name for inp in compiled_model.inputs]

        # 设置输入
        for name, data in zip(input_names, numpy_inputs):
            infer_request.set_tensor(name, self._to_ov_tensor(data))

        # 启动异步推理
        infer_request.start_async()

    def wait_and_get_output(self, infer_request: Any, stages: List[int], input_data: Dict,
                            device: str = None) -> Dict:
        """
        等待异步推理完成并获取输出

        Args:
            infer_request: OpenVINO InferRequest
            stages: 执行的 stages
            input_data: 原始输入数据（用于构建输出）
            device: 设备名称（用于 NPU unpad）

        Returns:
            output_data: 输出数据字典
        """
        # 等待完成
        infer_request.wait()

        # 获取输出
        output_tensor = infer_request.get_output_tensor(0)
        output = torch.from_numpy(output_tensor.data.copy())

        # NPU 输出需要 unpad
        if device == 'NPU' and hasattr(self, '_async_original_nodes') and self._async_original_nodes is not None:
            output = unpad_tensor_from_npu(output, self._async_original_nodes, dim=0)
            self._async_original_nodes = None  # 清理

        # 准备输出数据
        output_data = self._prepare_next_input(output, stages, input_data)

        return output_data

    def _to_ov_tensor(self, data: np.ndarray):
        """
        将 numpy 数组转换为 OpenVINO Tensor
        """
        from openvino.runtime import Tensor
        return Tensor(data)

    def _run_partition_on_device(self, block_id: int, device: str,
                                 partition: Dict, partition_input: List,
                                 stages: List[int] = None) -> torch.Tensor:
        """
        在单个设备上执行partition的推理（供ThreadPoolExecutor调用）

        Args:
            block_id: Block ID
            device: 设备名称 ('CPU', 'GPU', 'NPU')
            partition: 分区信息
            partition_input: 模型输入数据
            stages: 执行的 stages（用于 NPU padding）

        Returns:
            device_output: 该设备的输出tensor
        """
        model = self.models[(block_id, device)]

        # NPU 需要 padding 到静态大小
        original_num_nodes = None
        if device == 'NPU' and self.npu_static_nodes > 0 and stages is not None:
            partition_input, original_num_nodes = self._pad_input_for_npu(partition_input, stages)

        # OpenVINO CompiledModel
        if OPENVINO_AVAILABLE and hasattr(model, 'create_infer_request'):
            device_output = self._run_openvino_inference(model, partition_input)

            # NPU 输出需要 unpad
            if device == 'NPU' and original_num_nodes is not None:
                device_output = unpad_tensor_from_npu(device_output, original_num_nodes, dim=0)

        elif hasattr(model, 'run'):
            # ONNX Runtime
            input_names = [inp.name for inp in model.get_inputs()]
            numpy_inputs = self._convert_to_numpy(partition_input)

            if len(input_names) == len(numpy_inputs):
                input_dict = {name: data for name, data in zip(input_names, numpy_inputs)}
            else:
                input_dict = {f'input_{i}': data for i, data in enumerate(numpy_inputs)}

            model_outputs = model.run(None, input_dict)
            device_output = torch.from_numpy(model_outputs[0])
        else:
            # PyTorch
            with torch.no_grad():
                partition_input_tensors = partition_input
                if device == 'GPU' and torch.cuda.is_available():
                    partition_input_tensors = [t.cuda() if isinstance(t, torch.Tensor) else t
                                              for t in partition_input_tensors]

                device_output = model(*partition_input_tensors)

                if isinstance(device_output, torch.Tensor) and device_output.is_cuda:
                    device_output = device_output.cpu()

        return device_output

    def _execute_data_parallel(self, block_id: int, devices: List[str],
                               ratios: List[float], input_data: Dict,
                               owned_nodes: torch.Tensor, stages: List[int]) -> Dict:
        """
        数据并行执行（多设备） - 真正并发执行

        策略: 按ratio分割owned_nodes，每个设备并发处理一部分

        Args:
            block_id: Block ID
            devices: ['CPU', 'GPU']
            ratios: [0.5, 0.5]
            input_data: 输入数据
            owned_nodes: [n] owned节点ID
            stages: 该block执行的stages

        Returns:
            output_data: 合并后的输出数据
        """
        x = input_data['x']
        edge_index = input_data['edge_index']
        num_nodes = input_data.get('num_nodes', x.size(0))

        # 使用NodePartitioner分割数据
        partitions = NodeBasedPartitioner.partition_by_nodes(
            edge_index, num_nodes, ratios, x
        )

        # 并发执行多个设备
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = []
            for partition, device in zip(partitions, devices):
                # 准备该设备的输入
                partition_input = self._prepare_partition_input(input_data, partition, stages)

                # 提交任务到线程池（传递 stages 用于 NPU padding）
                future = executor.submit(
                    self._run_partition_on_device,
                    block_id, device, partition, partition_input, stages
                )
                futures.append(future)

            # 等待所有设备完成并收集结果
            outputs = [f.result() for f in futures]

        # 合并输出
        merged_output = NodeBasedPartitioner.merge_outputs(outputs, partitions)

        # 准备下一个block的输入
        output_data = self._prepare_next_input(merged_output, stages, input_data)

        return output_data

    def _prepare_model_input(self, input_data: Dict, stages: List[int]) -> List:
        """
        根据stages准备模型输入

        Args:
            input_data: 输入数据字典
            stages: 该block执行的stages

        Returns:
            模型输入列表
        """
        first_stage = stages[0]

        if first_stage == 1:
            # Stage 1-N: 从原始输入开始
            return [input_data['x'], input_data['edge_index']]

        elif first_stage == 5:
            # Stage 5-7: 需要sum_agg, count, x
            return [input_data['sum_agg'], input_data['count'], input_data['x']]

        elif first_stage == 6:
            # Stage 6-7: 需要mean_agg和x
            return [input_data['mean_agg'], input_data['x']]

        else:
            # 其他情况：尝试从input_data提取
            # 这是简化版，实际可能需要更复杂的逻辑
            if 'intermediate_output' in input_data:
                return [input_data['intermediate_output']]
            else:
                return [input_data['x'], input_data['edge_index']]

    def _prepare_partition_input(self, input_data: Dict, partition: Dict, stages: List[int]) -> List:
        """
        为partition准备模型输入

        Args:
            input_data: 全局输入数据
            partition: 分区信息
            stages: 执行的stages

        Returns:
            模型输入列表
        """
        first_stage = stages[0]

        if first_stage == 1:
            # Stage 1-N: 使用x_full和partition的edge_index
            return [partition['x_full'], partition['edge_index']]

        elif first_stage == 5:
            # Stage 5-7: 使用sum_agg, count, x_owned
            sum_agg = input_data.get('sum_agg')
            count = input_data.get('count')
            start, end = partition['node_range']

            # Extract partition's owned nodes
            if sum_agg.size(0) == partition['total_num_nodes']:
                sum_agg_part = sum_agg[start:end]
            else:
                sum_agg_part = sum_agg

            if count.size(0) == partition['total_num_nodes']:
                count_part = count[start:end]
            else:
                count_part = count

            return [sum_agg_part, count_part, partition['x_owned']]

        elif first_stage == 6:
            # Stage 6-7: 使用mean_agg和x_owned
            # 注意：mean_agg应该是从block 0传递过来的，已经是全图大小
            mean_agg = input_data.get('mean_agg')
            start, end = partition['node_range']

            if mean_agg.size(0) == partition['total_num_nodes']:
                # Full-sized mean_agg，提取owned部分
                return [mean_agg[start:end], partition['x_owned']]
            else:
                # 已经是partition-sized
                return [mean_agg, partition['x_owned']]

        else:
            # 默认处理
            return [partition['x_full'], partition['edge_index']]

    def _prepare_next_input(self, output: torch.Tensor, completed_stages: List[int],
                           input_data: Dict) -> Dict:
        """
        准备传递给下一个block的输入

        Args:
            output: 当前block的输出
            completed_stages: 当前block执行的stages
            input_data: 原始输入数据

        Returns:
            下一个block需要的输入数据
        """
        last_stage = completed_stages[-1]

        if last_stage == 4:
            # Stage 1-4完成，输出是concatenated [sum_agg, count]
            # 需要分离: sum_agg: [num_nodes, 256], count: [num_nodes, 1]
            # 下一个block (Stage 5-7)需要: (sum_agg, count, x)
            sum_agg = output[:, :-1]  # All except last column: [num_nodes, 256]
            count = output[:, -1]  # Last column: [num_nodes] (squeezed)
            return {
                'sum_agg': sum_agg,
                'count': count,
                'x': input_data['x'],  # 原始特征
                'edge_index': input_data['edge_index'],
                'num_nodes': input_data['num_nodes']
            }

        elif last_stage == 5:
            # Stage 1-5完成，输出是mean_agg
            # 下一个block (Stage 6-7)需要: (mean_agg, x)
            return {
                'mean_agg': output,
                'x': input_data['x'],  # 原始特征
                'edge_index': input_data['edge_index'],
                'num_nodes': input_data['num_nodes']
            }

        elif last_stage == 7:
            # Stage 6-7完成，输出是最终embeddings
            return {
                'output': output,
                'x': output,  # 也保存为x，防止后续访问
                'edge_index': input_data['edge_index'],
                'num_nodes': input_data.get('num_nodes', output.size(0))
            }

        else:
            # 其他情况：保持输出作为下一步的输入
            return {
                'intermediate_output': output,
                'x': input_data['x'],
                'edge_index': input_data['edge_index'],
                'num_nodes': input_data['num_nodes']
            }

    def _convert_to_numpy(self, tensors: List) -> List[np.ndarray]:
        """将torch.Tensor转换为numpy数组（使用FP32精度以匹配OpenVINO GPU）"""
        numpy_arrays = []
        for t in tensors:
            if isinstance(t, torch.Tensor):
                # 使用 FP32 精度以匹配 OpenVINO 模型（GPU 兼容性）
                if t.dtype in [torch.float32, torch.float16]:
                    numpy_arrays.append(t.cpu().numpy().astype(np.float32))
                else:
                    numpy_arrays.append(t.cpu().numpy().astype(np.int64))
            elif isinstance(t, int):
                numpy_arrays.append(np.array(t, dtype=np.int64))
            else:
                numpy_arrays.append(t)
        return numpy_arrays

    def _pad_input_for_npu(self, model_input: List, stages: List[int]) -> Tuple[List, int]:
        """
        为 NPU 静态模型 padding 输入数据

        Args:
            model_input: 模型输入列表
            stages: 执行的 stages

        Returns:
            (padded_input, original_num_nodes)
        """
        first_stage = stages[0]

        if first_stage == 1:
            # Stage 1-N: input is (x, edge_index)
            x, edge_index = model_input[0], model_input[1]
            x_padded, edge_index_padded, original_nodes, _ = pad_graph_data_for_npu(
                x, edge_index,
                self.npu_static_nodes, self.npu_static_edges
            )
            return [x_padded, edge_index_padded], original_nodes

        elif first_stage == 5:
            # Stage 5-7: input is (sum_agg, count, x)
            sum_agg, count, x = model_input[0], model_input[1], model_input[2]
            original_nodes = sum_agg.size(0)

            sum_agg_padded, _ = pad_tensor_for_npu(sum_agg, self.npu_static_nodes)
            count_padded, _ = pad_tensor_for_npu(count, self.npu_static_nodes)
            x_padded, _ = pad_tensor_for_npu(x, self.npu_static_nodes)

            return [sum_agg_padded, count_padded, x_padded], original_nodes

        elif first_stage == 6:
            # Stage 6-7: input is (mean_agg, x)
            mean_agg, x = model_input[0], model_input[1]
            original_nodes = mean_agg.size(0)

            mean_agg_padded, _ = pad_tensor_for_npu(mean_agg, self.npu_static_nodes)
            x_padded, _ = pad_tensor_for_npu(x, self.npu_static_nodes)

            return [mean_agg_padded, x_padded], original_nodes

        else:
            # 其他情况：不 padding
            return model_input, model_input[0].size(0) if isinstance(model_input[0], torch.Tensor) else 0
