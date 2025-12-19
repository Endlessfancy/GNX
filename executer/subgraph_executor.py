"""
Subgraph Executor - Execute One Subgraph with PEP
单个subgraph执行器（支持灵活的1-3个block）
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


class SubgraphExecutor:
    """
    Subgraph执行器

    功能:
    1. 执行单个subgraph的推理
    2. 支持灵活的1-3个block
    3. 支持data parallelism（多设备按ratio分割）
    4. 返回owned nodes的embeddings
    """

    def __init__(self, subgraph_id: int, subgraph_config: Dict, pep: List, models: Dict):
        """
        Args:
            subgraph_id: Subgraph ID
            subgraph_config: Subgraph配置 (from partition_config)
            pep: PEP定义 [[devices, stages, ratios], ...]
            models: {(block_id, device): compiled_model}
        """
        self.subgraph_id = subgraph_id
        self.subgraph_config = subgraph_config
        self.pep = pep
        self.models = models

        self.num_blocks = len(pep)
        self.n = subgraph_config['n']  # owned nodes
        self.n_pad = subgraph_config['n_pad']

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

        # 当前特征（会被每个block更新）
        current_x = x

        # 逐block执行
        for block_id, block in enumerate(self.pep):
            current_x = self._execute_block(
                block_id,
                block,
                current_x,
                edge_index,
                owned_nodes
            )

        # 只返回owned nodes的embeddings
        output = current_x[owned_nodes]

        execution_time = (time.time() - start_time) * 1000  # ms

        return output, execution_time

    def _execute_block(self, block_id: int, block: List,
                      x: torch.Tensor, edge_index: torch.Tensor,
                      owned_nodes: torch.Tensor) -> torch.Tensor:
        """
        执行单个block

        Args:
            block_id: Block ID
            block: [devices, stages, ratios]
            x: [total_nodes, feat_dim] 输入特征
            edge_index: [2, m] 边索引
            owned_nodes: [n] owned节点ID

        Returns:
            output: [total_nodes, hidden_dim] 输出特征
        """
        devices = block[0]  # ['CPU', 'GPU']
        stages = block[1]   # [1, 2, 3, 4, 5, 6, 7]
        ratios = block[2]   # [0.5, 0.5]

        if len(devices) == 1:
            # 单设备执行
            return self._execute_single_device(
                block_id,
                devices[0],
                x,
                edge_index
            )
        else:
            # 数据并行执行
            return self._execute_data_parallel(
                block_id,
                devices,
                ratios,
                x,
                edge_index,
                owned_nodes
            )

    def _execute_single_device(self, block_id: int, device: str,
                               x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        单设备执行

        Args:
            block_id: Block ID
            device: 'CPU', 'GPU', or 'NPU'
            x: [n, feat_dim] 输入
            edge_index: [2, m] 边

        Returns:
            output: [n, hidden_dim] 输出
        """
        model = self.models[(block_id, device)]

        # Check if model is ONNX Runtime session or PyTorch model
        if hasattr(model, 'run'):
            # ONNX Runtime model
            # Get actual input names from ONNX model
            input_names = [inp.name for inp in model.get_inputs()]

            # Prepare input dictionary with actual ONNX input names
            x_np = x.cpu().numpy().astype(np.float32)
            edge_index_np = edge_index.cpu().numpy().astype(np.int64)

            if len(input_names) == 2:
                input_dict = {
                    input_names[0]: x_np,
                    input_names[1]: edge_index_np
                }
            else:
                # Fallback to common names
                input_dict = {
                    'x': x_np,
                    'edge_index': edge_index_np
                }

            # 推理
            outputs = model.run(None, input_dict)

            # 返回第一个输出（通常是node embeddings）
            output = torch.from_numpy(outputs[0])
        else:
            # PyTorch model
            with torch.no_grad():
                # Move inputs to model's device
                if device == 'GPU' and torch.cuda.is_available():
                    x = x.cuda()
                    edge_index = edge_index.cuda()

                output = model(x, edge_index)

                # Move back to CPU if needed
                if output.is_cuda:
                    output = output.cpu()

        return output

    def _execute_data_parallel(self, block_id: int, devices: List[str],
                               ratios: List[float], x: torch.Tensor,
                               edge_index: torch.Tensor, owned_nodes: torch.Tensor) -> torch.Tensor:
        """
        数据并行执行（多设备）

        策略: 按ratio分割owned_nodes，每个设备处理一部分

        Args:
            block_id: Block ID
            devices: ['CPU', 'GPU']
            ratios: [0.5, 0.5]
            x: [total_nodes, feat_dim] 输入
            edge_index: [2, m] 边
            owned_nodes: [n] owned节点ID

        Returns:
            output: [total_nodes, hidden_dim] 输出（只更新owned_nodes部分）
        """
        num_owned = len(owned_nodes)
        outputs = []

        # 按ratio分割owned_nodes
        node_splits = self._split_nodes_by_ratio(num_owned, ratios)

        for device, node_range in zip(devices, node_splits):
            # 该设备负责的节点（局部ID）
            target_nodes = owned_nodes[node_range[0]:node_range[1]]

            # 过滤边：只保留目标节点在target_nodes中的边
            mask = torch.isin(edge_index[1], target_nodes)
            device_edges = edge_index[:, mask]

            # 执行推理
            model = self.models[(block_id, device)]

            if hasattr(model, 'run'):
                # ONNX Runtime model
                # Get actual input names from ONNX model
                input_names = [inp.name for inp in model.get_inputs()]

                x_np = x.cpu().numpy().astype(np.float32)
                edge_np = device_edges.cpu().numpy().astype(np.int64)

                if len(input_names) == 2:
                    input_dict = {
                        input_names[0]: x_np,
                        input_names[1]: edge_np
                    }
                else:
                    input_dict = {
                        'x': x_np,
                        'edge_index': edge_np
                    }

                model_outputs = model.run(None, input_dict)
                device_output = torch.from_numpy(model_outputs[0])
            else:
                # PyTorch model
                with torch.no_grad():
                    x_input = x
                    edge_input = device_edges

                    if device == 'GPU' and torch.cuda.is_available():
                        x_input = x_input.cuda()
                        edge_input = edge_input.cuda()

                    device_output = model(x_input, edge_input)

                    if device_output.is_cuda:
                        device_output = device_output.cpu()

            # 只保留target_nodes的输出
            outputs.append((target_nodes, device_output[target_nodes]))

        # 合并输出
        output = torch.zeros(x.shape[0], outputs[0][1].shape[1])  # [total_nodes, hidden_dim]

        for target_nodes, partial_output in outputs:
            output[target_nodes] = partial_output

        return output

    def _split_nodes_by_ratio(self, num_nodes: int, ratios: List[float]) -> List[Tuple[int, int]]:
        """
        按ratio分割节点

        Args:
            num_nodes: 节点总数
            ratios: [0.5, 0.5] 分割比例

        Returns:
            [(start, end), ...] 每个设备的节点范围
        """
        splits = []
        start = 0

        for ratio in ratios[:-1]:
            end = start + int(num_nodes * ratio)
            splits.append((start, end))
            start = end

        # 最后一个设备拿剩余的
        splits.append((start, num_nodes))

        return splits
