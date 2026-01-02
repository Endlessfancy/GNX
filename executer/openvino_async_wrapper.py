"""
OpenVINO Async Inference Wrapper
OpenVINO 异步推理包装器

提供简单的异步推理接口，支持:
- start_async(): 启动异步推理（立即返回）
- wait(): 等待推理完成
- get_output(): 获取推理结果
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any

try:
    from openvino.runtime import Core, CompiledModel, InferRequest, AsyncInferQueue
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False


class OpenVINOInferRequest:
    """
    单个 OpenVINO 推理请求的包装器

    封装 InferRequest，提供简单的异步接口
    """

    def __init__(self, compiled_model: 'CompiledModel'):
        """
        Args:
            compiled_model: OpenVINO 编译后的模型
        """
        self.compiled_model = compiled_model
        self.infer_request = compiled_model.create_infer_request()
        self._is_running = False

    def start_async(self, inputs: Dict[str, np.ndarray]):
        """
        启动异步推理（立即返回）

        Args:
            inputs: 输入数据字典 {input_name: numpy_array}
        """
        # 设置输入
        for name, data in inputs.items():
            self.infer_request.set_tensor(name, self._to_ov_tensor(data))

        # 启动异步推理
        self.infer_request.start_async()
        self._is_running = True

    def wait(self):
        """
        等待推理完成
        """
        if self._is_running:
            self.infer_request.wait()
            self._is_running = False

    def get_output(self, output_index: int = 0) -> np.ndarray:
        """
        获取推理输出

        Args:
            output_index: 输出索引（默认为第一个输出）

        Returns:
            output: numpy 数组
        """
        output_tensor = self.infer_request.get_output_tensor(output_index)
        return output_tensor.data.copy()

    def get_output_tensor(self, output_index: int = 0) -> torch.Tensor:
        """
        获取推理输出（返回 PyTorch Tensor）

        Args:
            output_index: 输出索引

        Returns:
            output: PyTorch Tensor
        """
        np_output = self.get_output(output_index)
        return torch.from_numpy(np_output)

    def _to_ov_tensor(self, data: Any):
        """
        将数据转换为 OpenVINO tensor
        """
        from openvino.runtime import Tensor

        if isinstance(data, torch.Tensor):
            np_data = data.cpu().numpy()
        elif isinstance(data, np.ndarray):
            np_data = data
        else:
            np_data = np.array(data)

        return Tensor(np_data)


class OpenVINOAsyncExecutor:
    """
    OpenVINO 异步执行器

    管理多个推理请求，支持批量异步执行
    """

    def __init__(self, compiled_models: Dict[str, 'CompiledModel']):
        """
        Args:
            compiled_models: {model_key: compiled_model} 字典
        """
        self.compiled_models = compiled_models

        # 为每个模型创建推理请求池
        self.request_pools: Dict[str, List[OpenVINOInferRequest]] = {}
        self._init_request_pools()

    def _init_request_pools(self, pool_size: int = 8):
        """
        初始化推理请求池

        Args:
            pool_size: 每个模型的请求池大小
        """
        for model_key, compiled_model in self.compiled_models.items():
            self.request_pools[model_key] = [
                OpenVINOInferRequest(compiled_model)
                for _ in range(pool_size)
            ]

    def get_request(self, model_key: str) -> OpenVINOInferRequest:
        """
        获取一个可用的推理请求

        Args:
            model_key: 模型标识符

        Returns:
            infer_request: OpenVINOInferRequest 实例
        """
        if model_key not in self.request_pools:
            raise KeyError(f"Model not found: {model_key}")

        # 简单的轮询策略，实际使用时可以优化
        pool = self.request_pools[model_key]

        # 找一个空闲的请求
        for request in pool:
            if not request._is_running:
                return request

        # 如果没有空闲的，等待第一个完成
        pool[0].wait()
        return pool[0]

    def create_infer_request(self, model_key: str) -> OpenVINOInferRequest:
        """
        为指定模型创建新的推理请求

        Args:
            model_key: 模型标识符

        Returns:
            infer_request: 新的推理请求
        """
        if model_key not in self.compiled_models:
            raise KeyError(f"Model not found: {model_key}")

        return OpenVINOInferRequest(self.compiled_models[model_key])


def prepare_inputs_for_openvino(input_data: Dict, stages: List[int]) -> Dict[str, np.ndarray]:
    """
    准备 OpenVINO 推理的输入数据

    Args:
        input_data: 原始输入数据字典
        stages: 执行的 stages

    Returns:
        inputs: {input_name: numpy_array} 格式的输入
    """
    first_stage = stages[0]

    def to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.cpu().numpy().astype(np.float32)
        return np.array(t, dtype=np.float32)

    if first_stage == 1:
        # Stage 1-N: 需要 x 和 edge_index
        return {
            'x': to_numpy(input_data['x']),
            'edge_index': input_data['edge_index'].cpu().numpy().astype(np.int64)
        }

    elif first_stage == 5:
        # Stage 5-7: 需要 sum_agg, count, x
        return {
            'sum_agg': to_numpy(input_data['sum_agg']),
            'count': to_numpy(input_data['count']),
            'x': to_numpy(input_data['x'])
        }

    elif first_stage == 6:
        # Stage 6-7: 需要 mean_agg 和 x
        return {
            'mean_agg': to_numpy(input_data['mean_agg']),
            'x': to_numpy(input_data['x'])
        }

    else:
        # 默认情况
        return {
            'x': to_numpy(input_data['x']),
            'edge_index': input_data['edge_index'].cpu().numpy().astype(np.int64)
        }
