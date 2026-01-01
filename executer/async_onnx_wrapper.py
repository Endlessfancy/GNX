"""
Async ONNX Runtime Wrapper
【阶段3优化】将ONNX Runtime推理封装为异步接口，实现真正的并行执行
"""

import asyncio
import torch
import numpy as np
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor


class AsyncONNXWrapper:
    """
    异步ONNX Runtime包装器

    功能:
    1. 将同步的ONNX Runtime推理调用转换为异步
    2. 使用独立的线程池执行推理（避免阻塞事件循环）
    3. 支持多个模型并发推理

    原理:
    - ONNX Runtime的C++后端在推理时会释放GIL
    - 使用asyncio.run_in_executor()将推理提交到线程池
    - 事件循环协调多个推理任务的并发执行
    """

    def __init__(self, max_workers: int = 4):
        """
        Args:
            max_workers: 线程池大小（建议 = 设备数量）
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loop = None  # Will be set when running in async context

    async def infer_async(self, model, inputs: List, device: str) -> torch.Tensor:
        """
        异步推理接口

        Args:
            model: ONNX Runtime session or PyTorch model
            inputs: 模型输入列表
            device: 'CPU', 'GPU', or 'NPU'

        Returns:
            output: 推理结果tensor
        """
        # 获取当前事件循环
        if self.loop is None:
            self.loop = asyncio.get_event_loop()

        # 将同步推理调用提交到线程池
        output = await self.loop.run_in_executor(
            self.executor,
            self._run_inference_sync,
            model,
            inputs,
            device
        )

        return output

    def _run_inference_sync(self, model, inputs: List, device: str) -> torch.Tensor:
        """
        同步推理（在线程池中执行）

        这个方法会在独立的线程中运行，ONNX Runtime的C++调用会释放GIL，
        允许其他线程并发执行
        """
        if hasattr(model, 'run'):
            # ONNX Runtime model
            input_names = [inp.name for inp in model.get_inputs()]

            # 转换为numpy
            numpy_inputs = self._convert_to_numpy(inputs)

            # 构建输入字典
            if len(input_names) == len(numpy_inputs):
                input_dict = {name: data for name, data in zip(input_names, numpy_inputs)}
            else:
                input_dict = {f'input_{i}': data for i, data in enumerate(numpy_inputs)}

            # ONNX Runtime推理 - 释放GIL
            outputs = model.run(None, input_dict)
            output = torch.from_numpy(outputs[0])

        else:
            # PyTorch model
            with torch.no_grad():
                # Move to device
                if device == 'GPU' and torch.cuda.is_available():
                    inputs = [t.cuda() if isinstance(t, torch.Tensor) else t for t in inputs]

                output = model(*inputs)

                # Move back to CPU
                if isinstance(output, torch.Tensor) and output.is_cuda:
                    output = output.cpu()

        return output

    def _convert_to_numpy(self, tensors: List) -> List[np.ndarray]:
        """将torch.Tensor转换为numpy数组"""
        numpy_arrays = []
        for t in tensors:
            if isinstance(t, torch.Tensor):
                numpy_arrays.append(
                    t.cpu().numpy().astype(
                        np.float32 if t.dtype == torch.float32 else np.int64
                    )
                )
            elif isinstance(t, int):
                numpy_arrays.append(np.array(t, dtype=np.int64))
            else:
                numpy_arrays.append(t)
        return numpy_arrays

    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)

    def __del__(self):
        """析构时关闭线程池"""
        try:
            self.executor.shutdown(wait=False)
        except:
            pass


class AsyncModelManager:
    """
    异步模型管理器

    管理多个模型的异步推理，支持批量提交和结果收集
    """

    def __init__(self, models: dict, max_workers: int = 4):
        """
        Args:
            models: {(block_id, device): model} 模型字典
            max_workers: 线程池大小
        """
        self.models = models
        self.wrapper = AsyncONNXWrapper(max_workers=max_workers)

    async def infer_batch_async(
        self,
        tasks: List[Tuple[int, str, List]]
    ) -> List[torch.Tensor]:
        """
        批量异步推理

        Args:
            tasks: [(block_id, device, inputs), ...] 推理任务列表

        Returns:
            outputs: 推理结果列表（顺序与tasks一致）
        """
        # 创建异步任务
        async_tasks = []
        for block_id, device, inputs in tasks:
            model = self.models[(block_id, device)]
            task = self.wrapper.infer_async(model, inputs, device)
            async_tasks.append(task)

        # 并发执行所有任务
        outputs = await asyncio.gather(*async_tasks)

        return outputs

    def shutdown(self):
        """关闭资源"""
        self.wrapper.shutdown()
