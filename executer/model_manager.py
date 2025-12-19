"""
Model Manager - Export, Load, and Cache Models
模型导出、加载和缓存管理
"""

import os
import sys
import torch
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# Import our standalone model export utilities
from model_export_utils import SimpleModelExporter, check_and_export_model


class ModelManager:
    """
    模型管理器

    功能:
    1. 检查模型文件是否存在且有效
    2. 调用old executor的导出器生成真实模型
    3. 加载和编译模型
    4. 缓存编译后的模型
    """

    def __init__(self, execution_plan: Dict, compilation_result_dir: Optional[Path] = None):
        """
        Args:
            execution_plan: Compiler输出的execution_plan部分
            compilation_result_dir: compilation_result.json所在目录（用于解析相对路径）
        """
        self.execution_plan = execution_plan
        self.clusters = execution_plan['clusters']
        self.compilation_result_dir = compilation_result_dir or Path.cwd()

        # 收集所有需要的模型
        self.model_refs = self._collect_model_refs()

        # 编译后的模型缓存
        self.compiled_models = {}

        print(f"  ✓ Model manager initialized")
        print(f"    Unique models needed: {len(self.model_refs)}")

    def _collect_model_refs(self) -> Dict[str, str]:
        """
        收集所有cluster的model_refs，并转换相对路径为绝对路径

        Returns:
            {model_key: absolute_model_path}
        """
        all_refs = {}

        for cluster in self.clusters:
            model_refs = cluster['model_refs']
            for key, rel_path in model_refs.items():
                if rel_path:
                    # Convert relative path to absolute path
                    # rel_path like "models/GPU_stages_1_2_3_4_5_6_7.onnx"
                    # Base is compilation_result_dir / compiler / output
                    if Path(rel_path).is_absolute():
                        # Already absolute (backward compatibility)
                        abs_path = rel_path
                    else:
                        # Relative path: resolve from compiler/output/
                        compiler_output_dir = self.compilation_result_dir
                        abs_path = str((compiler_output_dir / rel_path).resolve())
                    all_refs[key] = abs_path

        return all_refs

    def ensure_models_exist(self):
        """
        确保所有模型文件存在且有效

        如果文件不存在或是占位符（<200字节），则调用导出器生成
        """
        print(f"\n  Checking model files...")

        for model_key, model_path in self.model_refs.items():
            if not os.path.exists(model_path):
                print(f"    Model missing: {model_key}")
                self._export_model(model_key, model_path)
            elif os.path.getsize(model_path) < 200:
                print(f"    Model is placeholder: {model_key} ({os.path.getsize(model_path)} bytes)")
                self._export_model(model_key, model_path)
            else:
                print(f"    ✓ Model exists: {model_key} ({os.path.getsize(model_path) / 1024:.1f} KB)")

    def _export_model(self, model_key: str, model_path: str):
        """
        导出模型（使用standalone export utility）

        Args:
            model_key: 例如 "block_0_CPU"
            model_path: 输出路径
        """
        # 解析model_key: "block_0_CPU" → block_id=0, device="CPU"
        parts = model_key.split('_')
        block_id = int(parts[1])
        device = parts[2]

        # 从execution_plan找到对应的PEP
        cluster = self.clusters[0]  # 目前只有1个cluster
        pep = cluster['pep']

        if block_id >= len(pep):
            raise ValueError(f"Block {block_id} not found in PEP")

        block = pep[block_id]
        stages = block[1]  # [1, 2, 3, 4, 5, 6, 7]

        print(f"      Exporting {device} model for stages {stages}...")

        try:
            # 使用standalone exporter
            exporter = SimpleModelExporter()

            # 从compilation_result.json中获取数据集信息
            # 默认使用Flickr的参数
            num_nodes = 89250  # Flickr默认
            num_edges = 899756
            num_features = 500

            # 导出模型
            exporter.export_combined_model(
                device=device,
                stages=stages,
                output_path=model_path,
                num_nodes=num_nodes,
                num_edges=num_edges,
                num_features=num_features,
                dynamic=True  # CPU/GPU使用动态模型
            )

            print(f"      ✓ Exported: {os.path.getsize(model_path) / 1024:.1f} KB")

        except Exception as e:
            print(f"      ERROR: Failed to export model: {e}")
            raise

    def load_models(self):
        """
        加载和编译所有模型
        """
        print(f"\n  Loading and compiling models...")

        try:
            import onnxruntime as ort
        except ImportError:
            print("    WARNING: onnxruntime not available, models won't be loaded")
            return

        for model_key, model_path in self.model_refs.items():
            if model_key not in self.compiled_models:
                print(f"    Loading {model_key}...")

                # 使用ONNX Runtime加载模型
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                # 根据设备选择provider
                device = model_key.split('_')[2]
                if device == 'CPU':
                    providers = ['CPUExecutionProvider']
                elif device == 'GPU':
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                else:  # NPU
                    # NPU需要OpenVINO，这里先用CPU fallback
                    providers = ['CPUExecutionProvider']

                try:
                    session = ort.InferenceSession(
                        model_path,
                        sess_options=session_options,
                        providers=providers
                    )

                    self.compiled_models[model_key] = session
                    print(f"      ✓ Compiled for {providers[0]}")

                except Exception as e:
                    print(f"      ERROR: Failed to load model: {e}")
                    raise

        print(f"  ✓ All models loaded: {len(self.compiled_models)}")

    def get_model(self, block_id: int, device: str):
        """
        获取编译后的模型

        Args:
            block_id: Block ID
            device: 'CPU', 'GPU', or 'NPU'

        Returns:
            compiled_model: ONNX Runtime session
        """
        model_key = f"block_{block_id}_{device}"

        if model_key not in self.compiled_models:
            raise KeyError(f"Model not found: {model_key}")

        return self.compiled_models[model_key]

    def get_cluster_models(self, cluster_id: int = 0) -> Dict:
        """
        获取某个cluster的所有模型

        Args:
            cluster_id: Cluster ID (default 0)

        Returns:
            {(block_id, device): model}
        """
        cluster = self.clusters[cluster_id]
        pep = cluster['pep']

        models = {}

        for block_id, block in enumerate(pep):
            devices = block[0]  # ['CPU', 'GPU']

            for device in devices:
                model_key = f"block_{block_id}_{device}"
                if model_key in self.compiled_models:
                    models[(block_id, device)] = self.compiled_models[model_key]

        return models
