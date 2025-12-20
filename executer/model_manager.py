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
            compilation_result_dir: compilation_result.json所在目录（用于解析相对路径，已废弃）
        """
        self.execution_plan = execution_plan
        self.clusters = execution_plan['clusters']

        # Use executor's own models directory (independent from compiler)
        self.models_dir = Path(__file__).parent / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 收集所有需要的模型
        self.model_refs = self._collect_model_refs()

        # 编译后的模型缓存
        self.compiled_models = {}

        print(f"  ✓ Model manager initialized")
        print(f"    Unique models needed: {len(self.model_refs)}")
        print(f"    Models directory: {self.models_dir}")

    def _collect_model_refs(self) -> Dict[str, str]:
        """
        收集所有cluster的model_refs，映射到executor自己的models目录

        Returns:
            {model_key: absolute_model_path_in_executor_models_dir}
        """
        all_refs = {}

        for cluster in self.clusters:
            # 如果 model_refs 为空（自定义 PEP），从 PEP 自动生成
            if not cluster.get('model_refs') or len(cluster['model_refs']) == 0:
                pep = cluster['pep']
                for block_id, block in enumerate(pep):
                    devices = block[0]  # ['CPU', 'GPU'] or ['NPU']
                    stages = block[1]   # [1, 2, 3, 4, 5] or [6, 7]

                    # 为每个设备生成模型引用
                    for device in devices:
                        stages_str = '_'.join(map(str, stages))
                        model_key = f"block_{block_id}_{device}"
                        model_filename = f"{device}_stages_{stages_str}.onnx"
                        abs_path = str((self.models_dir / model_filename).resolve())
                        all_refs[model_key] = abs_path
            else:
                # 使用已有的 model_refs
                model_refs = cluster['model_refs']
                for key, ref_path in model_refs.items():
                    if ref_path:
                        # Extract model filename from compilation_result reference
                        model_filename = Path(ref_path).name
                        abs_path = str((self.models_dir / model_filename).resolve())
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

        # 从model_refs找到对应的cluster和PEP
        # 需要查找哪个cluster包含这个model_key
        cluster = None
        pep = None
        for c in self.clusters:
            # Check if this cluster contains the model_key by regenerating keys
            temp_pep = c['pep']
            for bid, block in enumerate(temp_pep):
                devices = block[0]
                if bid == block_id and device in devices:
                    cluster = c
                    pep = temp_pep
                    break
            if cluster is not None:
                break

        if cluster is None or pep is None:
            raise ValueError(f"Model {model_key} not found in any cluster")

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
            use_onnx = True
        except ImportError:
            print("    WARNING: onnxruntime not available, using PyTorch models as fallback")
            use_onnx = False

        for model_key, model_path in self.model_refs.items():
            if model_key not in self.compiled_models:
                print(f"    Loading {model_key}...")

                # 根据设备选择provider
                device = model_key.split('_')[2]

                if use_onnx:
                    # 使用ONNX Runtime加载模型
                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

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
                        print(f"      ERROR: Failed to load ONNX model: {e}")
                        raise
                else:
                    # Fallback: Create PyTorch model
                    try:
                        from model_export_utils import create_combined_sage_model

                        # Extract stages from model_key (e.g., "block_0_GPU" -> stages from PEP)
                        block_id = int(model_key.split('_')[1])
                        cluster = self.clusters[0]
                        pep = cluster['pep']
                        stages = pep[block_id][1]  # [1, 2, 3, 4, 5, 6, 7]

                        # Create PyTorch model
                        pytorch_model = create_combined_sage_model(
                            stages=stages,
                            in_channels=500,
                            hidden_channels=256,
                            out_channels=256
                        )

                        # Move to appropriate device
                        if device == 'GPU' and torch.cuda.is_available():
                            pytorch_model = pytorch_model.cuda()

                        pytorch_model.eval()

                        self.compiled_models[model_key] = pytorch_model
                        print(f"      ✓ Loaded PyTorch model on {device}")

                    except Exception as e:
                        print(f"      ERROR: Failed to create PyTorch fallback: {e}")
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
