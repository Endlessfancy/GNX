"""
Model Manager - Export, Load, and Cache Models
模型导出、加载和缓存管理

使用 OpenVINO 进行推理，支持异步执行
"""

import os
import sys
import torch
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# Import our standalone model export utilities
from model_export_utils import SimpleModelExporter, check_and_export_model

# OpenVINO imports
try:
    from openvino.runtime import Core
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("WARNING: OpenVINO not available, will use ONNX Runtime fallback")


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

        # OpenVINO Core (延迟初始化)
        self.ov_core = None

        # IR 模型目录
        self.ir_models_dir = self.models_dir / 'ir'
        self.ir_models_dir.mkdir(parents=True, exist_ok=True)

        print(f"  ✓ Model manager initialized")
        print(f"    Unique models needed: {len(self.model_refs)}")
        print(f"    Models directory: {self.models_dir}")
        print(f"    OpenVINO available: {OPENVINO_AVAILABLE}")

    def _collect_model_refs(self) -> Dict[str, str]:
        """
        收集所有cluster的model_refs，映射到executor自己的models目录

        Returns:
            {model_key: absolute_model_path_in_executor_models_dir}
        """
        all_refs = {}

        for cluster_id, cluster in enumerate(self.clusters):
            # 如果 model_refs 为空（自定义 PEP），从 PEP 自动生成
            if not cluster.get('model_refs') or len(cluster['model_refs']) == 0:
                pep = cluster['pep']
                for block_id, block in enumerate(pep):
                    devices = block[0]  # ['CPU', 'GPU'] or ['NPU']
                    stages = block[1]   # [1, 2, 3, 4, 5] or [6, 7]

                    # 为每个设备生成模型引用
                    # Use cluster-specific model keys to avoid conflicts
                    for device in devices:
                        stages_str = '_'.join(map(str, stages))
                        model_key = f"cluster_{cluster_id}_block_{block_id}_{device}"
                        model_filename = f"c{cluster_id}_b{block_id}_{device}_stages_{stages_str}.onnx"
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
            model_key: 例如 "cluster_0_block_0_CPU"
            model_path: 输出路径
        """
        # 解析model_key: "cluster_0_block_0_CPU" → cluster_id=0, block_id=0, device="CPU"
        parts = model_key.split('_')
        cluster_id = int(parts[1])
        block_id = int(parts[3])
        device = parts[4]

        # 从对应的cluster获取PEP
        if cluster_id >= len(self.clusters):
            raise ValueError(f"Cluster {cluster_id} not found")

        cluster = self.clusters[cluster_id]
        pep = cluster['pep']

        if block_id >= len(pep):
            raise ValueError(f"Block {block_id} not found in cluster {cluster_id}")

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

    def _convert_onnx_to_ir(self, onnx_path: str, model_key: str) -> str:
        """
        将 ONNX 模型转换为 OpenVINO IR 格式

        Args:
            onnx_path: ONNX 模型路径
            model_key: 模型标识符

        Returns:
            ir_xml_path: IR 模型的 XML 文件路径
        """
        ir_xml_path = self.ir_models_dir / f"{model_key}.xml"
        ir_bin_path = self.ir_models_dir / f"{model_key}.bin"

        # 如果 IR 模型已存在且比 ONNX 更新，则跳过转换
        if ir_xml_path.exists() and ir_bin_path.exists():
            onnx_mtime = os.path.getmtime(onnx_path)
            ir_mtime = os.path.getmtime(ir_xml_path)
            if ir_mtime >= onnx_mtime:
                print(f"      ✓ IR model exists: {model_key}")
                return str(ir_xml_path)

        print(f"      Converting ONNX to IR: {model_key}...")

        try:
            # 使用 OpenVINO Model Optimizer 转换
            ov_model = ov.convert_model(onnx_path)

            # 保存 IR 模型
            ov.save_model(ov_model, str(ir_xml_path))

            print(f"      ✓ Converted to IR: {ir_xml_path.name}")
            return str(ir_xml_path)

        except Exception as e:
            print(f"      ERROR: Failed to convert ONNX to IR: {e}")
            raise

    def load_models(self):
        """
        加载和编译所有模型（使用 OpenVINO）
        """
        print(f"\n  Loading and compiling models...")

        if OPENVINO_AVAILABLE:
            # 初始化 OpenVINO Core
            self.ov_core = Core()
            print(f"    Using OpenVINO backend")
            print(f"    Available devices: {self.ov_core.available_devices}")

            for model_key, onnx_path in self.model_refs.items():
                if model_key not in self.compiled_models:
                    print(f"    Loading {model_key}...")

                    # 解析设备
                    parts = model_key.split('_')
                    device = parts[-1]  # 最后一个部分是设备名

                    try:
                        # 转换 ONNX 到 IR
                        ir_path = self._convert_onnx_to_ir(onnx_path, model_key)

                        # 读取模型
                        model = self.ov_core.read_model(ir_path)

                        # 选择 OpenVINO 设备
                        if device == 'NPU':
                            ov_device = 'NPU'
                        elif device == 'GPU':
                            # 检查 GPU 是否可用
                            if 'GPU' in self.ov_core.available_devices:
                                ov_device = 'GPU'
                            else:
                                ov_device = 'CPU'
                                print(f"      WARNING: GPU not available, using CPU")
                        else:
                            ov_device = 'CPU'

                        # 编译模型
                        compiled_model = self.ov_core.compile_model(model, ov_device)

                        self.compiled_models[model_key] = compiled_model
                        print(f"      ✓ Compiled for {ov_device}")

                    except Exception as e:
                        print(f"      ERROR: Failed to load model: {e}")
                        raise

        else:
            # Fallback 到 ONNX Runtime
            print(f"    Using ONNX Runtime backend (OpenVINO not available)")
            self._load_models_onnxruntime()

        print(f"  ✓ All models loaded: {len(self.compiled_models)}")

    def _load_models_onnxruntime(self):
        """
        使用 ONNX Runtime 加载模型（fallback）
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Neither OpenVINO nor ONNX Runtime is available")

        for model_key, model_path in self.model_refs.items():
            if model_key not in self.compiled_models:
                print(f"    Loading {model_key}...")

                parts = model_key.split('_')
                device = parts[-1]

                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                if device == 'GPU':
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                else:
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
                model_key = f"cluster_{cluster_id}_block_{block_id}_{device}"
                if model_key in self.compiled_models:
                    models[(block_id, device)] = self.compiled_models[model_key]

        return models
