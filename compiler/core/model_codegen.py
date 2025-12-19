"""
Model Codegen - Phase 4
根据PEP导出和缓存模型
"""

from typing import Dict, List, Set
from pathlib import Path
from core.pep_generator import PEP


class ModelCodegen:
    """
    模型代码生成和缓存管理器

    负责：
    1. 根据PEP导出ONNX/IR模型
    2. 管理模型缓存（避免重复导出）
    3. 生成模型索引表
    """

    def __init__(self, config):
        """
        Args:
            config: CompilerConfig对象
        """
        self.config = config
        self.models_dir = config.output_dir / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 模型缓存
        self.model_cache = {}  # {model_key: file_path}

    def generate_models(self, assignment: Dict, clusters: Dict, subgraphs: List) -> Dict:
        """
        根据PEP分配生成所有需要的模型

        Args:
            assignment: {sg.id: (pep, cost)}
            clusters: {cluster_key: [sg_list]}
            subgraphs: Subgraph列表

        Returns:
            模型索引表 {model_key: file_path}
        """
        required_models = self._collect_required_models(assignment, subgraphs)

        if self.config.verbose:
            print(f"\nGenerating {len(required_models)} unique models...")

        for model_key in required_models:
            if model_key not in self.model_cache:
                file_path = self._export_model(model_key)
                self.model_cache[model_key] = file_path

                if self.config.verbose:
                    print(f"  ✓ Exported: {model_key}")

        return self.model_cache.copy()

    def _collect_required_models(self, assignment: Dict, subgraphs: List) -> Set[str]:
        """
        收集所有需要的模型

        Args:
            assignment: PEP分配
            subgraphs: Subgraph列表

        Returns:
            模型key集合
        """
        required = set()

        for sg in subgraphs:
            pep, _ = assignment[sg.id]

            for block in pep.blocks:
                for device in block.devices:
                    if device in ['CPU', 'GPU']:
                        # Dynamic model - 不依赖shape
                        model_key = self._generate_dynamic_model_key(device, block.stages)
                        required.add(model_key)
                    elif device == 'NPU':
                        # Static model - 依赖shape
                        model_key = self._generate_static_model_key(
                            device,
                            block.stages,
                            sg.n_pad,
                            sg.m_pad
                        )
                        required.add(model_key)

        return required

    def _generate_dynamic_model_key(self, device: str, stages: List[int]) -> str:
        """生成dynamic model的key"""
        stages_str = '_'.join(map(str, stages))
        return f"{device}_stages_{stages_str}"

    def _generate_static_model_key(self, device: str, stages: List[int], n_pad: int, m_pad: int) -> str:
        """生成static model的key（NPU）"""
        stages_str = '_'.join(map(str, stages))
        return f"{device}_stages_{stages_str}_n{n_pad}_m{m_pad}"

    def _export_model(self, model_key: str) -> Path:
        """
        导出模型

        Args:
            model_key: 模型key

        Returns:
            模型文件路径
        """
        # 解析model_key
        parts = model_key.split('_')
        device = parts[0]

        if device in ['CPU', 'GPU']:
            # Dynamic model - 导出ONNX
            file_path = self.models_dir / f"{model_key}.onnx"
            self._export_dynamic_onnx(model_key, file_path)
        elif device == 'NPU':
            # Static model - 导出IR
            file_path = self.models_dir / f"{model_key}.xml"
            self._export_static_ir(model_key, file_path)
        else:
            raise ValueError(f"Unknown device: {device}")

        return file_path

    def _export_dynamic_onnx(self, model_key: str, file_path: Path):
        """
        导出dynamic ONNX模型

        这里是简化版，实际需要调用executor的model_exporter

        Args:
            model_key: 模型key
            file_path: 输出路径
        """
        # 简化：创建一个占位文件
        # 实际应该调用:
        # from executor.pep_model_exporter import PEPModelExporter
        # exporter = PEPModelExporter()
        # exporter.export_combined_model(device, stages, file_path, dynamic=True)

        with open(file_path, 'w') as f:
            f.write(f"# Placeholder for dynamic ONNX model: {model_key}\n")
            f.write(f"# This should be exported using executor/pep_model_exporter.py\n")

    def _export_static_ir(self, model_key: str, file_path: Path):
        """
        导出static IR模型（NPU）

        Args:
            model_key: 模型key
            file_path: 输出路径
        """
        # 简化：创建占位文件
        # 实际应该:
        # 1. 先导出ONNX（with fixed shape）
        # 2. 使用OpenVINO Model Optimizer转换为IR

        with open(file_path, 'w') as f:
            f.write(f"# Placeholder for static IR model: {model_key}\n")
            f.write(f"# This should be exported and compiled for NPU\n")

    def get_model_path(self, device: str, stages: List[int], n_pad: int = None, m_pad: int = None) -> Path:
        """
        获取模型路径

        Args:
            device: 设备名
            stages: stage列表
            n_pad: NPU padding节点数（NPU需要）
            m_pad: NPU padding边数（NPU需要）

        Returns:
            模型文件路径
        """
        if device in ['CPU', 'GPU']:
            model_key = self._generate_dynamic_model_key(device, stages)
        elif device == 'NPU':
            if n_pad is None or m_pad is None:
                raise ValueError("NPU model requires n_pad and m_pad")
            model_key = self._generate_static_model_key(device, stages, n_pad, m_pad)
        else:
            raise ValueError(f"Unknown device: {device}")

        return self.model_cache.get(model_key)
