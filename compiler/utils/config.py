"""
Compiler Configuration Management
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class CompilerConfig:
    """GNN Compiler配置"""

    # Graph partition配置
    k_set: List[int] = field(default_factory=lambda: list(range(8, 16)))  # 候选切分数
    use_metis: bool = True  # 是否使用METIS进行图划分
    metis_ufactor: int = 30  # METIS平衡约束 (1-50, 30=3%不平衡度)
    dataset_name: str = 'flickr'  # 数据集名称

    # PEP generation配置
    max_pipeline_blocks: int = 2  # 最大pipeline段数（1-3）
    top_k_peps: int = 5  # 每个subgraph保留的候选PEP数量

    # Global optimization配置
    max_iterations: int = 20  # 最大迭代次数
    convergence_threshold: float = 0.1  # 收敛阈值（ms）
    patience: int = 3  # 连续几次无改进则停止

    # Device配置
    devices: List[str] = field(default_factory=lambda: ['CPU', 'GPU', 'NPU'])
    device_memory: Dict[str, float] = field(default_factory=lambda: {
        'CPU': 32 * 1024 * 1024 * 1024,  # 32GB
        'GPU': 16 * 1024 * 1024 * 1024,  # 16GB
        'NPU': 8 * 1024 * 1024 * 1024,   # 8GB
    })

    # NPU配置
    npu_padding_multiple: int = 1000  # NPU padding到1000的倍数
    npu_unsupported_stages: List[int] = field(default_factory=lambda: [3, 4])  # NPU不支持的stage

    # Profiling data path (relative to project root)
    profiling_dir: Path = field(default_factory=lambda:
        Path(__file__).parent.parent.parent / 'profiling' / 'results')

    # Model configuration
    num_stages: int = 7  # Total GraphSAGE stages
    feature_dim: int = 500  # Feature dimension (must match profiling)

    # Output configuration (relative to compiler directory)
    output_dir: Path = field(default_factory=lambda:
        Path(__file__).parent.parent / 'output')

    # Debug配置
    verbose: bool = True
    visualize_timeline: bool = False  # 是否生成timeline可视化

    def __post_init__(self):
        """确保路径存在"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CompilerConfig':
        """从字典创建配置"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'k_set': self.k_set,
            'use_metis': self.use_metis,
            'metis_ufactor': self.metis_ufactor,
            'dataset_name': self.dataset_name,
            'max_pipeline_blocks': self.max_pipeline_blocks,
            'top_k_peps': self.top_k_peps,
            'max_iterations': self.max_iterations,
            'convergence_threshold': self.convergence_threshold,
            'patience': self.patience,
            'devices': self.devices,
            'device_memory': self.device_memory,
            'npu_padding_multiple': self.npu_padding_multiple,
            'npu_unsupported_stages': self.npu_unsupported_stages,
            'num_stages': self.num_stages,
            'feature_dim': self.feature_dim,
            'verbose': self.verbose,
            'visualize_timeline': self.visualize_timeline
        }
