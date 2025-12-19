"""
Graph Data Loader
加载和缓存图数据集用于编译器
"""

from pathlib import Path
from typing import Optional
import torch


class GraphLoader:
    """
    图数据加载器

    负责：
    1. 加载各种图数据集（Flickr, Reddit, Yelp等）
    2. 缓存已加载的数据避免重复加载
    3. 提供统一的数据接口
    """

    def __init__(self, cache_dir: Path = Path('data')):
        """
        Args:
            cache_dir: 数据集缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cached_data = {}

    def load_flickr(self):
        """
        加载Flickr数据集

        Returns:
            PyG Data对象，包含：
            - edge_index: [2, num_edges]
            - x: [num_nodes, num_features] (optional)
            - y: [num_nodes] (optional)
            - num_nodes, num_edges
        """
        if 'flickr' not in self._cached_data:
            try:
                import torch
                from torch_geometric.datasets import Flickr
                from torch_geometric.data.data import Data

                # 允许加载PyG Data对象（解决PyTorch 2.6+的weights_only限制）
                torch.serialization.add_safe_globals([Data])

                dataset = Flickr(root=str(self.cache_dir / 'Flickr'))
                data = dataset[0]

                print(f"  Loaded Flickr dataset:")
                print(f"  Nodes: {data.num_nodes:,}")
                print(f"  Edges: {data.num_edges:,}")
                print(f"  Features: {data.x.shape[1] if hasattr(data, 'x') else 'N/A'}")

                self._cached_data['flickr'] = data
            except ImportError:
                raise ImportError("torch_geometric not installed. Please install: pip install torch_geometric")

        return self._cached_data['flickr']

    def load_reddit(self):
        """
        加载Reddit数据集

        Returns:
            PyG Data对象
        """
        if 'reddit' not in self._cached_data:
            from torch_geometric.datasets import Reddit

            dataset = Reddit(root=str(self.cache_dir / 'Reddit'))
            data = dataset[0]

            print(f"✓ Loaded Reddit dataset:")
            print(f"  Nodes: {data.num_nodes:,}")
            print(f"  Edges: {data.num_edges:,}")

            self._cached_data['reddit'] = data

        return self._cached_data['reddit']

    def load_yelp(self):
        """
        加载Yelp数据集

        Returns:
            PyG Data对象
        """
        if 'yelp' not in self._cached_data:
            from torch_geometric.datasets import Yelp

            dataset = Yelp(root=str(self.cache_dir / 'Yelp'))
            data = dataset[0]

            print(f"✓ Loaded Yelp dataset:")
            print(f"  Nodes: {data.num_nodes:,}")
            print(f"  Edges: {data.num_edges:,}")

            self._cached_data['yelp'] = data

        return self._cached_data['yelp']

    def load_synthetic(self, num_nodes: int, num_edges: int, seed: int = 42):
        """
        生成合成图（用于测试）

        Args:
            num_nodes: 节点数
            num_edges: 边数
            seed: 随机种子

        Returns:
            PyG Data对象
        """
        cache_key = f'synthetic_{num_nodes}_{num_edges}_{seed}'

        if cache_key not in self._cached_data:
            torch.manual_seed(seed)

            # 生成随机边
            edge_index = torch.randint(0, num_nodes, (2, num_edges))

            # 创建Data对象
            from torch_geometric.data import Data
            data = Data(
                edge_index=edge_index,
                num_nodes=num_nodes
            )

            print(f"✓ Generated synthetic graph:")
            print(f"  Nodes: {num_nodes:,}")
            print(f"  Edges: {num_edges:,}")

            self._cached_data[cache_key] = data

        return self._cached_data[cache_key]

    def load_dataset(self, name: str, **kwargs):
        """
        通用数据集加载接口

        Args:
            name: 数据集名称 ('flickr', 'reddit', 'yelp', 'synthetic')
            **kwargs: 数据集特定参数（如synthetic的num_nodes, num_edges）

        Returns:
            PyG Data对象
        """
        name_lower = name.lower()

        if name_lower == 'flickr':
            return self.load_flickr()
        elif name_lower == 'reddit':
            return self.load_reddit()
        elif name_lower == 'yelp':
            return self.load_yelp()
        elif name_lower == 'synthetic':
            return self.load_synthetic(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {name}. "
                           f"Supported: flickr, reddit, yelp, synthetic")

    def clear_cache(self):
        """清除缓存的数据"""
        self._cached_data.clear()
        print("✓ Cache cleared")
