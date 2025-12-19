"""
PEP Generator - Phase 2
生成候选的PEP（Parallel Execution Plan）
"""

from typing import List, Dict, Tuple, Optional
from itertools import combinations, product
from dataclasses import dataclass, field


@dataclass
class PEPBlock:
    """PEP中的一个block"""
    devices: List[str]  # 设备列表，如['CPU', 'GPU']
    stages: List[int]  # stage列表，如[1, 2, 3]
    ratios: List[float]  # split ratio，如[0.5, 0.5]

    def __repr__(self):
        devices_str = '_'.join(self.devices)
        stages_str = ''.join(map(str, self.stages))
        ratios_str = '_'.join([f"{r:.2f}" for r in self.ratios])
        return f"{devices_str}_s{stages_str}_r{ratios_str}"


@dataclass
class PEP:
    """完整的PEP"""
    blocks: List[PEPBlock]

    def __repr__(self):
        return ' | '.join([str(block) for block in self.blocks])

    def to_executor_format(self) -> List[List]:
        """转换为executor需要的格式"""
        result = []
        for block in self.blocks:
            result.append([block.devices, block.stages, block.ratios])
        return result


class PEPGenerator:
    """
    PEP生成器

    枚举所有合法的PEP候选
    """

    def __init__(self, config):
        """
        Args:
            config: CompilerConfig对象
        """
        self.config = config
        self.num_stages = config.num_stages
        self.devices = config.devices
        self.max_blocks = config.max_pipeline_blocks
        self.npu_unsupported_stages = config.npu_unsupported_stages

    def generate_candidates(self, subgraph) -> List[PEP]:
        """
        为一个subgraph生成所有候选PEP

        Args:
            subgraph: Subgraph对象

        Returns:
            候选PEP列表
        """
        all_peps = []

        # 枚举pipeline block数量（1, 2, 3）
        for num_blocks in range(1, self.max_blocks + 1):
            # 枚举stage的划分方式
            for stage_partition in self._enumerate_stage_partitions(num_blocks):
                # 为每个block分配设备
                for device_allocation in self._enumerate_device_allocations(num_blocks, stage_partition):
                    # 为每个block分配split ratio
                    for split_ratios in self._enumerate_split_ratios(device_allocation):
                        # 创建PEP
                        blocks = []
                        for i in range(num_blocks):
                            block = PEPBlock(
                                devices=device_allocation[i],
                                stages=stage_partition[i],
                                ratios=split_ratios[i]
                            )
                            blocks.append(block)

                        pep = PEP(blocks=blocks)

                        # 验证PEP合法性
                        if self._validate_pep(pep, subgraph):
                            all_peps.append(pep)

        return all_peps

    def _enumerate_stage_partitions(self, num_blocks: int) -> List[List[List[int]]]:
        """
        枚举stage的连续划分方式

        Args:
            num_blocks: block数量

        Returns:
            划分方式列表，例如[[1,2,3], [4,5,6,7]]
        """
        if num_blocks == 1:
            return [[list(range(1, self.num_stages + 1))]]

        partitions = []
        all_stages = list(range(1, self.num_stages + 1))

        # 枚举num_blocks-1个切分点
        for split_points in combinations(range(1, self.num_stages), num_blocks - 1):
            partition = []
            prev = 0
            for sp in split_points:
                partition.append(all_stages[prev:sp])
                prev = sp
            partition.append(all_stages[prev:])

            # 确保每个block至少有1个stage
            if all(len(p) > 0 for p in partition):
                partitions.append(partition)

        return partitions

    def _enumerate_device_allocations(self, num_blocks: int, stage_partition: List[List[int]]) -> List[List[List[str]]]:
        """
        枚举设备分配方式

        Args:
            num_blocks: block数量
            stage_partition: stage划分

        Returns:
            设备分配列表，例如[['CPU', 'GPU'], ['NPU']]
        """
        allocations = []

        # 为每个block枚举可能的设备组合
        block_device_options = []
        for block_stages in stage_partition:
            # 检查该block能使用哪些设备
            valid_devices = []
            for device in self.devices:
                if device == 'NPU':
                    # NPU不能包含unsupported stages
                    if not any(s in self.npu_unsupported_stages for s in block_stages):
                        valid_devices.append(device)
                else:
                    valid_devices.append(device)

            # 该block的设备选项：单设备或多设备DP
            options = []
            # 单设备
            for dev in valid_devices:
                options.append([dev])
            # 双设备DP（不包含NPU，NPU目前不支持DP）
            if 'CPU' in valid_devices and 'GPU' in valid_devices:
                options.append(['CPU', 'GPU'])

            block_device_options.append(options)

        # 枚举所有组合
        for allocation in product(*block_device_options):
            # 验证设备独占性（同一设备不能出现在多个block）
            all_devices = []
            for block_devs in allocation:
                all_devices.extend(block_devs)

            if len(all_devices) == len(set(all_devices)):  # 无重复
                allocations.append(list(allocation))

        return allocations

    def _enumerate_split_ratios(self, device_allocation: List[List[str]]) -> List[List[List[float]]]:
        """
        枚举split ratio

        Args:
            device_allocation: 设备分配

        Returns:
            ratio列表
        """
        ratios_list = []

        for block_devices in device_allocation:
            if len(block_devices) == 1:
                # 单设备，ratio=1.0
                ratios_list.append([[1.0]])
            elif len(block_devices) == 2:
                # 双设备，枚举几种常见的分配比例
                # 简化：只考虑[0.3, 0.7], [0.5, 0.5], [0.7, 0.3]
                ratios_list.append([
                    [0.3, 0.7],
                    [0.5, 0.5],
                    [0.7, 0.3]
                ])
            else:
                # 多设备（暂不支持）
                ratios_list.append([[1.0 / len(block_devices)] * len(block_devices)])

        # 枚举所有组合
        all_ratio_combos = []
        for combo in product(*ratios_list):
            all_ratio_combos.append(list(combo))

        return all_ratio_combos

    def _validate_pep(self, pep: PEP, subgraph) -> bool:
        """
        验证PEP是否合法

        Args:
            pep: PEP对象
            subgraph: Subgraph对象

        Returns:
            是否合法
        """
        # 1. 检查stage覆盖性（1-7都要有）
        all_stages = []
        for block in pep.blocks:
            all_stages.extend(block.stages)

        expected_stages = set(range(1, self.num_stages + 1))
        if set(all_stages) != expected_stages:
            return False

        # 2. 检查stage连续性
        prev_max = 0
        for block in pep.blocks:
            if block.stages[0] != prev_max + 1:
                return False
            prev_max = block.stages[-1]

        # 3. 检查NPU约束
        for block in pep.blocks:
            if 'NPU' in block.devices:
                if any(s in self.npu_unsupported_stages for s in block.stages):
                    return False

        # 4. 检查ratio合法性
        for block in pep.blocks:
            if abs(sum(block.ratios) - 1.0) > 0.01:  # 允许小误差
                return False

        # 5. 检查设备独占性
        all_devices = []
        for block in pep.blocks:
            all_devices.extend(block.devices)
        if len(all_devices) != len(set(all_devices)):
            return False

        return True
