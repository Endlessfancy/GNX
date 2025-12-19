"""
Global Optimizer - Phase 3
Pipeline-aware全局优化
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from core.pep_generator import PEP
from core.cost_estimator import CostEstimator


class GlobalOptimizer:
    """
    Pipeline-aware全局优化器

    核心思想：
    1. 按PEP结构聚类subgraph
    2. 在cluster内排序subgraph减少bubble
    3. 估算真实pipeline makespan（包括bubble和传输）
    4. 找bubble最大的subgraph，尝试切换到次优PEP
    5. 迭代优化直到收敛
    """

    def __init__(self, cost_estimator: CostEstimator, config):
        """
        Args:
            cost_estimator: CostEstimator对象
            config: CompilerConfig对象
        """
        self.cost_estimator = cost_estimator
        self.config = config
        self.profiling = cost_estimator.profiling

    def optimize(self, subgraphs: List, top_k_peps: Dict) -> Tuple[Dict, Dict, float]:
        """
        全局优化主函数

        Args:
            subgraphs: Subgraph列表
            top_k_peps: {sg.id: [(pep, cost), ...]} 每个sg的Top-K候选PEP

        Returns:
            (assignment, clusters, best_makespan)
            - assignment: {sg.id: (pep, cost)}
            - clusters: {cluster_key: [sg_list]}
            - best_makespan: 最优makespan（ms）
        """
        # Step 1: 初始分配（每个sg选最快的PEP）
        assignment = {}
        for sg in subgraphs:
            assignment[sg.id] = top_k_peps[sg.id][0]  # (pep, cost)

        # Step 2: 迭代优化
        best_makespan = float('inf')
        no_improvement_count = 0

        for iteration in range(self.config.max_iterations):
            # Step 2.1: 按当前PEP进行聚类
            clusters = self._cluster_by_pep(subgraphs, assignment)

            # Step 2.2: 为每个cluster内的subgraph排序（减少bubble）
            for cluster_key in clusters:
                clusters[cluster_key] = self._sort_subgraphs_to_minimize_bubble(
                    clusters[cluster_key],
                    assignment
                )

            # Step 2.3: 估算真实pipeline makespan
            makespan, bottleneck_info = self._estimate_pipeline_makespan(
                clusters,
                assignment,
                subgraphs
            )

            if self.config.verbose:
                print(f"Iteration {iteration}: Makespan = {makespan:.2f}ms, "
                      f"Total bubble = {bottleneck_info.get('total_bubble_time', 0):.2f}ms")

            # Step 2.4: 检查收敛
            if makespan >= best_makespan - self.config.convergence_threshold:
                no_improvement_count += 1
                if no_improvement_count >= self.config.patience:
                    if self.config.verbose:
                        print(f"Converged after {iteration} iterations")
                    break
            else:
                best_makespan = makespan
                no_improvement_count = 0

            # Step 2.5: 尝试减少bottleneck
            swap_made = self._try_reduce_bottleneck(
                subgraphs,
                clusters,
                assignment,
                bottleneck_info,
                top_k_peps,
                makespan
            )

            if not swap_made:
                if self.config.verbose:
                    print("No beneficial swap found, stopping")
                break

        # 最后一次聚类和排序
        clusters = self._cluster_by_pep(subgraphs, assignment)
        for cluster_key in clusters:
            clusters[cluster_key] = self._sort_subgraphs_to_minimize_bubble(
                clusters[cluster_key],
                assignment
            )

        return assignment, clusters, best_makespan

    def _cluster_by_pep(self, subgraphs: List, assignment: Dict) -> Dict[str, List]:
        """
        按PEP结构聚类（不考虑具体shape）

        Args:
            subgraphs: Subgraph列表
            assignment: {sg.id: (pep, cost)}

        Returns:
            {cluster_key: [sg_list]}
        """
        clusters = defaultdict(list)

        for sg in subgraphs:
            pep, _ = assignment[sg.id]
            cluster_key = self._generate_pep_structure_key(pep)
            clusters[cluster_key].append(sg)

        return dict(clusters)

    def _generate_pep_structure_key(self, pep: PEP) -> str:
        """
        生成PEP结构签名（忽略具体shape）

        Args:
            pep: PEP对象

        Returns:
            结构key，例如 "CPU_GPU_s123_r0.50_0.50|NPU_s4567"
        """
        key_parts = []
        for block in pep.blocks:
            devices_str = '_'.join(sorted(block.devices))
            stages_str = 's' + ''.join(map(str, block.stages))
            ratios_str = 'r' + '_'.join([f"{r:.2f}" for r in block.ratios])
            key_parts.append(f"{devices_str}_{stages_str}_{ratios_str}")

        return '|'.join(key_parts)

    def _sort_subgraphs_to_minimize_bubble(self, sg_list: List, assignment: Dict) -> List:
        """
        在同一cluster内排序subgraph以减少bubble

        策略：
        - 如果Block 0慢，Block 1快 → 先执行大的sg
        - 如果Block 0快，Block 1慢 → 先执行小的sg
        - 比较平衡 → 交替大小

        Args:
            sg_list: Subgraph列表
            assignment: {sg.id: (pep, cost)}

        Returns:
            排序后的sg列表
        """
        if len(sg_list) <= 1:
            return sg_list

        # 计算每个sg的block时间比
        sg_ratios = []
        for sg in sg_list:
            pep, _ = assignment[sg.id]

            if len(pep.blocks) >= 2:
                # 估算Block 0和Block 1的时间
                block0_time = self.cost_estimator._estimate_block_time(pep.blocks[0], sg)
                block1_time = self.cost_estimator._estimate_block_time(pep.blocks[1], sg)

                ratio = block0_time / block1_time if block1_time > 0 else 1.0
                sg_ratios.append((sg, ratio, block0_time))
            else:
                # 单block，无需特殊排序
                sg_ratios.append((sg, 1.0, 0.0))

        # 计算平均ratio
        avg_ratio = sum(r for _, r, _ in sg_ratios) / len(sg_ratios)

        # 根据ratio决定排序策略
        if avg_ratio > 1.2:
            # Block 0明显慢 → 先执行大的（让Block 1不空闲）
            sorted_sg = sorted(sg_ratios, key=lambda x: x[2], reverse=True)
        elif avg_ratio < 0.8:
            # Block 1明显慢 → 先执行小的（避免Block 1积压）
            sorted_sg = sorted(sg_ratios, key=lambda x: x[2])
        else:
            # 比较平衡 → 按大小排序即可
            sorted_sg = sorted(sg_ratios, key=lambda x: x[2])

        return [sg for sg, _, _ in sorted_sg]

    def _estimate_pipeline_makespan(self, clusters: Dict, assignment: Dict, subgraphs: List) -> Tuple[float, Dict]:
        """
        估算真实的pipeline执行时间（包括bubble）

        Args:
            clusters: {cluster_key: [sg_list]}
            assignment: {sg.id: (pep, cost)}
            subgraphs: 所有subgraph列表（用于查找）

        Returns:
            (makespan, bottleneck_info)
        """
        cluster_timelines = []
        all_bubbles = []

        for cluster_key, sg_list in clusters.items():
            # 获取该cluster的PEP（所有sg共用）
            sample_sg = sg_list[0]
            pep, _ = assignment[sample_sg.id]
            num_blocks = len(pep.blocks)

            # 为每个block维护时间轴
            block_end_times = [0.0 for _ in range(num_blocks)]
            block_timelines = [[] for _ in range(num_blocks)]

            # 按排序后的顺序执行sg
            for sg in sg_list:
                pep, _ = assignment[sg.id]

                for block_idx, block in enumerate(pep.blocks):
                    # 计算block的计算时间
                    compute_time = self.cost_estimator._estimate_block_time(block, sg)

                    # 计算数据传输时间
                    transfer_time = 0
                    if block_idx > 0:
                        transfer_time = self.cost_estimator._estimate_transfer_time(
                            pep.blocks[block_idx - 1],
                            block,
                            sg
                        )

                    # 确定开始时间
                    if block_idx == 0:
                        # 第一个block：等前一个sg的block 0完成
                        start_time = block_end_times[block_idx]
                    else:
                        # 后续block：等两个条件
                        # 1. 当前sg的前一个block完成并传输完成
                        prev_block_ready = block_end_times[block_idx - 1] + transfer_time
                        # 2. 前一个sg的该block完成
                        prev_sg_done = block_end_times[block_idx]

                        start_time = max(prev_block_ready, prev_sg_done)

                    # 计算bubble
                    if block_idx > 0:
                        ideal_start = block_end_times[block_idx - 1] + transfer_time
                        bubble = start_time - ideal_start
                        if bubble > 0:
                            all_bubbles.append({
                                'cluster': cluster_key,
                                'sg_id': sg.id,
                                'block': block_idx,
                                'bubble_time': bubble
                            })
                    else:
                        bubble = 0

                    # 更新block结束时间
                    end_time = start_time + compute_time
                    block_end_times[block_idx] = end_time

                    block_timelines[block_idx].append({
                        'sg_id': sg.id,
                        'start': start_time,
                        'end': end_time,
                        'compute': compute_time,
                        'transfer': transfer_time,
                        'bubble': bubble
                    })

            # Cluster完成时间
            cluster_completion = max(block_end_times)
            cluster_timelines.append({
                'cluster_key': cluster_key,
                'completion_time': cluster_completion,
                'block_timelines': block_timelines
            })

        # 总makespan = 所有cluster的完成时间之和（cluster顺序执行）
        total_makespan = sum(ct['completion_time'] for ct in cluster_timelines)

        # Bottleneck信息
        if all_bubbles:
            all_bubbles.sort(key=lambda x: x['bubble_time'], reverse=True)

        bottleneck_info = {
            'worst_bubbles': all_bubbles[:10] if all_bubbles else [],
            'total_bubble_time': sum(b['bubble_time'] for b in all_bubbles),
            'cluster_timelines': cluster_timelines
        }

        return total_makespan, bottleneck_info

    def _try_reduce_bottleneck(self, subgraphs: List, clusters: Dict, assignment: Dict,
                               bottleneck_info: Dict, top_k_peps: Dict, current_makespan: float) -> bool:
        """
        尝试通过切换PEP来减少bottleneck

        Args:
            subgraphs: Subgraph列表
            clusters: 当前cluster
            assignment: 当前分配
            bottleneck_info: Bottleneck信息
            top_k_peps: 所有候选PEP
            current_makespan: 当前makespan

        Returns:
            是否成功swap
        """
        worst_bubbles = bottleneck_info['worst_bubbles']

        if not worst_bubbles:
            return False

        # 尝试调整bubble最大的几个subgraph
        for bubble_info in worst_bubbles[:5]:
            sg_id = bubble_info['sg_id']

            # 找到对应的subgraph
            sg = next((s for s in subgraphs if s.id == sg_id), None)
            if sg is None:
                continue

            current_pep, current_cost = assignment[sg_id]

            # 尝试备选PEP
            for alt_pep, alt_cost in top_k_peps[sg_id][1:]:
                # 临时切换
                old_assignment = assignment[sg_id]
                assignment[sg_id] = (alt_pep, alt_cost)

                # 重新聚类和排序
                new_clusters = self._cluster_by_pep(subgraphs, assignment)
                for cluster_key in new_clusters:
                    new_clusters[cluster_key] = self._sort_subgraphs_to_minimize_bubble(
                        new_clusters[cluster_key],
                        assignment
                    )

                # 重新估算makespan
                new_makespan, _ = self._estimate_pipeline_makespan(
                    new_clusters,
                    assignment,
                    subgraphs
                )

                # 如果有改善，保持切换
                if new_makespan < current_makespan - self.config.convergence_threshold:
                    if self.config.verbose:
                        print(f"  ✓ Swapped sg_{sg_id} to alt PEP, "
                              f"makespan: {current_makespan:.2f} → {new_makespan:.2f}")
                    return True
                else:
                    # 没有改善，恢复
                    assignment[sg_id] = old_assignment

        return False
