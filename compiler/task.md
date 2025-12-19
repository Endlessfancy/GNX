中文：更新后的 Compiler 设计说明（简版）
目标

在 CPU/GPU/NPU 异构平台上执行 GNN inference。给定模型（7 个顺序 stage）和大图数据集，自动生成一套执行策略，使处理完整数据集的总时间（global total time / makespan）最小。策略支持：

PP：把 stage 连续切成 1/2/3 个 block pipeline 执行

DP：每个 block 可在多个 PU 上并行执行，并带 split ratio

核心约束与工程假设

Graph partition 很贵：只允许少量候选切分数 Kset={10,11,12,...}，每个 k 只跑一次 METIS

NPU 需要 static shape：通过 padding 到 1000 的整数倍解决，不需要强制 subgraph 实际 size 相同

通过把相同 PEP 的 subgraph 排到一起执行，减少 plan switching

编译期用 offline profiling 的 lookup tables 估算时间与内存（不在线跑真推理）

Offline Profiling（一次）

生成 lookup tables：

T_comp[device][stage_or_block][n,m]：不同设备、不同输入规模的计算时间（插值）

Mem_peak[device][stage_or_block][n,m]：峰值内存（可行性剪枝）

T_comm[src][dst](bytes) 或 BW/Lat：设备间搬运时间模型

Compile 阶段（对每个 k in Kset）
Step 1：METIS 切分

输入：原图 G、k

输出：k 个 subgraph（每个 subgraph 有 n_i、m_i、可选 cut 边统计）

Step 2：为每个 subgraph 定义 NPU static shape 模板（padding）

对每个 subgraph：

n_pad = ceil(n_i/1000)*1000

m_pad = ceil(m_i/1000)*1000

若某 PEP 使用 NPU，则该 subgraph 在 NPU 上的估时、可行性都基于 (n_pad,m_pad) 对应的 lookup/插值（并计入 padding/pack/unpack 的固定开销）

Step 3：生成候选 PEP，并估算代价

对每个 subgraph（或 subgraph 类别）：

枚举合法 PEP：

pipeline 段数 1/2/3

block 必须是连续 stage fuse

段内可 DP（多个设备 + ratio）

设备不能跨段重复使用

NPU 不允许包含不支持的 stage（如 stage3/4）

内存必须可行（Mem_peak）

用 lookup table 估算该 PEP 的资源消耗（计算+通信+合并）与总延迟

保留 Top-K 候选（可选，便于后续全局优化）

Step 4：全局优化（最小 makespan / 总时间）

目标不是每个 subgraph 自己最快，而是让 CPU/GPU/NPU 总负载最均衡，从而总时间最小：

对每个 subgraph 选择一个候选 PEP（或对每类 subgraph 选择）

目标：min max(Load_CPU, Load_GPU, Load_NPU)

可用简单贪心：先选局部最快 → 找瓶颈设备 → 将部分 subgraph 切换到“更少用瓶颈设备”的备选 PEP，迭代到收敛

得到该 k 的预计总时间 T_k。

Step 5：选择最优 k

取 k* = argmin_k T_k

输出 k* 对应的 partition + 全局 PEP 分配结果

Step 6：Codegen + Cache

CPU/GPU：导出 dynamic ONNX/IR（按 block/PEP）

NPU：按 (block_id, n_pad, m_pad) 编译 static IR 并缓存复用（避免重复编译）

Step 7：按 PEP 聚类并生成执行顺序

生成 pep_key（PEP 结构 + ratios + 如含 NPU 则包含 shape_key）

将 subgraphs 按 pep_key 分组（cluster）

runtime 依次批量执行每个 cluster，减少 plan switching

Runtime

按 cluster 顺序执行：

段内 DP：按 ratio 分发 subgraph token → async launch

段末 merge/搬运 → 下一段

NPU 直接复用缓存的 static IR（由 shape_key 决定）

English: Updated Compiler Design (Simple)
Goal

Run GNN inference on a heterogeneous edge platform (CPU/GPU/NPU). Given a 7-stage sequential GNN model and a large graph, generate an execution strategy that minimizes the global end-to-end total time (makespan). The strategy supports:

PP: split stages into 1/2/3 contiguous pipeline blocks

DP: run each block on multiple PUs with split ratios

Key assumptions

Graph partitioning is expensive: only evaluate a small set Kset={10,11,12,...}; run METIS once per k

NPU requires static shapes: satisfy it via padding to multiples of 1000 (no need to force equal subgraph sizes)

Reduce plan switching by clustering subgraphs that share the same PEP and executing them in batches

Use offline profiling lookup tables for compile-time cost estimation

Offline profiling (one-time)

Build lookup tables:

T_comp[device][stage_or_block][n,m]

Mem_peak[device][stage_or_block][n,m]

T_comm[src][dst](bytes) (or BW/Lat)

Compile (for each k in Kset)

METIS partition: produce k subgraphs with (n_i, m_i, optional cut stats)

NPU static template via padding:

n_pad = ceil(n_i/1000)*1000, m_pad = ceil(m_i/1000)*1000

If a PEP uses NPU, feasibility and cost use (n_pad,m_pad) plus padding/pack overhead

Generate candidate PEPs and estimate costs:

enumerate legal PEPs (1/2/3 segments, contiguous blocks, DP groups+ratios, device exclusivity, NPU stage constraints, memory feasibility)

estimate compute/comm/merge using lookup tables; keep Top-K optional

Global optimization (makespan minimization):

choose PEPs to balance total device loads: min max(Load_CPU, Load_GPU, Load_NPU)

greedy iterative balancing is sufficient initially

Pick best k by comparing estimated makespan T_k

Codegen + caching:

CPU/GPU: dynamic IR per block/PEP

NPU: static IR cached by (block_id, n_pad, m_pad)

PEP clustering:

group subgraphs by pep_key and execute in batches at runtime to reduce switching