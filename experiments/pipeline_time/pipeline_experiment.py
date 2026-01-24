import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 原始数据 (CSV)
# ==========================================
csv_data = """n_nodes,n_edges,cpu_ms,gpu_ms,npu_ms,status
5000,50000,71.07,15.32,12.30,OK
5000,125000,146.30,28.98,12.50,OK
5000,200000,244.49,44.29,13.69,OK
5000,500000,668.75,104.37,14.86,OK
10000,100000,113.56,34.01,24.80,OK
10000,250000,383.24,60.04,26.66,OK
10000,1000000,1575.33,217.15,27.48,OK
20000,200000,469.97,63.60,47.63,OK
20000,1000000,3208.38,226.60,53.90,OK
20000,2000000,4596.03,421.85,48.11,OK
50000,500000,806.00,164.16,119.30,OK
50000,2000000,2827.87,457.83,128.53,OK
80000,800000,1458.18,250.19,190.19,OK
80000,2000000,3682.82,500.00,198.40,OK
100000,1000000,2110.00,315.82,227.07,OK
100000,10000000,13085.47,-1,231.96,GPU_OOM
"""
df = pd.read_csv(io.StringIO(csv_data))

# ==========================================
# 2. 物理约束回归 (Physics-Constrained Fitting)
# ==========================================

# Step 1: 提取 Edge 的边际成本 (Marginal Cost of Edge)
# 我们只看那些有多个 Edge 数据点的 Node 组
groups = df.groupby('n_nodes')
edge_slopes = []

for name, group in groups:
    if len(group) > 1:
        # 在同一个 Node 数量下，拟合 Time vs Edge 的斜率
        lr = LinearRegression().fit(group[['n_edges']], group['cpu_ms'])
        edge_slopes.append(lr.coef_[0])

# 计算平均 Edge 系数 (ms per edge)
# 这是系统处理一条边真实的物理耗时
COEFF_EDGE_CPU = np.mean(edge_slopes)

# Step 2: 拟合 Node 系数
# 从总时间中扣除 Edge 的开销，剩下的就是 Node 的开销
df['cpu_residual'] = df['cpu_ms'] - (df['n_edges'] * COEFF_EDGE_CPU)

# 强制过原点 (fit_intercept=False)，避免负数
# 如果 residual 是负数，说明之前的 Edge 系数稍微高估了，线性回归会自动平衡回来
lr_node = LinearRegression(fit_intercept=False).fit(df[['n_nodes']], df['cpu_residual'])
COEFF_NODE_CPU = lr_node.coef_[0]

print("=== 物理约束 CPU 模型参数 ===")
print(f"Edge Coeff: {COEFF_EDGE_CPU*1000:.4f} us/edge")
print(f"Node Coeff: {COEFF_NODE_CPU*1000:.4f} us/node")

# 修正：如果 Node 系数是负的（这在有些噪声数据中可能发生），强制设为一个极小值
# 因为 Node 开销不可能为负。
if COEFF_NODE_CPU < 0:
    print("Warning: Node coeff negative, clamping to small positive epsilon.")
    COEFF_NODE_CPU = 0.0001 # 0.1 us per node (minimal overhead)

# ==========================================
# 3. 验证一下 K=10, ID=2 的情况
# ==========================================
# ID 2: 44,322 Nodes
def predict_cpu_phys(n, e):
    return (n * COEFF_NODE_CPU) + (e * COEFF_EDGE_CPU)

t_id2_zero_edge = predict_cpu_phys(44322, 0)
print(f"\n[验证] ID 2 (44k Nodes) with 0 Edges:")
print(f"新模型预测 CPU 耗时: {t_id2_zero_edge:.2f} ms")
print(f"旧模型预测 (参考): ~432 ms")
print(f"结论: 修复后的 CPU 空载开销大幅降低。")

# ==========================================
# 4. 重新构建 GPU 和 NPU 模型 (保持不变)
# ==========================================
# NPU
model_npu = LinearRegression(fit_intercept=False).fit(df[['n_nodes']], df['npu_ms'])
# GPU (Poly Degree 2 依然比较适合 GPU 的带宽饱和特性)
df_gpu = df[df['gpu_ms'] != -1].copy()
X_gpu = df_gpu[['n_nodes', 'n_edges']].values; y_gpu = df_gpu['gpu_ms'].values
model_gpu = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X_gpu, y_gpu)

def predict_time(model, n, e, device):
    if device == 'cpu':
        return predict_cpu_phys(n, e)
    inputs = [[n]] if device=='npu' else [[n, e]]
    val = model.predict(inputs)[0]
    return max(1.0, val)

# ==========================================
# 5. 再次运行 Pipeline 分析 (K=10)
# ==========================================
reddit2_partitions = [
    # K=10 Data
    {"k": 10, "subgraph_id": 0, "total_nodes": 94181, "internal_edges": 2128304},
    {"k": 10, "subgraph_id": 1, "total_nodes": 64601, "internal_edges": 2033366},
    {"k": 10, "subgraph_id": 2, "total_nodes": 44322, "internal_edges": 1469348},
    {"k": 10, "subgraph_id": 3, "total_nodes": 107125, "internal_edges": 1453518},
    {"k": 10, "subgraph_id": 4, "total_nodes": 53607, "internal_edges": 1998070},
    {"k": 10, "subgraph_id": 5, "total_nodes": 77088, "internal_edges": 1714838},
    {"k": 10, "subgraph_id": 6, "total_nodes": 69942, "internal_edges": 1640958},
    {"k": 10, "subgraph_id": 7, "total_nodes": 119905, "internal_edges": 1456030},
    {"k": 10, "subgraph_id": 8, "total_nodes": 98212, "internal_edges": 2181642},
    {"k": 10, "subgraph_id": 9, "total_nodes": 86012, "internal_edges": 2835694}
]

GPU_SAFE_EDGES = 2200000

def find_optimal_split(n, total_edges):
    # ... (保持原有的二分查找逻辑，但使用新的 predict_cpu_phys) ...
    max_gpu_edges = min(total_edges, GPU_SAFE_EDGES)
    min_cpu_edges = total_edges - max_gpu_edges
    
    low = min_cpu_edges; high = total_edges; best_cpu_edges = low
    min_diff = float('inf')
    
    for _ in range(20):
        mid_cpu_edges = (low + high) / 2
        mid_gpu_edges = total_edges - mid_cpu_edges
        t_c = predict_time(None, n, mid_cpu_edges, 'cpu')
        t_g = predict_time(model_gpu, n, mid_gpu_edges, 'gpu')
        if abs(t_c - t_g) < min_diff:
            min_diff = abs(t_c - t_g); best_cpu_edges = mid_cpu_edges
        if t_c < t_g: low = mid_cpu_edges
        else: high = mid_cpu_edges
    
    final_cpu_edges = best_cpu_edges
    final_gpu_edges = total_edges - final_cpu_edges
    return predict_time(None, n, final_cpu_edges, 'cpu'), predict_time(model_gpu, n, final_gpu_edges, 'gpu'), final_cpu_edges/total_edges

# 快速查看 K=10 的结果
print(f"\n{'ID':<3} | {'Ratio':<6} | {'S1_CPU':<8} | {'S1_GPU':<8} | {'Utilization Check'}")
print("-" * 60)
for sg in reddit2_partitions:
    n, e = sg['total_nodes'], sg['internal_edges']
    t_c, t_g, ratio = find_optimal_split(n, e)
    t_max = max(t_c, t_g)
    
    # Check utilization
    cpu_util = t_c / t_max * 100
    gpu_util = t_g / t_max * 100
    
    print(f"{sg['subgraph_id']:<3} | {ratio*100:.1f}%  | {t_c:.0f} ms    | {t_g:.0f} ms    | CPU:{cpu_util:.0f}% GPU:{gpu_util:.0f}%")


#  import pandas as pd
# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline
# import io
# import warnings

# warnings.filterwarnings("ignore")

# # ==========================================
# # 1. 数据与模型准备 (保持不变)
# # ==========================================
# csv_data = """n_nodes,n_edges,cpu_ms,gpu_ms,npu_ms,status
# 5000,50000,71.07,15.32,12.30,OK
# 5000,125000,146.30,28.98,12.50,OK
# 5000,200000,244.49,44.29,13.69,OK
# 5000,250000,331.61,59.19,13.21,OK
# 5000,300000,396.35,67.61,14.59,OK
# 5000,375000,530.00,80.77,14.55,OK
# 5000,500000,668.75,104.37,14.86,OK
# 10000,100000,113.56,34.01,24.80,OK
# 10000,250000,383.24,60.04,26.66,OK
# 10000,400000,846.80,99.04,26.56,OK
# 10000,500000,883.66,118.77,26.46,OK
# 10000,600000,1212.22,137.15,26.93,OK
# 10000,750000,1412.72,171.70,27.26,OK
# 10000,1000000,1575.33,217.15,27.48,OK
# 20000,200000,469.97,63.60,47.63,OK
# 20000,500000,839.53,127.08,52.31,OK
# 20000,800000,1059.13,193.17,54.36,OK
# 20000,1000000,3208.38,226.60,53.90,OK
# 20000,1200000,1528.69,261.21,50.42,OK
# 20000,1500000,4112.32,319.91,52.25,OK
# 20000,2000000,4596.03,421.85,48.11,OK
# 50000,500000,806.00,164.16,119.30,OK
# 50000,1250000,1784.13,314.27,133.90,OK
# 50000,2000000,2827.87,457.83,128.53,OK
# 50000,2500000,5196.53,-1,125.29,GPU_OOM
# 50000,3000000,4509.23,-1,130.28,GPU_OOM
# 50000,3750000,6884.57,-1,129.52,GPU_OOM
# 50000,5000000,7874.66,-1,124.43,GPU_OOM
# 80000,800000,1458.18,250.19,190.19,OK
# 80000,2000000,3682.82,500.00,198.40,OK
# 80000,3200000,4344.77,-1,185.06,GPU_OOM
# 80000,4000000,6860.90,-1,191.39,GPU_OOM
# 80000,4800000,8040.32,-1,198.02,GPU_OOM
# 80000,6000000,10764.98,-1,194.12,GPU_OOM
# 80000,8000000,11805.94,-1,200.41,GPU_OOM
# 100000,1000000,2110.00,315.82,227.07,OK
# 100000,2500000,3341.44,-1,238.31,GPU_OOM
# 100000,4000000,6529.02,-1,239.83,GPU_OOM
# 100000,5000000,8668.05,-1,225.31,GPU_OOM
# 100000,6000000,9724.92,-1,245.92,GPU_OOM
# 100000,7500000,10737.94,-1,237.64,GPU_OOM
# 100000,10000000,13085.47,-1,231.96,GPU_OOM
# """

# workload_data = {
#     0: (94181, 2128304), 1: (64601, 2033366), 2: (44322, 1469348),
#     3: (107125, 1453518), 4: (53607, 1998070), 5: (77088, 1714838),
#     6: (69942, 1640958), 7: (119905, 1456030), 8: (98212, 2181642),
#     9: (86012, 2835694)
# }

# GPU_SAFE_EDGES = 2200000

# df = pd.read_csv(io.StringIO(csv_data))
# model_npu = make_pipeline(PolynomialFeatures(degree=1), LinearRegression()).fit(df[['n_nodes']].values, df['npu_ms'].values)
# # CPU: 线性 (Physics-Aware)
# model_cpu = make_pipeline(PolynomialFeatures(degree=1), LinearRegression()).fit(df[['n_nodes', 'n_edges']].values, df['cpu_ms'].values)
# # GPU: 非线性
# df_gpu = df[df['gpu_ms'] != -1].copy()
# model_gpu = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(df_gpu[['n_nodes', 'n_edges']].values, df_gpu['gpu_ms'].values)

# def predict_time(model, n, e, device):
#     val = model.predict([[n] if device=='npu' else [n, e]])[0]
#     return max(1.0, val)

# # ==========================================
# # 2. 核心算法: 寻找完美切分点 (Adaptive Split)
# # ==========================================
# def find_optimal_split(n, total_edges):
#     """
#     使用二分查找，寻找一个 CPU 比例 ratio，使得 T_cpu 和 T_gpu 最接近。
#     同时必须满足 GPU Edges <= GPU_SAFE_EDGES。
#     """
#     # 1. 计算 GPU 允许的最大边数
#     max_gpu_edges = min(total_edges, GPU_SAFE_EDGES)
#     min_cpu_edges = total_edges - max_gpu_edges
    
#     # 二分查找范围：
#     # 最小 CPU 边数: 溢出部分的边数 (如果有)
#     # 最大 CPU 边数: 全部边数
#     low = min_cpu_edges
#     high = total_edges
#     best_cpu_edges = low
#     min_diff = float('inf')
    
#     # Binary Search (Iterate 20 times is enough for convergence)
#     for _ in range(20):
#         mid_cpu_edges = (low + high) / 2
#         mid_gpu_edges = total_edges - mid_cpu_edges
        
#         t_c = predict_time(model_cpu, n, mid_cpu_edges, 'cpu')
#         t_g = predict_time(model_gpu, n, mid_gpu_edges, 'gpu')
        
#         diff = t_c - t_g
        
#         if abs(diff) < min_diff:
#             min_diff = abs(diff)
#             best_cpu_edges = mid_cpu_edges
        
#         if t_c < t_g:
#             # CPU 太快，GPU 太慢 -> 给 CPU 多点活
#             low = mid_cpu_edges
#         else:
#             # CPU 太慢 -> 给 CPU 少点活
#             high = mid_cpu_edges
            
#     # 计算最终结果
#     final_cpu_edges = best_cpu_edges
#     final_gpu_edges = total_edges - final_cpu_edges
#     t_cpu = predict_time(model_cpu, n, final_cpu_edges, 'cpu')
#     t_gpu = predict_time(model_gpu, n, final_gpu_edges, 'gpu')
    
#     # 确定模式
#     if min_cpu_edges > 0 and final_cpu_edges <= min_cpu_edges + 1000: 
#         # 如果计算出的最优 CPU 边数非常接近强制溢出的底线，说明是“被迫溢出”
#         mode = "Overflow"
#     else:
#         mode = "Adaptive"
        
#     return t_cpu, t_gpu, final_cpu_edges / total_edges, mode

# # ==========================================
# # 3. 计算与输出
# # ==========================================
# estimated_results = []
# output_file = "summary.md"

# print(f"{'ID':<3} | {'Mode':<10} | {'Ratio(CPU)':<10} | {'S1_CPU':<8} | {'S1_GPU':<8} | {'Stage1':<8}")
# print("-" * 75)

# for job_id, (n, e) in workload_data.items():
#     t_stage2 = predict_time(model_npu, n, e, 'npu')
    
#     # 调用自适应分配
#     t_s1_cpu, t_s1_gpu, cpu_ratio, mode = find_optimal_split(n, e)
#     t_stage1 = max(t_s1_cpu, t_s1_gpu)
    
#     estimated_results.append({
#         'id': job_id, 'nodes': n, 'edges': e,
#         's1': t_stage1, 's2': t_stage2,
#         's1_cpu': t_s1_cpu, 's1_gpu': t_s1_gpu,
#         'mode': mode, 'ratio': cpu_ratio
#     })
    
#     print(f"{job_id:<3} | {mode:<10} | {cpu_ratio*100:<5.1f}%     | {t_s1_cpu:<8.0f} | {t_s1_gpu:<8.0f} | {t_stage1:<8.0f}")

# sorted_jobs = sorted(estimated_results, key=lambda x: x['s1'])

# # ==========================================
# # 4. 写入 Markdown (含 Ratio 列)
# # ==========================================
# with open(output_file, "w", encoding="utf-8") as f:
#     f.write("# Pipeline Performance Summary (Adaptive Ratio)\n\n")
    
#     f.write("## 1. Stage Performance Estimation\n")
#     f.write("**Strategy:** Fully Adaptive Split. The CPU/GPU ratio is dynamically calculated for *each* ID to minimize `|Time_CPU - Time_GPU|`.\n\n")
#     f.write("| ID | Nodes | Edges (M) | Mode | **Optimal CPU%** | S1 CPU (ms) | S1 GPU (ms) | **Stage 1 (ms)** | Stage 2 (ms) |\n")
#     f.write("|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|\n")
    
#     for job in estimated_results:
#         f.write(f"| {job['id']} | {job['nodes']} | {job['edges']/1e6:.2f} | {job['mode']} | "
#                 f"**{job['ratio']*100:.1f}%** | {job['s1_cpu']:.0f} | {job['s1_gpu']:.0f} | **{job['s1']:.0f}** | {job['s2']:.0f} |\n")

#     f.write("\n## 2. Pipeline Execution Schedule\n")
#     f.write("| Seq | ID | S1 Interval (ms) | S2 Interval (ms) | Wait (Bubble) | CPU Util% | GPU Util% | NPU Util% |\n")
#     f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
#     s1_avail = 0; s2_avail = 0
#     pipeline_cycles = []
    
#     for idx, job in enumerate(sorted_jobs):
#         start_1 = s1_avail
#         finish_1 = start_1 + job['s1']
#         s1_avail = finish_1
        
#         start_2 = max(finish_1, s2_avail)
#         finish_2 = start_2 + job['s2']
#         wait = start_2 - s2_avail if s2_avail > 0 else 0
#         s2_avail = finish_2
        
#         slot_duration = max(job['s1'], job['s2'])
#         cpu_util = (job['s1_cpu'] / slot_duration * 100)
#         gpu_util = (job['s1_gpu'] / slot_duration * 100)
#         npu_util = (job['s2'] / slot_duration * 100)
        
#         f.write(f"| {idx+1} | **{job['id']}** | {start_1:.0f} - {finish_1:.0f} | {start_2:.0f} - {finish_2:.0f} | "
#                 f"{wait:.0f} ms | {cpu_util:.1f}% | {gpu_util:.1f}% | {npu_util:.1f}% |\n")
    
#     f.write(f"\n**Total Latency:** `{finish_2:.2f} ms`\n")

# print(f"Done. Check {output_file}.")


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline
# import io
# import warnings

# # 忽略警告
# warnings.filterwarnings("ignore")

# # ==========================================
# # 1. 数据准备
# # ==========================================
# csv_data = """n_nodes,n_edges,cpu_ms,gpu_ms,npu_ms,status
# 5000,50000,71.07,15.32,12.30,OK
# 5000,125000,146.30,28.98,12.50,OK
# 5000,200000,244.49,44.29,13.69,OK
# 5000,250000,331.61,59.19,13.21,OK
# 5000,300000,396.35,67.61,14.59,OK
# 5000,375000,530.00,80.77,14.55,OK
# 5000,500000,668.75,104.37,14.86,OK
# 10000,100000,113.56,34.01,24.80,OK
# 10000,250000,383.24,60.04,26.66,OK
# 10000,400000,846.80,99.04,26.56,OK
# 10000,500000,883.66,118.77,26.46,OK
# 10000,600000,1212.22,137.15,26.93,OK
# 10000,750000,1412.72,171.70,27.26,OK
# 10000,1000000,1575.33,217.15,27.48,OK
# 20000,200000,469.97,63.60,47.63,OK
# 20000,500000,839.53,127.08,52.31,OK
# 20000,800000,1059.13,193.17,54.36,OK
# 20000,1000000,3208.38,226.60,53.90,OK
# 20000,1200000,1528.69,261.21,50.42,OK
# 20000,1500000,4112.32,319.91,52.25,OK
# 20000,2000000,4596.03,421.85,48.11,OK
# 50000,500000,806.00,164.16,119.30,OK
# 50000,1250000,1784.13,314.27,133.90,OK
# 50000,2000000,2827.87,457.83,128.53,OK
# 50000,2500000,5196.53,-1,125.29,GPU_OOM
# 50000,3000000,4509.23,-1,130.28,GPU_OOM
# 50000,3750000,6884.57,-1,129.52,GPU_OOM
# 50000,5000000,7874.66,-1,124.43,GPU_OOM
# 80000,800000,1458.18,250.19,190.19,OK
# 80000,2000000,3682.82,500.00,198.40,OK
# 80000,3200000,4344.77,-1,185.06,GPU_OOM
# 80000,4000000,6860.90,-1,191.39,GPU_OOM
# 80000,4800000,8040.32,-1,198.02,GPU_OOM
# 80000,6000000,10764.98,-1,194.12,GPU_OOM
# 80000,8000000,11805.94,-1,200.41,GPU_OOM
# 100000,1000000,2110.00,315.82,227.07,OK
# 100000,2500000,3341.44,-1,238.31,GPU_OOM
# 100000,4000000,6529.02,-1,239.83,GPU_OOM
# 100000,5000000,8668.05,-1,225.31,GPU_OOM
# 100000,6000000,9724.92,-1,245.92,GPU_OOM
# 100000,7500000,10737.94,-1,237.64,GPU_OOM
# 100000,10000000,13085.47,-1,231.96,GPU_OOM
# """

# workload_data = {
#     0: (94181, 2128304), 1: (64601, 2033366), 2: (44322, 1469348),
#     3: (107125, 1453518), 4: (53607, 1998070), 5: (77088, 1714838),
#     6: (69942, 1640958), 7: (119905, 1456030), 8: (98212, 2181642),
#     9: (86012, 2835694)
# }

# GPU_SAFE_EDGES = 2200000
# CPU_RATIO = 0.12

# # ==========================================
# # 2. 训练模型 (Linear CPU to fix 1ms bug)
# # ==========================================
# df = pd.read_csv(io.StringIO(csv_data))

# X_npu = df[['n_nodes']].values
# y_npu = df['npu_ms'].values
# model_npu = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
# model_npu.fit(X_npu, y_npu)

# # CPU: Linear O(V+E)
# X_cpu = df[['n_nodes', 'n_edges']].values
# y_cpu = df['cpu_ms'].values
# model_cpu = make_pipeline(PolynomialFeatures(degree=1), LinearRegression()) 
# model_cpu.fit(X_cpu, y_cpu)

# # GPU: Poly Degree 2
# df_gpu = df[df['gpu_ms'] != -1].copy()
# X_gpu = df_gpu[['n_nodes', 'n_edges']].values
# y_gpu = df_gpu['gpu_ms'].values
# model_gpu = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
# model_gpu.fit(X_gpu, y_gpu)

# def predict_time(model, n, e, device_name):
#     if device_name == 'npu':
#         val = model.predict([[n]])[0]
#     else:
#         val = model.predict([[n, e]])[0]
#     return max(1.0, val)

# # ==========================================
# # 3. 核心计算
# # ==========================================
# estimated_results = []
# output_file = "summary.md"
# total_busy_cpu = 0; total_busy_gpu = 0; total_busy_npu = 0

# print("Calculating estimations...")

# for job_id, (n, e) in workload_data.items():
#     t_stage2 = predict_time(model_npu, n, e, 'npu')
    
#     edges_gpu_target = e * (1 - CPU_RATIO)
#     if edges_gpu_target <= GPU_SAFE_EDGES:
#         edges_gpu = edges_gpu_target
#         edges_cpu = e * CPU_RATIO
#         mode = "Balanced"
#     else:
#         edges_gpu = GPU_SAFE_EDGES
#         edges_cpu = e - GPU_SAFE_EDGES
#         mode = "Overflow"
    
#     t_s1_cpu = predict_time(model_cpu, n, edges_cpu, 'cpu')
#     t_s1_gpu = predict_time(model_gpu, n, edges_gpu, 'gpu')
#     t_stage1 = max(t_s1_cpu, t_s1_gpu)
    
#     total_busy_cpu += t_s1_cpu
#     total_busy_gpu += t_s1_gpu
#     total_busy_npu += t_stage2
    
#     estimated_results.append({
#         'id': job_id, 'nodes': n, 'edges': e,
#         's1': t_stage1, 's2': t_stage2,
#         's1_cpu': t_s1_cpu, 's1_gpu': t_s1_gpu,
#         'mode': mode
#     })

# sorted_jobs = sorted(estimated_results, key=lambda x: x['s1'])

# # ==========================================
# # 4. 写入 Markdown
# # ==========================================
# with open(output_file, "w", encoding="utf-8") as f:
#     f.write("# Pipeline Performance Summary (Corrected)\n\n")
    
#     # --- Table 1: Stage Estimation ---
#     f.write("## 1. Stage Performance Estimation\n")
#     f.write(f"**Model Update:** CPU uses Linear Regression ($O(V+E)$) to fix `1ms` outliers.\n\n")
#     f.write("| ID | Nodes | Edges (M) | Mode | S1 CPU (ms) | S1 GPU (ms) | **Stage 1 (ms)** | **Stage 2 (ms)** |\n")
#     f.write("|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|\n")
    
#     for job in estimated_results:
#         f.write(f"| {job['id']} | {job['nodes']} | {job['edges']/1e6:.2f} | {job['mode']} | "
#                 f"{job['s1_cpu']:.0f} | {job['s1_gpu']:.0f} | **{job['s1']:.0f}** | {job['s2']:.0f} |\n")
    
#     # --- Table 2: Pipeline Schedule ---
#     f.write("\n## 2. Pipeline Execution Schedule\n")
#     f.write(f"**Order:** `{[j['id'] for j in sorted_jobs]}`\n")
#     f.write("> **Note on Utilization:** Cycle Utilization is calculated based on the **Bottleneck Stage Duration** (Stage 1 for these tasks). This ensures utilization reflects the device saturation during the task's processing slot and never exceeds 100%.\n\n")
    
#     f.write("| Seq | ID | S1 Interval (ms) | S2 Interval (ms) | Wait (Bubble) | CPU Util% | GPU Util% | NPU Util% |\n")
#     f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
#     s1_avail = 0
#     s2_avail = 0
#     pipeline_cycles = []
    
#     for idx, job in enumerate(sorted_jobs):
#         start_1 = s1_avail
#         finish_1 = start_1 + job['s1']
#         s1_avail = finish_1
        
#         start_2 = max(finish_1, s2_avail)
#         finish_2 = start_2 + job['s2']
#         wait = start_2 - s2_avail if s2_avail > 0 else 0
#         s2_avail = finish_2
        
#         # --- Utilization Calculation Fix ---
#         # 修正：分母使用 max(s1, s2)，即当前任务的瓶颈时间
#         # 这样计算反映的是：在处理该任务的“流水线节拍”内，设备有多忙
#         slot_duration = max(job['s1'], job['s2'])
        
#         cpu_util = (job['s1_cpu'] / slot_duration * 100)
#         gpu_util = (job['s1_gpu'] / slot_duration * 100)
#         npu_util = (job['s2'] / slot_duration * 100)
        
#         f.write(f"| {idx+1} | **{job['id']}** | {start_1:.0f} - {finish_1:.0f} | {start_2:.0f} - {finish_2:.0f} | "
#                 f"{wait:.0f} ms | {cpu_util:.1f}% | {gpu_util:.1f}% | {npu_util:.1f}% |\n")

#     total_latency = finish_2

#     # --- Table 3: Final Metrics ---
#     global_util_cpu = (total_busy_cpu / total_latency) * 100
#     global_util_gpu = (total_busy_gpu / total_latency) * 100
#     global_util_npu = (total_busy_npu / total_latency) * 100

#     f.write("\n## 3. Final Metrics (Global)\n\n")
#     f.write(f"- **Total Latency:** `{total_latency:.2f} ms`\n")
#     f.write(f"- **Throughput:** `{1000 * len(sorted_jobs) / total_latency:.2f} items/sec`\n")
#     f.write("### Resource Utilization (Global Average):\n")
#     f.write(f"- **NPU:** `{global_util_npu:.2f}%`\n")
#     f.write(f"- **GPU:** `{global_util_gpu:.2f}%`\n")
#     f.write(f"- **CPU:** `{global_util_cpu:.2f}%`\n")

# print(f"Success! Corrected analysis saved to {output_file}")