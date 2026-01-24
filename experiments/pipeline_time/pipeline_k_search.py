import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import io
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. Reddit2 Partition Data (K=1 to K=10)
# ==========================================
reddit2_partitions = [
    # K=1
    {"k": 1, "subgraph_id": 0, "total_nodes": 232965, "internal_edges": 23213838},
    # K=2
    {"k": 2, "subgraph_id": 0, "total_nodes": 172755, "internal_edges": 9255308},
    {"k": 2, "subgraph_id": 1, "total_nodes": 180580, "internal_edges": 12628624},
    # K=3
    {"k": 3, "subgraph_id": 0, "total_nodes": 125538, "internal_edges": 6471810},
    {"k": 3, "subgraph_id": 1, "total_nodes": 162163, "internal_edges": 6886728},
    {"k": 3, "subgraph_id": 2, "total_nodes": 145638, "internal_edges": 7577348},
    # K=4
    {"k": 4, "subgraph_id": 0, "total_nodes": 119799, "internal_edges": 4897770},
    {"k": 4, "subgraph_id": 1, "total_nodes": 106059, "internal_edges": 4169684},
    {"k": 4, "subgraph_id": 2, "total_nodes": 118360, "internal_edges": 4833152},
    {"k": 4, "subgraph_id": 3, "total_nodes": 153951, "internal_edges": 6931354},
    # K=5
    {"k": 5, "subgraph_id": 0, "total_nodes": 96297, "internal_edges": 3378796},
    {"k": 5, "subgraph_id": 1, "total_nodes": 91483, "internal_edges": 4470720},
    {"k": 5, "subgraph_id": 2, "total_nodes": 91497, "internal_edges": 3321954},
    {"k": 5, "subgraph_id": 3, "total_nodes": 144895, "internal_edges": 5840834},
    {"k": 5, "subgraph_id": 4, "total_nodes": 120837, "internal_edges": 3488304},
    # K=6
    {"k": 6, "subgraph_id": 0, "total_nodes": 76034, "internal_edges": 2309272},
    {"k": 6, "subgraph_id": 1, "total_nodes": 90836, "internal_edges": 3284734},
    {"k": 6, "subgraph_id": 2, "total_nodes": 89731, "internal_edges": 3430732},
    {"k": 6, "subgraph_id": 3, "total_nodes": 81316, "internal_edges": 3135686},
    {"k": 6, "subgraph_id": 4, "total_nodes": 136867, "internal_edges": 3069920},
    {"k": 6, "subgraph_id": 5, "total_nodes": 112064, "internal_edges": 5144042},
    # K=7
    {"k": 7, "subgraph_id": 0, "total_nodes": 82564, "internal_edges": 3116138},
    {"k": 7, "subgraph_id": 1, "total_nodes": 68459, "internal_edges": 2069448},
    {"k": 7, "subgraph_id": 2, "total_nodes": 72987, "internal_edges": 2978622},
    {"k": 7, "subgraph_id": 3, "total_nodes": 93968, "internal_edges": 3896074},
    {"k": 7, "subgraph_id": 4, "total_nodes": 141569, "internal_edges": 3260868},
    {"k": 7, "subgraph_id": 5, "total_nodes": 71761, "internal_edges": 2585668},
    {"k": 7, "subgraph_id": 6, "total_nodes": 88366, "internal_edges": 2263332},
    # K=8
    {"k": 8, "subgraph_id": 0, "total_nodes": 85944, "internal_edges": 2311058},
    {"k": 8, "subgraph_id": 1, "total_nodes": 67427, "internal_edges": 2519868},
    {"k": 8, "subgraph_id": 2, "total_nodes": 58475, "internal_edges": 1709128},
    {"k": 8, "subgraph_id": 3, "total_nodes": 68467, "internal_edges": 2421854},
    {"k": 8, "subgraph_id": 4, "total_nodes": 59228, "internal_edges": 2422320},
    {"k": 8, "subgraph_id": 5, "total_nodes": 80208, "internal_edges": 2216248},
    {"k": 8, "subgraph_id": 6, "total_nodes": 138280, "internal_edges": 2746616},
    {"k": 8, "subgraph_id": 7, "total_nodes": 83126, "internal_edges": 3794162},
    # K=9
    {"k": 9, "subgraph_id": 0, "total_nodes": 94933, "internal_edges": 2184874},
    {"k": 9, "subgraph_id": 1, "total_nodes": 64826, "internal_edges": 2384938},
    {"k": 9, "subgraph_id": 2, "total_nodes": 70219, "internal_edges": 1638828},
    {"k": 9, "subgraph_id": 3, "total_nodes": 50613, "internal_edges": 1555170},
    {"k": 9, "subgraph_id": 4, "total_nodes": 53958, "internal_edges": 2332408},
    {"k": 9, "subgraph_id": 5, "total_nodes": 72842, "internal_edges": 1781822},
    {"k": 9, "subgraph_id": 6, "total_nodes": 118457, "internal_edges": 2108558},
    {"k": 9, "subgraph_id": 7, "total_nodes": 78081, "internal_edges": 3485394},
    {"k": 9, "subgraph_id": 8, "total_nodes": 112620, "internal_edges": 2273458},
    # K=10
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

# 硬件参数
GPU_SAFE_EDGES = 2200000

# ==========================================
# 2. 训练预测模型 (Linear CPU, Poly GPU)
# ==========================================
csv_data = """n_nodes,n_edges,cpu_ms,gpu_ms,npu_ms,status
5000,50000,71.07,15.32,12.30,OK
5000,125000,146.30,28.98,12.50,OK
5000,200000,244.49,44.29,13.69,OK
5000,250000,331.61,59.19,13.21,OK
5000,300000,396.35,67.61,14.59,OK
5000,375000,530.00,80.77,14.55,OK
5000,500000,668.75,104.37,14.86,OK
10000,100000,113.56,34.01,24.80,OK
10000,250000,383.24,60.04,26.66,OK
10000,400000,846.80,99.04,26.56,OK
10000,500000,883.66,118.77,26.46,OK
10000,600000,1212.22,137.15,26.93,OK
10000,750000,1412.72,171.70,27.26,OK
10000,1000000,1575.33,217.15,27.48,OK
20000,200000,469.97,63.60,47.63,OK
20000,500000,839.53,127.08,52.31,OK
20000,800000,1059.13,193.17,54.36,OK
20000,1000000,3208.38,226.60,53.90,OK
20000,1200000,1528.69,261.21,50.42,OK
20000,1500000,4112.32,319.91,52.25,OK
20000,2000000,4596.03,421.85,48.11,OK
50000,500000,806.00,164.16,119.30,OK
50000,1250000,1784.13,314.27,133.90,OK
50000,2000000,2827.87,457.83,128.53,OK
50000,2500000,5196.53,-1,125.29,GPU_OOM
50000,3000000,4509.23,-1,130.28,GPU_OOM
50000,3750000,6884.57,-1,129.52,GPU_OOM
50000,5000000,7874.66,-1,124.43,GPU_OOM
80000,800000,1458.18,250.19,190.19,OK
80000,2000000,3682.82,500.00,198.40,OK
80000,3200000,4344.77,-1,185.06,GPU_OOM
80000,4000000,6860.90,-1,191.39,GPU_OOM
80000,4800000,8040.32,-1,198.02,GPU_OOM
80000,6000000,10764.98,-1,194.12,GPU_OOM
80000,8000000,11805.94,-1,200.41,GPU_OOM
100000,1000000,2110.00,315.82,227.07,OK
100000,2500000,3341.44,-1,238.31,GPU_OOM
100000,4000000,6529.02,-1,239.83,GPU_OOM
100000,5000000,8668.05,-1,225.31,GPU_OOM
100000,6000000,9724.92,-1,245.92,GPU_OOM
100000,7500000,10737.94,-1,237.64,GPU_OOM
100000,10000000,13085.47,-1,231.96,GPU_OOM
"""

df = pd.read_csv(io.StringIO(csv_data))
# NPU Model
X_npu = df[['n_nodes']].values; y_npu = df['npu_ms'].values
model_npu = make_pipeline(PolynomialFeatures(degree=1), LinearRegression()).fit(X_npu, y_npu)

# CPU Model (Linear to fix 1ms bug)
X_cpu = df[['n_nodes', 'n_edges']].values; y_cpu = df['cpu_ms'].values
model_cpu = make_pipeline(PolynomialFeatures(degree=1), LinearRegression()).fit(X_cpu, y_cpu)

# GPU Model (Polynomial Degree 2)
df_gpu = df[df['gpu_ms'] != -1].copy()
X_gpu = df_gpu[['n_nodes', 'n_edges']].values; y_gpu = df_gpu['gpu_ms'].values
model_gpu = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X_gpu, y_gpu)

def predict_time(model, n, e, device):
    inputs = [[n]] if device=='npu' else [[n, e]]
    val = model.predict(inputs)[0]
    return max(1.0, val)

def find_optimal_split(n, total_edges):
    """Adaptive split: Find ratio where T_cpu approx T_gpu"""
    max_gpu_edges = min(total_edges, GPU_SAFE_EDGES)
    min_cpu_edges = total_edges - max_gpu_edges
    
    low = min_cpu_edges
    high = total_edges
    best_cpu_edges = low
    min_diff = float('inf')
    
    # Binary Search for equilibrium
    for _ in range(20):
        mid_cpu_edges = (low + high) / 2
        mid_gpu_edges = total_edges - mid_cpu_edges
        
        t_c = predict_time(model_cpu, n, mid_cpu_edges, 'cpu')
        t_g = predict_time(model_gpu, n, mid_gpu_edges, 'gpu')
        
        diff = t_c - t_g
        if abs(diff) < min_diff:
            min_diff = abs(diff)
            best_cpu_edges = mid_cpu_edges
        
        if t_c < t_g: low = mid_cpu_edges
        else: high = mid_cpu_edges
            
    final_cpu_edges = best_cpu_edges
    final_gpu_edges = total_edges - final_cpu_edges
    t_cpu = predict_time(model_cpu, n, final_cpu_edges, 'cpu')
    t_gpu = predict_time(model_gpu, n, final_gpu_edges, 'gpu')
    
    # Determine Mode
    # If calculate CPU edges is very close to the forced minimum, it's Overflow
    if min_cpu_edges > 0 and final_cpu_edges <= min_cpu_edges + 1000:
        mode = "Overflow"
    else:
        mode = "Adaptive"
        
    return t_cpu, t_gpu, final_cpu_edges / total_edges, mode

# ==========================================
# 3. Analyze Loop (K=1 to 10)
# ==========================================
# Store results for all Ks
all_k_data = {} 

print("Running analysis for K=1 to K=10...")

for k in range(1, 11):
    subgraphs = [x for x in reddit2_partitions if x['k'] == k]
    if not subgraphs: continue
    
    # --- Step 1: Estimation ---
    estimated_jobs = []
    for sg in subgraphs:
        n, e = sg['total_nodes'], sg['internal_edges']
        t_s2 = predict_time(model_npu, n, e, 'npu')
        t_s1_cpu, t_s1_gpu, ratio, mode = find_optimal_split(n, e)
        t_s1 = max(t_s1_cpu, t_s1_gpu)
        
        estimated_jobs.append({
            'id': sg['subgraph_id'], 'nodes': n, 'edges': e,
            's1': t_s1, 's2': t_s2, 'mode': mode, 'cpu_ratio': ratio,
            's1_cpu': t_s1_cpu, 's1_gpu': t_s1_gpu
        })
        
    # --- Step 2: Sorting (SJF) ---
    sorted_jobs = sorted(estimated_jobs, key=lambda x: x['s1'])
    
    # --- Step 3: Pipeline Schedule ---
    s1_avail = 0
    s2_avail = 0
    schedule_log = []
    
    for idx, job in enumerate(sorted_jobs):
        start_1 = s1_avail
        finish_1 = start_1 + job['s1']
        s1_avail = finish_1
        
        start_2 = max(finish_1, s2_avail)
        finish_2 = start_2 + job['s2']
        wait = start_2 - s2_avail if s2_avail > 0 else 0
        s2_avail = finish_2
        
        # Utilization Logic
        # Slot duration = The bottleneck time for this specific task
        slot_duration = max(job['s1'], job['s2'])
        cpu_util = (job['s1_cpu'] / slot_duration * 100)
        gpu_util = (job['s1_gpu'] / slot_duration * 100)
        npu_util = (job['s2'] / slot_duration * 100)

        schedule_log.append({
            'seq': idx+1, 'id': job['id'],
            's1_range': f"{start_1:.0f} - {finish_1:.0f}",
            's2_range': f"{start_2:.0f} - {finish_2:.0f}",
            'wait': wait,
            'cpu_util': cpu_util, 'gpu_util': gpu_util, 'npu_util': npu_util
        })
        
    # Save everything for this K
    all_k_data[k] = {
        'total_latency': s2_avail,
        'speedup': 0, # Will calc later
        'jobs_detail': sorted_jobs,
        'schedule': schedule_log
    }

# Calculate speedup relative to K=1
base_latency = all_k_data[1]['total_latency']
for k in all_k_data:
    all_k_data[k]['speedup'] = base_latency / all_k_data[k]['total_latency']

# Find Winner
best_k = min(all_k_data, key=lambda k: all_k_data[k]['total_latency'])
min_latency = all_k_data[best_k]['total_latency']

# ==========================================
# 4. Generate Full Markdown Report
# ==========================================
output_file = "reddit2_full_analysis.md"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("# Reddit2 Pipeline Analysis (Full Report)\n\n")
    
    # --- Summary Section ---
    f.write("## 1. Comparative Summary\n")
    f.write(f"**Winner:** K={best_k} (Latency: `{min_latency:.0f} ms`)\n\n")
    f.write("| K | Total Latency (ms) | Speedup (vs K=1) | Throughput (items/sec) |\n")
    f.write("|:---:|:---:|:---:|:---:|\n")
    
    for k in range(1, 11):
        if k not in all_k_data: continue
        d = all_k_data[k]
        tp = 1.0 if k==1 else k / (d['total_latency']/1000)
        marker = "🏆" if k == best_k else ""
        f.write(f"| {k} | **{d['total_latency']:.0f}** {marker} | {d['speedup']:.2f}x | {tp:.2f} |\n")

    # --- Detailed Analysis for Each K ---
    f.write("\n---\n")
    for k in range(1, 11):
        if k not in all_k_data: continue
        data = all_k_data[k]
        
        f.write(f"\n## Analysis for K={k}\n")
        f.write(f"**Total Latency:** `{data['total_latency']:.2f} ms`\n\n")
        
        # Table 1: Estimation
        f.write("### 1. Stage Performance Estimation (Adaptive Split)\n")
        f.write("| ID | Nodes | Edges (M) | Mode | **Optimal CPU%** | S1 CPU (ms) | S1 GPU (ms) | **Stage 1 (ms)** | Stage 2 (ms) |\n")
        f.write("|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|\n")
        
        for job in data['jobs_detail']:
            f.write(f"| {job['id']} | {job['nodes']} | {job['edges']/1e6:.2f} | {job['mode']} | "
                    f"**{job['cpu_ratio']*100:.1f}%** | {job['s1_cpu']:.0f} | {job['s1_gpu']:.0f} | **{job['s1']:.0f}** | {job['s2']:.0f} |\n")
            
        # Table 2: Schedule
        f.write("\n### 2. Pipeline Execution Schedule\n")
        f.write("| Seq | ID | S1 Interval (ms) | S2 Interval (ms) | Wait (Bubble) | CPU Util% | GPU Util% | NPU Util% |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        
        for s in data['schedule']:
            f.write(f"| {s['seq']} | **{s['id']}** | {s['s1_range']} | {s['s2_range']} | {s['wait']:.0f} ms | "
                    f"{s['cpu_util']:.1f}% | {s['gpu_util']:.1f}% | {s['npu_util']:.1f}% |\n")
            
        f.write("\n---\n")

print(f"Analysis Complete. Full report saved to {output_file}")