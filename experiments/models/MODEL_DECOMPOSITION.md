# GNN Model Decomposition 方案文档 (v2.0 - 原语化重构)

本文档描述了 GraphSAGE、GAT、GCN 三种 GNN 模型的**原语化分解方案**，采用"原子化 + 泛化"策略，最大化 Profiling 复用。

---

## 核心优化：最小公共原语集 (Minimal Common Primitive Set)

### 设计理念

> **不要根据"模型阶段"命名算子，而是根据"数学操作"命名算子**

传统方案按模型阶段划分：
- SAGE: 7 stages (GATHER, MESSAGE, REDUCE_SUM, ...)
- GAT: 7 stages (LINEAR, GATHER_BOTH, ATT_SCORE, ...)
- GCN: 6 stages (COMPUTE_NORM, GATHER, MESSAGE, ...)
- **总计: 20 个需要 Profile 的阶段**

新方案按数学操作划分：
- **7 个通用原语**覆盖所有模型
- **Profiling 工作量减少 ~65%**

### 原语库定义

| ID | 原语名称 | 核心操作 | 复用情况 | NPU 兼容 |
|----|----------|----------|----------|----------|
| **P1** | MATMUL | `F.linear / matmul` | SAGE, GAT, GCN (100%) | ✅ |
| **P2** | GATHER | `x[idx]` / `index_select` | SAGE, GAT(×2), GCN (100%) | ✅ |
| **P3** | SCATTER_ADD | `scatter_add` | SAGE, GAT, GCN (100%) | ❌ |
| **P4** | ELEWISE_MUL | `a * b` (Broadcast) | SAGE, GAT, GCN (100%) | ✅ |
| **P5** | ELEWISE_ACT | `ReLU / ELU` | SAGE, GAT, GCN (100%) | ✅ |
| **P6** | GAT_EDGE_ATT | `LeakyReLU(a·(x_i+x_j))` | GAT only (33%) | ✅ |
| **P7** | GAT_SOFTMAX | Edge Softmax (含scatter) | GAT only (33%) | ❌ |

### Profiling 效率对比

| 方案 | 需要 Profile 的数量 | 复用率 |
|------|---------------------|--------|
| 原方案 (按模型阶段) | 7 + 7 + 6 = 20 | ~30% |
| **新方案 (原语库)** | **5 通用 + 2 GAT = 7** | **~90%** |

---

## 一、GraphSAGE → 原语映射

### 1.1 计算流程

```
GraphSAGE (mean aggregator):
h_i' = σ( W_l · mean({h_j : j ∈ N(i)}) + W_r · h_i )
```

### 1.2 原语映射

| 原 Stage | 原名称 | → 原语 | 说明 |
|----------|--------|--------|------|
| 1 | GATHER | P2: GATHER | `x[edge_index[0]]` |
| 2 | MESSAGE | **(Skip)** | 恒等函数，零开销 |
| 3 | REDUCE_SUM | P3: SCATTER_ADD | `scatter_add` |
| 4 | REDUCE_COUNT | P3: SCATTER_ADD | 计数变体 |
| 5 | NORMALIZE | P4: ELEWISE_DIV | `sum / count` |
| 6 | TRANSFORM | P1: MATMUL_DUAL | `W_l(agg) + W_r(x)` |
| 7 | ACTIVATE | P5: ELEWISE_ACT | ReLU |

### 1.3 执行流程图

```
     x[N,F]
        │
        ▼
   ┌──────────┐
   │ P2_GATHER │ x_j = x[edge_index[0]]
   └────┬─────┘
        │ x_j[E,F]
        │
        │ (Skip: MESSAGE is identity)
        │
        ▼
   ┌───────────────┐
   │ P3_SCATTER_ADD │ sum_agg = scatter_add(x_j)
   └───────┬───────┘
           │ sum_agg[N,F]
           │
           │           ┌───────────────────┐
           │           │ P3_SCATTER_ADD    │ count = count_neighbors
           │           │ (COUNT variant)   │
           │           └─────────┬─────────┘
           │                     │ count[N]
           ▼                     ▼
   ┌─────────────┐
   │ P4_ELEWISE  │ mean_agg = sum / count
   │ (DIV)       │
   └──────┬──────┘
          │ mean_agg[N,F]
          │                    x[N,F]
          ▼                        ▼
   ┌────────────────┐
   │ P1_MATMUL_DUAL │ out = W_l(agg) + W_r(x)
   └───────┬────────┘
           │ out[N,F']
           ▼
   ┌──────────────┐
   │ P5_ELEWISE   │ output = ReLU(out)
   │ (ACT)        │
   └──────┬───────┘
          │
          ▼
      output[N,F']
```

### 1.4 NPU 兼容性

| 原语 | NPU 兼容 | 原因 |
|------|---------|------|
| P2_GATHER | ✅ | 简单索引操作 |
| P3_SCATTER_ADD | ❌ | 动态 scatter 聚合 |
| P4_ELEWISE | ✅ | 逐元素除法 |
| P1_MATMUL | ✅ | 矩阵乘法 |
| P5_ELEWISE | ✅ | 逐元素激活 |

---

## 二、GAT → 原语映射

### 2.1 计算流程

```
GAT (单头注意力):
h_i' = σ( Σ_j α_ij · W·h_j )

其中:
α_ij = softmax_j( LeakyReLU(a^T · [W·h_i || W·h_j]) )
```

**关键区别**: GAT 的线性变换在 GATHER 之前！

### 2.2 原语映射

| 原 Stage | 原名称 | → 原语 | 说明 |
|----------|--------|--------|------|
| 1 | LINEAR | P1: MATMUL | `W @ x` (在 gather 之前!) |
| 2 | GATHER_BOTH | **P2: GATHER × 2** | `Wx[src]`, `Wx[dst]` |
| 3 | ATTENTION_SCORE | P6: GAT_EDGE_ATT | `LeakyReLU(a·[xi‖xj])` |
| 4 | ATTENTION_SOFTMAX | P7: GAT_SOFTMAX | Edge softmax |
| 5 | MESSAGE_WEIGHTED | P4: ELEWISE_MUL | `alpha * Wx_j` |
| 6 | REDUCE_SUM | P3: SCATTER_ADD | `scatter_add` |
| 7 | ACTIVATE | P5: ELEWISE_ACT | ELU |

### 2.3 关键优化：GATHER_BOTH 分解

```
原设计:
  GATHER_BOTH: 输出 (Wx_i, Wx_j) → 特殊实现，无法复用

新设计:
  2 × P2_GATHER:
    Wx_j = P2_Gather(Wx, edge_index[0])  // source
    Wx_i = P2_Gather(Wx, edge_index[1])  // target

  Cost(GATHER_BOTH) = 2 × Cost(P2_GATHER)  // 可直接从 SAGE 复用!
```

### 2.4 执行流程图

```
     x[N,F]
        │
        ▼
   ┌──────────┐
   │ P1_MATMUL │ Wx = W @ x
   └────┬─────┘
        │ Wx[N,F']
        ▼
   ┌───────────────────────────┐
   │ P2_GATHER × 2             │
   │   Wx_j = Wx[edge_index[0]]│ (source)
   │   Wx_i = Wx[edge_index[1]]│ (target)
   └───────────┬───────────────┘
               │ Wx_i[E,F'], Wx_j[E,F']
               ▼
   ┌─────────────────┐
   │ P6_GAT_EDGE_ATT │ e = LeakyReLU(a · [Wx_i || Wx_j])
   └────────┬────────┘
            │ e[E]
            ▼
   ┌───────────────┐
   │ P7_GAT_SOFTMAX │ alpha = softmax(e) per target
   └───────┬───────┘
           │ alpha[E]
           │                    Wx_j[E,F']
           │                        │
           ▼                        ▼
   ┌─────────────────┐
   │ P4_ELEWISE_MUL  │ msg = alpha * Wx_j
   └────────┬────────┘
            │ msg[E,F']
            ▼
   ┌───────────────┐
   │ P3_SCATTER_ADD │ h = scatter_add(msg)
   └───────┬───────┘
           │ h[N,F']
           ▼
   ┌─────────────┐
   │ P5_ELEWISE  │ output = ELU(h)
   │ (ACT)       │
   └──────┬──────┘
          │
          ▼
      output[N,F']
```

### 2.5 NPU 兼容性

| 原语 | NPU 兼容 | 原因 |
|------|---------|------|
| P1_MATMUL | ✅ | 矩阵乘法 |
| P2_GATHER | ✅ | 索引操作 |
| P6_GAT_EDGE_ATT | ✅ | 逐元素计算 |
| P7_GAT_SOFTMAX | ❌ | 含 scatter_max + scatter_add |
| P4_ELEWISE | ✅ | 逐元素乘法 |
| P3_SCATTER_ADD | ❌ | scatter 聚合 |
| P5_ELEWISE | ✅ | 逐元素激活 |

---

## 三、GCN → 原语映射

### 3.1 计算流程

```
GCN (对称归一化):
H' = σ( D̃^(-1/2) Ã D̃^(-1/2) H W )

边级别实现:
h_i' = σ( W · Σ_j (1/√(d_i·d_j)) · h_j )
```

### 3.2 原语映射

| 原 Stage | 原名称 | → 原语 | 说明 |
|----------|--------|--------|------|
| 1 | COMPUTE_NORM | **(Preprocess)** | 静态图缓存，不计入 Online Cost |
| 2 | GATHER | P2: GATHER | `x[edge_index[0]]` |
| 3 | MESSAGE | P4: ELEWISE_MUL | `norm * x_j` |
| 4 | REDUCE_SUM | P3: SCATTER_ADD | `scatter_add` |
| 5 | TRANSFORM | P1: MATMUL | `W @ agg + b` |
| 6 | ACTIVATE | P5: ELEWISE_ACT | ReLU |

### 3.3 关键优化：COMPUTE_NORM 作为预处理

```
原设计:
  COMPUTE_NORM 每次前向传播都执行 → 重复计算

新设计:
  静态图场景下:
    norm = 1/sqrt(deg_i * deg_j)
    只需计算一次并缓存!

  Online Cost 只包含:
    P2 → P4 → P3 → P1 → P5
```

### 3.4 执行流程图

```
    [Preprocessing - ONE TIME]
    edge_index
        │
        ▼
   ┌────────────────┐
   │ COMPUTE_NORM   │ norm = 1/sqrt(deg_i * deg_j)
   │ (Cached)       │
   └───────┬────────┘
           │ norm[E] (cached)
           │
    =======│======== Online Execution ========
           │
     x[N,F]│
        │  │
        ▼  │
   ┌──────────┐
   │ P2_GATHER │ x_j = x[edge_index[0]]
   └────┬─────┘
        │ x_j[E,F]
        │                    norm[E] (from cache)
        ▼                        │
   ┌─────────────────┐           │
   │ P4_ELEWISE_MUL  │ ◄─────────┘
   │                 │ msg = norm * x_j
   └────────┬────────┘
            │ msg[E,F]
            ▼
   ┌───────────────┐
   │ P3_SCATTER_ADD │ agg = scatter_add(msg)
   └───────┬───────┘
           │ agg[N,F]
           ▼
   ┌──────────┐
   │ P1_MATMUL │ out = W @ agg + b
   └────┬─────┘
        │ out[N,F']
        ▼
   ┌──────────────┐
   │ P5_ELEWISE   │ output = ReLU(out)
   │ (ACT)        │
   └──────┬───────┘
          │
          ▼
      output[N,F']
```

### 3.5 NPU 兼容性

| 原语 | NPU 兼容 | 原因 |
|------|---------|------|
| (Preprocess) | - | 不计入在线开销 |
| P2_GATHER | ✅ | 索引操作 |
| P4_ELEWISE | ✅ | 逐元素乘法 |
| P3_SCATTER_ADD | ❌ | scatter 聚合 |
| P1_MATMUL | ✅ | 矩阵乘法 |
| P5_ELEWISE | ✅ | 逐元素激活 |

---

## 四、三种模型原语使用对比

### 4.1 原语使用矩阵

| 原语 | SAGE | GAT | GCN | 总复用率 |
|------|------|-----|-----|----------|
| P1_MATMUL | ✅ (TRANSFORM) | ✅ (LINEAR) | ✅ (TRANSFORM) | 100% |
| P2_GATHER | ✅ (1×) | ✅ (2×) | ✅ (1×) | 100% |
| P3_SCATTER_ADD | ✅ (SUM+COUNT) | ✅ | ✅ | 100% |
| P4_ELEWISE | ✅ (DIV) | ✅ (MUL) | ✅ (MUL) | 100% |
| P5_ELEWISE | ✅ (ReLU) | ✅ (ELU) | ✅ (ReLU) | 100% |
| P6_GAT_EDGE_ATT | - | ✅ | - | 33% |
| P7_GAT_SOFTMAX | - | ✅ | - | 33% |

### 4.2 执行顺序对比

```
GraphSAGE:
  P2 → (skip) → P3 → P3_cnt → P4_div → P1_dual → P5

GAT:
  P1 → P2(×2) → P6 → P7 → P4_mul → P3 → P5

GCN:
  (preprocess) → P2 → P4_mul → P3 → P1 → P5
```

### 4.3 关键设计差异

| 特性 | GraphSAGE | GAT | GCN |
|------|-----------|-----|-----|
| 线性变换位置 | 聚合后 (P1 在末尾) | 聚合前 (P1 在开头) | 聚合后 (P1 在末尾) |
| GATHER 次数 | 1× | **2×** | 1× |
| 特有原语 | - | P6, P7 | - |
| 预处理 | 无 | 无 | COMPUTE_NORM |
| 归一化方式 | 节点级 (mean) | 注意力 (softmax) | 边级别 (对称) |

### 4.4 NPU 调度建议

| 模型 | NPU 可执行原语 | CPU/GPU 执行原语 |
|------|----------------|------------------|
| GraphSAGE | P2, P4, P1, P5 | **P3** |
| GAT | P1, P2, P6, P4, P5 | **P7, P3** |
| GCN | P2, P4, P1, P5 | **P3** |

---

## 五、Profiling 策略

### 5.1 仅需 Profile 的 7 个原语

```python
PROFILING_PRIMITIVES = [
    'P1_MATMUL',        # 矩阵乘法 - 所有模型使用
    'P2_GATHER',        # 索引选择 - 所有模型使用
    'P3_SCATTER_ADD',   # Scatter 聚合 - 所有模型使用
    'P4_ELEWISE_MUL',   # 逐元素乘法 - 所有模型使用
    'P5_ELEWISE_ACT',   # 逐元素激活 - 所有模型使用
    'P6_GAT_EDGE_ATT',  # GAT 注意力 - GAT only
    'P7_GAT_SOFTMAX',   # GAT softmax - GAT only
]
```

### 5.2 模型代价估算公式

```
Cost(SAGE) = Cost(P2) + 2×Cost(P3) + Cost(P4) + 2×Cost(P1) + Cost(P5)

Cost(GAT) = Cost(P1) + 2×Cost(P2) + Cost(P6) + Cost(P7) + Cost(P4) + Cost(P3) + Cost(P5)

Cost(GCN) = Cost(P2) + Cost(P4) + Cost(P3) + Cost(P1) + Cost(P5)
            [COMPUTE_NORM 不计入在线开销]
```

### 5.3 从 SAGE Profiling 复用到其他模型

| 模型 | 可直接复用的原语 | 需要额外 Profile | 复用率 |
|------|-----------------|------------------|--------|
| GAT | P1, P2, P3, P4, P5 | P6, P7 | 71% |
| GCN | P1, P2, P3, P4, P5 | - | **100%** |

---

## 六、文件结构

```
experiments/models/
├── primitives.py              # 7 个通用原语定义
├── model_sage_primitives.py   # SAGE 原语组合实现
├── model_gat_primitives.py    # GAT 原语组合实现
├── model_gcn_primitives.py    # GCN 原语组合实现
├── Model_sage.py              # SAGE 原始 7-stage 实现 (兼容)
├── Model_gat.py               # GAT 原始 7-stage 实现 (兼容)
├── Model_gcn.py               # GCN 原始 6-stage 实现 (兼容)
└── MODEL_DECOMPOSITION.md     # 本文档

profiling/
├── profile_stages.py          # 原始 stage-based profiling
└── profile_primitives.py      # 新的 primitive-based profiling
```

---

## 七、验证计划

1. **数值正确性**: 原语组合输出 vs PyG 原始实现
2. **Profiling 准确性**: 原语求和 vs 端到端测量
3. **NPU 调度**: 验证 scatter 原语正确跳过 NPU

---

*文档版本: v2.0 (原语化重构)*
*创建日期: 2026-01-21*
