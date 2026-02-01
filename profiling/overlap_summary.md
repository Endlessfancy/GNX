# Profiling Overlap Summary

## Goal

Determine the minimal set of stages to profile such that all core aten operators across SAGE, GAT, and GCN are covered.

## Baseline: SAGE (7 Stages) — Profile All

| Stage | Name | Core aten op | Shape pattern |
|-------|------|-------------|---------------|
| 1 | Gather | `index` [N,F]→[E,F] | [1000,500] |
| 2 | Message | identity (no op) | — |
| 3 | ReduceSum | `scatter_add` [E,F]→[N,F] | [10000,500]→[1000,500] |
| 4 | ReduceCount | `scatter_add_` [E]→[N] (1D) | [10000]→[1000] |
| 5 | Normalize | `clamp` + `div` | [1000,500] |
| 6 | Transform | `addmm` (lin_l) + `mm` (lin_r) + `add` | [1000,500]×[500,256] |
| 7 | Activate | `relu` | [1000,256] |

## Supplementary: GAT Stage 3, 4, 5 — Profile These Three

These three stages form the GAT attention mechanism, which has no equivalent in SAGE.

| GAT Stage | Name | Core aten ops | Why SAGE doesn't cover |
|-----------|------|--------------|----------------------|
| **3** | **AttentionScore** | `mul`+`sum` (dot product) ×2, `add`, `leaky_relu` | SAGE has no attention; missing edge-domain dot product and leaky_relu |
| **4** | **AttentionSoftmax** | `scatter_reduce`(amax), `sub`, `exp`, `scatter_add`, `div` | SAGE has no per-node softmax; missing scatter_reduce(amax), exp, sub |
| **5** | **MessageWeighted** | `mul` [E,1]×[E,F] | SAGE message is identity; missing edge-domain element-wise mul |

## Not Required: GAT Stage 1, 2, 6, 7

| GAT Stage | Name | Already covered by |
|-----------|------|--------------------|
| 1 | Linear | SAGE Stage 6 `mm` (same matmul op) |
| 2 | GatherBoth | SAGE Stage 1 `index` (same gather, run twice) |
| 6 | ReduceSum | SAGE Stage 3 `scatter_add` (identical) |
| 7 | Activate (ELU) | SAGE Stage 7 `relu` (same element-wise activation pattern) |

## Not Required: GCN (All Stages Covered)

| GCN Stage | Name | Covered by |
|-----------|------|-----------|
| 1 | ComputeNorm | SAGE Stage 4 (`scatter_add_` 1D) + cacheable, only runs once |
| 2 | Gather | SAGE Stage 1 (`index`, identical) |
| 3 | Message | **GAT Stage 5** (`mul` [E,1]×[E,F], same op, scale by F) |
| 4 | ReduceSum | SAGE Stage 3 (`scatter_add`, identical) |
| 5 | Transform | SAGE Stage 6 (`addmm`, identical) |
| 6 | Activate | SAGE Stage 7 (`relu`, identical) |

## Summary (CPU — All Stages)

```
Total stages across 3 models:  SAGE(7) + GAT(7) + GCN(6) = 20 stages
Stages to profile:             SAGE(7) + GAT(3)           = 10 stages
Coverage:                      100%
```

---

## NPU Profiling (Scatter Stages Excluded)

NPU does not support scatter operations (`scatter_add`, `scatter_add_`, `scatter_reduce`).
These stages must stay on CPU and are excluded from NPU profiling.

### Scatter stages skipped on NPU

| Model | Skipped Stage | Scatter op |
|-------|--------------|------------|
| SAGE | Stage 3 ReduceSum | `scatter_add` [E,F]→[N,F] |
| SAGE | Stage 4 ReduceCount | `scatter_add_` [E]→[N] |
| GAT | Stage 4 AttentionSoftmax | `scatter_reduce`(amax) + `scatter_add` |
| GAT | Stage 6 ReduceSum | `scatter_add` [E,F]→[N,F] |
| GCN | Stage 1 ComputeNorm | `scatter_add_` (degree computation) |
| GCN | Stage 4 ReduceSum | `scatter_add` [E,F]→[N,F] |

### NPU-compatible stages and coverage

| NPU Stage | Core op | Covered by |
|-----------|---------|-----------|
| SAGE Stage 1 Gather | `index` [N,F]→[E,F] | **SAGE** |
| SAGE Stage 5 Normalize | `clamp` + `div` | **SAGE** |
| SAGE Stage 6 Transform | `addmm` + `mm` + `add` | **SAGE** |
| SAGE Stage 7 Activate | `relu` | **SAGE** |
| GAT Stage 1 Linear | `mm` | = SAGE Stage 6 `mm` |
| GAT Stage 2 GatherBoth | `index` ×2 | = SAGE Stage 1 `index` |
| GAT Stage 3 AttentionScore | `mul`+`sum`+`leaky_relu` | **GAT Stage 3 (supplementary)** |
| GAT Stage 5 MessageWeighted | `mul` [E,1]×[E,F] | **GAT Stage 5 (supplementary)** |
| GAT Stage 7 Activate | `elu` | ≈ SAGE Stage 7 `relu` |
| GCN Stage 2 Gather | `index` | = SAGE Stage 1 |
| GCN Stage 3 Message | `mul` [E,1]×[E,F] | = GAT Stage 5 |
| GCN Stage 5 Transform | `addmm` | = SAGE Stage 6 |
| GCN Stage 6 Activate | `relu` | = SAGE Stage 7 |

### Summary (NPU)

```
NPU-compatible stages across 3 models:  SAGE(4) + GAT(5) + GCN(4) = 13 stages
Stages to profile on NPU:               SAGE(4) + GAT(2)          =  6 stages
  - SAGE:  Stage 1 (Gather), Stage 5 (Normalize), Stage 6 (Transform), Stage 7 (Activate)
  - GAT:   Stage 3 (AttentionScore), Stage 5 (MessageWeighted)
Coverage:                                100%
```
