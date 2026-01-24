# Reddit2 Partition & Pipeline Analysis

## 1. Executive Summary
After analyzing partitions from **K=1 to K=10**, the optimal configuration is:
# 🏆 Winner: K=10

- **Minimum Latency:** `6684.51 ms`
- **Performance Gain:** vs K=1 (Sequential): `4.77x` faster.
- **Reason:** Higher K reduces the per-subgraph size, allowing the GPU to process a larger percentage of edges (Adaptive Mode) instead of overflowing to the slow CPU. The pipeline overlap further hides latency.

## 2. Comparative Analysis (K=1 to K=10)
| K | Total Latency (ms) | Speedup (vs K=1) | Bottleneck Mode (Slowest Task) | Note |
|:---:|:---:|:---:|:---:|:---:|
| 1 | **31910** | 1.00x | Overflow | Sequential |
| 2 | **27383** | 1.17x | Overflow | Pipeline |
| 3 | **23313** | 1.37x | Overflow | Pipeline |
| 4 | **20511** | 1.56x | Overflow | Pipeline |
| 5 | **17281** | 1.85x | Overflow | Pipeline |
| 6 | **14274** | 2.24x | Overflow | Pipeline |
| 7 | **11347** | 2.81x | Overflow | Pipeline |
| 8 | **8983** | 3.55x | Overflow | Pipeline |
| 9 | **7552** | 4.23x | Overflow | Pipeline |
| 10 | **6685** | 4.77x | Overflow | Pipeline (Best) |

## 3. Deep Dive: Optimal Schedule (K=10)
### 3.1 Subgraph Estimations
| ID | Nodes | Edges (M) | Mode | CPU Ratio | S1 Time (ms) | S2 Time (ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 44322 | 1.47 | Adaptive | 0.0% | 432 | 108 |
| 4 | 53607 | 2.00 | Adaptive | 0.0% | 464 | 130 |
| 1 | 64601 | 2.03 | Adaptive | 0.0% | 500 | 156 |
| 6 | 69942 | 1.64 | Adaptive | 0.0% | 517 | 169 |
| 5 | 77088 | 1.71 | Adaptive | 0.0% | 541 | 185 |
| 0 | 94181 | 2.13 | Adaptive | 0.0% | 598 | 225 |
| 8 | 98212 | 2.18 | Adaptive | 0.0% | 612 | 235 |
| 3 | 107125 | 1.45 | Adaptive | 0.0% | 641 | 256 |
| 7 | 119905 | 1.46 | Adaptive | 0.0% | 684 | 286 |
| 9 | 86012 | 2.84 | Overflow | 22.4% | 1488 | 206 |

### 3.2 Pipeline Timeline
| Seq | ID | Stage 1 Interval | Stage 2 Interval | Bubble (Wait) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | **2** | 0 - 432 | 432 - 540 | 0 ms |
| 2 | **4** | 432 - 896 | 896 - 1027 | 356 ms |
| 3 | **1** | 896 - 1396 | 1396 - 1552 | 370 ms |
| 4 | **6** | 1396 - 1914 | 1914 - 2082 | 361 ms |
| 5 | **5** | 1914 - 2455 | 2455 - 2640 | 373 ms |
| 6 | **0** | 2455 - 3053 | 3053 - 3279 | 413 ms |
| 7 | **8** | 3053 - 3665 | 3665 - 3900 | 386 ms |
| 8 | **3** | 3665 - 4306 | 4306 - 4562 | 407 ms |
| 9 | **7** | 4306 - 4991 | 4991 - 5277 | 428 ms |
| 10 | **9** | 4991 - 6478 | 6478 - 6685 | 1202 ms |
