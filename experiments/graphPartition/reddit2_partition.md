# Reddit2 Dataset METIS Partition Analysis

## Dataset Information

| Property | Value |
|----------|-------|
| Dataset | Reddit2 |
| Nodes | 232,965 |
| Edges | 23,213,838 |
| Features | 602 |
| Average Degree | ~199.3 (highly connected) |

## Partition Constraints

| Constraint | Value |
|------------|-------|
| Max nodes per subgraph | 100,000 |
| Halo ratio limit | 30% |
| K_min (based on size) | 3 |
| K_max (tested) | 20 |

## Summary

**Result: No valid partitions found for K=1 to K=20**

- **K=1**: 0% halo ratio but 232,965 nodes per subgraph (exceeds 100k limit)
- **K=2**: 51.7% halo ratio and 116k nodes per subgraph (exceeds both limits)
- **K=3-20**: All exceed 30% halo ratio limit (86% to 413%)

Reddit2's extremely high connectivity (~23M edges for ~233k nodes, avg degree ~199) makes it impossible to satisfy both constraints simultaneously.

## Partition Results Overview

| K | Valid | Node Range | Max Halo Ratio | Edge Cut | Overall Halo Ratio | Notes |
|---|-------|------------|----------------|----------|-------------------|-------|
| 1 | No | 232,965 | 0.0% | 0 | 0.0% | Exceeds 100k node limit |
| 2 | No | 116,482 - 116,483 | 55.0% | 664,953 | 51.7% | Exceeds 100k node limit |
| 3 | No | 77,654 - 77,656 | 108.8% | 1,138,976 | 86.0% | Exceeds 30% halo limit |
| 4 | No | 58,241 - 58,242 | 164.3% | 1,190,939 | 113.8% | Exceeds 30% halo limit |
| 5 | No | 46,592 - 46,594 | 211.0% | 1,356,615 | 133.9% | Exceeds 30% halo limit |
| 6 | No | 38,827 - 38,829 | 252.5% | 1,419,726 | 151.9% | Exceeds 30% halo limit |
| 7 | No | 33,280 - 33,281 | 325.4% | 1,521,844 | 166.0% | Exceeds 30% halo limit |
| 8 | No | 29,120 - 29,122 | 374.8% | 1,536,292 | 175.2% | Exceeds 30% halo limit |
| 9 | No | 25,129 - 26,662 | 344.3% | 1,734,194 | 207.6% | Exceeds 30% halo limit |
| 10 | No | 22,616 - 23,996 | 399.7% | 2,151,035 | 249.8% | Exceeds 30% halo limit |
| 11 | No | 20,512 - 21,815 | 425.9% | 2,051,687 | 247.4% | Exceeds 30% halo limit |
| 12 | No | 18,683 - 19,997 | 396.7% | 2,264,911 | 261.1% | Exceeds 30% halo limit |
| 13 | No | 17,393 - 18,458 | 570.5% | 2,271,926 | 261.8% | Exceeds 30% halo limit |
| 14 | No | 16,062 - 17,140 | 523.8% | 2,414,490 | 286.5% | Exceeds 30% halo limit |
| 15 | No | 14,917 - 15,997 | 507.1% | 2,421,009 | 297.1% | Exceeds 30% halo limit |
| 16 | No | 14,112 - 14,997 | 606.3% | 2,652,089 | 333.8% | Exceeds 30% halo limit |
| 17 | No | 13,221 - 14,115 | 638.2% | 2,649,714 | 339.2% | Exceeds 30% halo limit |
| 18 | No | 12,564 - 13,331 | 707.7% | 2,921,754 | 379.5% | Exceeds 30% halo limit |
| 19 | No | 11,903 - 12,629 | 1010.3% | 2,963,326 | 379.9% | Exceeds 30% halo limit |
| 20 | No | 11,308 - 11,998 | 848.0% | 3,202,334 | 413.5% | Exceeds 30% halo limit |

## Detailed Subgraph Statistics (K=1 to K=10)

### K=1 (No Partitioning)

| ID | Owned | Halo | Total Nodes | Internal Edges | Boundary Edges | Halo% |
|----|-------|------|-------------|----------------|----------------|-------|
| 0 | 232,965 | 0 | 232,965 | 23,213,838 | 0 | 0.0% |

- **Edge cut**: 0 | **Overall halo ratio**: 0.0% | **Status**: ❌ Exceeds 100k node limit

---

### K=2

| ID | Owned | Halo | Total Nodes | Internal Edges | Boundary Edges | Halo% |
|----|-------|------|-------------|----------------|----------------|-------|
| 0 | 116,482 | 56,273 | 172,755 | 9,255,308 | 664,953 | 48.3% |
| 1 | 116,483 | 64,097 | 180,580 | 12,628,624 | 664,953 | 55.0% |

- **Edge cut**: 664,953 | **Overall halo ratio**: 51.7% | **Status**: ❌ Exceeds 100k node limit + 30% halo limit

---

### K=3

| ID | Owned | Halo | Total Nodes | Internal Edges | Boundary Edges | Halo% |
|----|-------|------|-------------|----------------|----------------|-------|
| 0 | 77,655 | 47,883 | 125,538 | 6,471,810 | 413,548 | 61.7% |
| 1 | 77,656 | 84,507 | 162,163 | 6,886,728 | 989,176 | 108.8% |
| 2 | 77,654 | 67,984 | 145,638 | 7,577,348 | 875,228 | 87.5% |

- **Edge cut**: 1,138,976 | **Overall halo ratio**: 86.0% | **Status**: ❌ Exceeds 30% halo limit

---

### K=4

| ID | Owned | Halo | Total Nodes | Internal Edges | Boundary Edges | Halo% |
|----|-------|------|-------------|----------------|----------------|-------|
| 0 | 58,241 | 61,558 | 119,799 | 4,897,770 | 517,883 | 105.7% |
| 1 | 58,241 | 47,818 | 106,059 | 4,169,684 | 334,924 | 82.1% |
| 2 | 58,241 | 60,119 | 118,360 | 4,833,152 | 588,154 | 103.2% |
| 3 | 58,242 | 95,709 | 153,951 | 6,931,354 | 940,917 | 164.3% |

- **Edge cut**: 1,190,939 | **Overall halo ratio**: 113.8% | **Status**: ❌ Exceeds 30% halo limit

---

### K=5

| ID | Owned | Halo | Total Nodes | Internal Edges | Boundary Edges | Halo% |
|----|-------|------|-------------|----------------|----------------|-------|
| 0 | 46,593 | 49,704 | 96,297 | 3,378,796 | 319,264 | 106.7% |
| 1 | 46,593 | 44,890 | 91,483 | 4,470,720 | 348,672 | 96.3% |
| 2 | 46,593 | 44,904 | 91,497 | 3,321,954 | 324,318 | 96.4% |
| 3 | 46,594 | 98,301 | 144,895 | 5,840,834 | 965,894 | 211.0% |
| 4 | 46,592 | 74,245 | 120,837 | 3,488,304 | 755,082 | 159.4% |

- **Edge cut**: 1,356,615 | **Overall halo ratio**: 133.9% | **Status**: ❌ Exceeds 30% halo limit

---

### K=6

| ID | Owned | Halo | Total Nodes | Internal Edges | Boundary Edges | Halo% |
|----|-------|------|-------------|----------------|----------------|-------|
| 0 | 38,828 | 37,206 | 76,034 | 2,309,272 | 181,591 | 95.8% |
| 1 | 38,827 | 52,009 | 90,836 | 3,284,734 | 332,067 | 134.0% |
| 2 | 38,827 | 50,904 | 89,731 | 3,430,732 | 381,865 | 131.1% |
| 3 | 38,827 | 42,489 | 81,316 | 3,135,686 | 321,235 | 109.4% |
| 4 | 38,829 | 98,038 | 136,867 | 3,069,920 | 843,639 | 252.5% |
| 5 | 38,827 | 73,237 | 112,064 | 5,144,042 | 779,055 | 188.6% |

- **Edge cut**: 1,419,726 | **Overall halo ratio**: 151.9% | **Status**: ❌ Exceeds 30% halo limit

---

### K=7

| ID | Owned | Halo | Total Nodes | Internal Edges | Boundary Edges | Halo% |
|----|-------|------|-------------|----------------|----------------|-------|
| 0 | 33,280 | 49,284 | 82,564 | 3,116,138 | 380,436 | 148.1% |
| 1 | 33,281 | 35,178 | 68,459 | 2,069,448 | 154,109 | 105.7% |
| 2 | 33,281 | 39,706 | 72,987 | 2,978,622 | 228,741 | 119.3% |
| 3 | 33,281 | 60,687 | 93,968 | 3,896,074 | 667,249 | 182.3% |
| 4 | 33,281 | 108,288 | 141,569 | 3,260,868 | 1,010,307 | 325.4% |
| 5 | 33,280 | 38,481 | 71,761 | 2,585,668 | 234,042 | 115.6% |
| 6 | 33,281 | 55,085 | 88,366 | 2,263,332 | 368,804 | 165.5% |

- **Edge cut**: 1,521,844 | **Overall halo ratio**: 166.0% | **Status**: ❌ Exceeds 30% halo limit

---

### K=8

| ID | Owned | Halo | Total Nodes | Internal Edges | Boundary Edges | Halo% |
|----|-------|------|-------------|----------------|----------------|-------|
| 0 | 29,120 | 56,824 | 85,944 | 2,311,058 | 365,838 | 195.1% |
| 1 | 29,121 | 38,306 | 67,427 | 2,519,868 | 218,889 | 131.5% |
| 2 | 29,120 | 29,355 | 58,475 | 1,709,128 | 121,871 | 100.8% |
| 3 | 29,121 | 39,346 | 68,467 | 2,421,854 | 251,755 | 135.1% |
| 4 | 29,120 | 30,108 | 59,228 | 2,422,320 | 191,661 | 103.4% |
| 5 | 29,121 | 51,087 | 80,208 | 2,216,248 | 336,818 | 175.4% |
| 6 | 29,122 | 109,158 | 138,280 | 2,746,616 | 977,575 | 374.8% |
| 7 | 29,120 | 54,006 | 83,126 | 3,794,162 | 608,177 | 185.5% |

- **Edge cut**: 1,536,292 | **Overall halo ratio**: 175.2% | **Status**: ❌ Exceeds 30% halo limit

---

### K=9

| ID | Owned | Halo | Total Nodes | Internal Edges | Boundary Edges | Halo% |
|----|-------|------|-------------|----------------|----------------|-------|
| 0 | 25,129 | 69,804 | 94,933 | 2,184,874 | 359,480 | 277.8% |
| 1 | 26,661 | 38,165 | 64,826 | 2,384,938 | 246,633 | 143.1% |
| 2 | 25,705 | 44,514 | 70,219 | 1,638,828 | 202,472 | 173.2% |
| 3 | 25,331 | 25,282 | 50,613 | 1,555,170 | 89,316 | 99.8% |
| 4 | 26,662 | 27,296 | 53,958 | 2,332,408 | 206,249 | 102.4% |
| 5 | 25,257 | 47,585 | 72,842 | 1,781,822 | 257,073 | 188.4% |
| 6 | 26,662 | 91,795 | 118,457 | 2,108,558 | 726,867 | 344.3% |
| 7 | 25,230 | 52,851 | 78,081 | 3,485,394 | 588,109 | 209.5% |
| 8 | 26,328 | 86,292 | 112,620 | 2,273,458 | 792,189 | 327.8% |

- **Edge cut**: 1,734,194 | **Overall halo ratio**: 207.6% | **Status**: ❌ Exceeds 30% halo limit

---

### K=10

| ID | Owned | Halo | Total Nodes | Internal Edges | Boundary Edges | Halo% |
|----|-------|------|-------------|----------------|----------------|-------|
| 0 | 22,616 | 71,565 | 94,181 | 2,128,304 | 370,652 | 316.4% |
| 1 | 23,524 | 41,077 | 64,601 | 2,033,366 | 300,675 | 174.6% |
| 2 | 23,874 | 20,448 | 44,322 | 1,469,348 | 83,189 | 85.6% |
| 3 | 22,617 | 84,508 | 107,125 | 1,453,518 | 459,670 | 373.6% |
| 4 | 23,996 | 29,611 | 53,607 | 1,998,070 | 341,261 | 123.4% |
| 5 | 23,995 | 53,093 | 77,088 | 1,714,838 | 326,515 | 221.3% |
| 6 | 23,115 | 46,827 | 69,942 | 1,640,958 | 236,737 | 202.6% |
| 7 | 23,996 | 95,909 | 119,905 | 1,456,030 | 790,193 | 399.7% |
| 8 | 22,616 | 75,596 | 98,212 | 2,181,642 | 754,678 | 334.3% |
| 9 | 22,616 | 63,396 | 86,012 | 2,835,694 | 638,500 | 280.3% |

- **Edge cut**: 2,151,035 | **Overall halo ratio**: 249.8% | **Status**: ❌ Exceeds 30% halo limit

---

## Trend Analysis

As K increases:
1. **Halo ratio increases**: More partitions = more boundary edges = more halo nodes
2. **Edge cut increases**: From 1.1M (K=3) to 3.2M (K=20)
3. **Partition balance**: METIS maintains good balance (node ranges are tight)

### Comparison with Flickr Dataset

| Metric | Reddit2 | Flickr |
|--------|---------|--------|
| Nodes | 232,965 | 89,250 |
| Edges | 23,213,838 | 899,756 |
| Avg Degree | ~199 | ~20 |
| Valid Partitions | None | K=1 only |
| K=3 Halo Ratio | 108.8% | 112.4% |

Both datasets have high connectivity, making low-halo partitioning extremely challenging.

## Conclusions

1. **Reddit2 is not partitionable under current constraints**: The 30% halo ratio limit cannot be achieved for any K value from 3 to 20.

2. **Root cause**: Reddit2's extremely high edge density (avg degree ~199) means that most nodes have neighbors in multiple partitions, creating large halo sets.

3. **Fundamental trade-off**:
   - Fewer partitions (lower K) = larger subgraphs but lower halo ratio
   - More partitions (higher K) = smaller subgraphs but higher halo ratio
   - With Reddit2's connectivity, even K=3 exceeds constraints

## Recommendations

### Option 1: Relax Halo Ratio Constraint
If halo nodes are acceptable, consider:
- **K=3 with ~86% halo ratio**: Each subgraph has ~78k owned nodes + 50-85k halo nodes
- **K=4 with ~114% halo ratio**: More parallelism but higher memory overhead

### Option 2: Increase Max Subgraph Size
- **K=2 with ~117k nodes**: 51.7% halo ratio (much better than K=3's 86%)
- Allow larger subgraphs (e.g., 120k-150k nodes) to enable K=2
- Trade-off: Larger memory per partition but significantly lower halo overhead

### Option 3: Alternative Partitioning Strategies
- **Vertex-cut partitioning**: Instead of edge-cut (METIS default), assign edges to partitions and replicate vertices
- **Streaming partitioning**: For very large graphs
- **Community-based partitioning**: Leverage Reddit2's community structure

### Option 4: Single-GPU Processing
- With ~233k nodes and 23M edges, Reddit2 may fit on a single high-memory GPU
- Avoid partitioning overhead entirely

## Output Files

- `partition_reddit2.py` - Partition experiment code
- `reddit2_partition_results.json` - Detailed JSON results
- `reddit2_partition.md` - This analysis report

---
*Generated by METIS partition analysis on Reddit2 dataset*
