\noindent \textbf{Subgraph Partition Candidate Generation.}
Instead of searching for arbitrary graph cuts, we generate a discrete set of \textit{subgraph partition candidates}, each corresponding to an optimal topological cut for a specific integer partition count.
We first determine the feasible range for the number of subgraphs, $[K_{min}, K_{max}]$. 
The lower bound $K_{min}$ is strictly enforced by the memory capacity of the most resource-constrained PU, while the upper bound $K_{max}$ is constrained by a pre-defined \textit{halo node ratio limit} to prevent excessive redundancy.
Within this range, for every integer $k \in [K_{min}, K_{max}]$, we employ a METIS-based min-cut algorithm to instantiate a single optimal partition scheme that minimizes the halo node count for that specific $k$. 
This step effectively converts the unbounded partitioning problem into a finite list of high-quality subgraph partition candidates based purely on graph topology constraints.