# Parallel Topological Sort using Kahn's Algorithm

A high-performance implementation of parallel topological sorting for large directed acyclic graphs (DAGs), achieving up to **2.25x speedup** over optimized serial implementations on complex graph structures.

## Overview

This project implements both serial and parallel versions of Kahn's algorithm for topological sorting, optimized for large-scale graphs with millions of nodes and edges. The parallel implementation uses OpenMP for thread-level parallelism and includes various optimizations like CSR graph representation, prefetching, and adaptive parallelization strategies.

## Key Features

- **Parallel Kahn's Algorithm** with adaptive frontier processing
- **Optimized Serial Baseline** for performance comparison
- **CSR (Compressed Sparse Row)** graph representation for cache efficiency
- **Memory prefetching** to reduce cache misses
- **Binary graph format** for fast I/O
- **Graph generator** with multiple DAG topologies for benchmarking
- **Thread-safe atomic operations** for concurrent indegree updates

## Performance Results

Performance varies significantly based on graph topology:

| Graph Type | Speedup | Best Use Case |
|------------|---------|---------------|
| **Easy (Sequential)** | 0.8x | Serial performs better due to low parallelism |
| **Narrow DAG** | 1.5-1.8x | Moderate parallelism, random structure |
| **Wide DAG** | **2.25x** | High parallelism, broad frontier |
| **Disjoint Clusters** | 2.0-2.2x | Independent subgraphs, excellent parallelism |

The parallel implementation excels on graphs with:
- Wide frontiers (many nodes processable simultaneously)
- Random or complex topology
- Multiple independent components

Serial implementation is preferable for:
- Sequential or nearly-sequential graphs
- Small graphs (< 100K nodes)
- Narrow frontiers with limited parallelism

## Repository Structure

```
.
├── graph_generator.cpp         # Generates test DAGs in binary format
├── serial_kahn.cpp             # Optimized serial baseline (includes I/O)
├── parallel_kahn.cpp           # Parallel implementation (includes I/O)
├── serial_kahn                 # Compiled serial binary
├── parallel_kahn               # Compiled parallel binary
├── graph_gen                   # Compiled graph generator
└── README.md                   # This file
```

**About VTune:**

The two cpp files can be used for research-grade profiling. They:
- Load the graph and convert to CSR format (setup phase)
- **Pause and wait** for you to attach VTune Profiler
- Run the pure algorithm 5 times 
- Ensure VTune captures **only** algorithm hotspots, with zero I/O "pollutants"

## Building

### Requirements

- **C++ Compiler** with C++11 support (GCC 7+ or Clang 6+ recommended)
- **OpenMP** support
- **x86-64 CPU** (for prefetching intrinsics)

### Compilation

```bash
# Graph generator
g++ -O3 -fopenmp -march=native graph_generator.cpp -o graph_gen

# Serial version (includes I/O - convenient for benchmarking)
g++ -O3 -march=native -g serial_kahn.cpp -o serial_topo

# Parallel version (includes I/O - convenient for benchmarking)
g++ -O3 -march=native -fopenmp -g parallel_kahn.cpp -o parallel_topo

```

## Usage

### 1. Generate Test Graphs

```bash
./graph_gen <type> <nodes> <edges_per_node> <output_file>
```

**Graph Types:**
- `easy`: Sequential DAG with predictable structure (tests worst-case for parallelism)
- `narrow`: Random DAG with narrow frontiers
- `wide`: Random DAG with 10% source nodes, wide frontiers (best for parallelism)
- `disjoint`: 8 independent clusters (excellent for parallel processing)

**Example:**
```bash
# Generate a 20M node wide DAG with 15 edges per node
./graph_gen wide 20000000 15 graph_wide.bin

# Generate smaller test graphs
./graph_gen narrow 5000000 10 graph_narrow.bin
./graph_gen disjoint 10000000 12 graph_disjoint.bin
```

### 2. Run Serial Baseline

```bash
./serial_kahn <graph_file.bin>
```

### 3. Run Parallel Version

```bash
# Set thread count (default: all available cores)
export OMP_NUM_THREADS=8
./parallel_kahn <graph_file.bin>

# Or inline
OMP_NUM_THREADS=16 ./parallel_kahn graph_wide.bin
```

### Sample Output

```
=================================================
PARALLEL TOPOLOGICAL SORT
=================================================
Input: graph_wide.bin
Threads: 8
=================================================

[1/3] Loading graph... Done (2.3451s)
  Loaded 20000000 nodes
[2/3] Converting to CSR... Done (1.2341s)
  Total edges: 299847238

[3/3] Running topological sort...
------------------------------------------------
  Run 1: 3.4521s (sorted 20000000 nodes)
  Run 2: 3.4102s (sorted 20000000 nodes)
  Run 3: 3.3987s (sorted 20000000 nodes)
  Run 4: 3.4156s (sorted 20000000 nodes)
  Run 5: 3.4234s (sorted 20000000 nodes)
------------------------------------------------
RESULTS:
  Best:    3.3987s
  Worst:   3.4521s
  Average: 3.4200s
  Std dev: 0.0267s
=================================================
```

## Algorithm Details

### Kahn's Algorithm

The implementation uses Kahn's algorithm for topological sorting:

1. **Initialize**: Compute indegree for all nodes
2. **Find sources**: Identify all nodes with indegree = 0
3. **Process frontier**: For each node in the current frontier:
   - Add to output ordering
   - Decrement indegrees of all neighbors
   - Add neighbors with indegree = 0 to next frontier
4. **Repeat**: Continue until frontier is empty

### Parallel Optimizations

**Adaptive Parallelization:**
- Frontiers ≥ 2048 nodes: Parallel processing with dynamic scheduling
- Frontiers < 2048 nodes: Serial processing to avoid overhead

**Memory Optimizations:**
- CSR format reduces memory footprint by ~40%
- Thread-local buffers reduce contention
- Atomic operations only where necessary

**Cache Optimizations:**
- Prefetching upcoming indegree values
- Static scheduling for initial parallel sections
- Dynamic scheduling (chunk size 32) for load balancing

### Memory Format

Binary graph format:
```
[int: num_nodes]
For each node u (0 to n-1):
  [int: degree]
  [int[degree]: neighbor_ids]
```

This format provides:
- Fast loading (2-3 seconds for 20M node graphs)
- Compact storage (typically 30-40% smaller than text format)
- Direct memory mapping capability


```bash
# This will profile everything: loading, conversion, AND algorithm
vtune -collect hotspots -- env OMP_NUM_THREADS=8 ./parallel_kahn graph_wide.bin
```

### Thread Count

Test different thread counts to find optimal performance for your hardware:

```bash
for t in 1 2 4 8 16; do
    echo "Testing with $t threads:"
    OMP_NUM_THREADS=$t ./parallel_kahn graph_wide.bin
done
```

### CPU Affinity

For NUMA systems, pin threads to specific cores:

```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=close
OMP_NUM_THREADS=8 ./parallel_kahn graph_wide.bin
```
## Implementation Notes

### Why CSR Format?

Compressed Sparse Row (CSR) format provides:
- Contiguous memory layout for better cache utilization
- Reduced pointer chasing compared to adjacency lists
- ~40% memory reduction for typical sparse graphs
- Better vectorization potential

### Atomic vs Mutex

The implementation uses `std::atomic<int>` for indegree updates because:
- Lightweight operations (single instruction on x86-64)
- No kernel involvement unlike mutexes
- Memory ordering can be relaxed (`memory_order_relaxed`) since we only care about the final count
- Scales better to many threads

### Prefetching Strategy

Manual prefetching hints (`_mm_prefetch`) with 8-element lookahead:
- Hides memory latency for sparse access patterns
- Distance tuned empirically for typical L1/L2 cache latencies
- Particularly effective for random graph topologies

## Limitations

- **DAG Assumption**: Input must be a directed acyclic graph (no cycle detection)
- **Memory Requirements**: Entire graph must fit in RAM (CSR format: ~8 bytes per edge + 4 bytes per node)
- **x86-64 Only**: Uses SSE prefetch intrinsics (portable version would need fallback)
- **No GPU Support**: CPU-only implementation

## Future Improvements

Potential enhancements for even better performance:
- GPU acceleration using CUDA or OpenCL
- NUMA-aware memory allocation
- Work-stealing scheduler for better load balancing
- Lock-free queue for frontier management
- SIMD vectorization for indegree updates

## License

This implementation is provided as-is for educational and research purposes.

## References

- Kahn, A. B. (1962). "Topological sorting of large networks". Communications of the ACM.
- Pearce, D. J., & Kelly, P. H. (2007). "A dynamic topological sort algorithm for directed acyclic graphs". ACM Journal of Experimental Algorithmics.
- Hong, S., et al. (2011). "Green-Marl: A DSL for Easy and Efficient Graph Analysis". ASPLOS.

## Contributing

Contributions welcome! Areas of interest:
- Additional graph generators (scale-free, small-world, etc.)
- GPU implementation
- Verification suite
- Performance comparisons with other parallel sorting approaches
