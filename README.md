# Parallel Computer Systems Projects

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-red)](https://developer.nvidia.com/cuda-toolkit)
[![OpenGL](https://img.shields.io/badge/OpenGL-3.3-blue)](https://www.opengl.org/)
[![OpenMP](https://img.shields.io/badge/OpenMP-4.5-orange)](https://www.openmp.org/)

Two high-performance parallel computing implementations showcasing different parallelization strategies: GPU acceleration with CUDA and multi-core CPU parallelism with OpenMP.

---

## 📂 Project 1: GPU Heat Diffusion Simulation 🔥

A real-time, interactive simulation of 2D heat transfer using the Finite Difference Method (FDM), accelerated by **CUDA** and visualized via **OpenGL interoperability**.

### Overview

This project solves the time-dependent heat equation on high-resolution grids:

$$\frac{\partial T}{\partial t} = \alpha \nabla^2 T$$

Instead of copying GPU data to the CPU for rendering, this implementation uses **CUDA–OpenGL Interop**, mapping the CUDA temperature buffer directly to an OpenGL texture. This eliminates PCIe bottlenecks and enables **real-time visualization at high resolutions**.

### Key Features

- **Real-Time Visualization:** Renders directly from GPU memory with zero CPU overhead for display
- **Interactive Scenarios:** Mouse-driven heat injection, corner radiators, central heating, and hot wall presets
- **Performance Metrics:** Live tracking of FPS, Min/Max/Avg temperature, and diffusion rate
- **Dual Kernel Implementations:** Global memory (optimized) and shared memory (research comparison)
- **CUDA-OpenGL Zero-Copy:** Direct rendering without intermediate host transfers

### Performance Results

Tested on **NVIDIA RTX 3060 (Ampere, Compute Capability 8.6)** with CUDA 12.4.

| Resolution | Total Pixels | Bandwidth Util. | Global Loads | FPS | Speedup vs CPU |
|------------|--------------|-----------------|--------------|-----|----------------|
| 800×600 | 480K | 60.2% | 357K | 79.5 | 4.2× |
| 2048×1536 | 3.15M | **84.4%** | 2,352K | 28.3 | **42.7×** |

**Key Findings:**
- **Memory Bandwidth Saturation:** Achieved 84% of peak memory bandwidth, confirming near-optimal performance for this memory-bound workload
- **Hardware vs Manual Optimization:** RTX 3060's 128KB L1 cache per SM automatically optimizes neighbor accesses. Attempted shared memory tiling reduced global memory traffic by 65% but decreased overall performance by 27% due to 75,800 bank conflicts per kernel
- **Roofline Model Validation:** Measured performance (252 GFLOP/s at 84% BW) matches theoretical prediction (300 GFLOP/s peak) with arithmetic intensity of 0.83 FLOP/byte

### Directory Structure

```
GPU/
├── main.cu              # Entry point, OpenGL setup, render loop
├── heat_kernel.cu       # CUDA kernels (global & shared memory)
├── heat_kernel.cuh      # Kernel declarations
├── render_kernel.cu     # Visualization kernels
├── render_kernel.cuh    # Render kernel declarations
└── heatdissipation.cpp  # CPU baseline implementation
```

### Build Instructions

#### Requirements

- NVIDIA CUDA Toolkit 12.x or 11.x
- OpenGL 3.3+
- GLFW 3.x, GLEW 2.x libraries
- Visual Studio 2022 (Windows) or GCC 7+ (Linux)

#### 🚀 Pre-built Releases (Quick Start)
If you do not wish to compile the project manually, you can download the standalone executable:
1. Go to the [Releases](https://github.com/AnesCenanovic/Parallel-Computer-Systems/releases) page.
2. Download the ZIP file containing the `.exe` and `.dll` files.
3. **Important:** Keep `heat_sim.exe`, `glew32.dll`, and `glfw3.dll` in the same folder for the application to launch.


#### Windows (Visual Studio 2022)

1. Open the `.sln` file
2. Set `main.cu` and `heat_kernel.cu` to **Item Type: CUDA C/C++**
3. **Project Properties → Configuration: All Configurations**
   - **CUDA C/C++ → Device → Code Generation:** `compute_86,sm_86` (for RTX 30xx)
   - **C/C++ → General → Additional Include Directories:** Add GLFW/GLEW include paths
   - **Linker → General → Additional Library Directories:** Add GLFW/GLEW lib paths
   - **Linker → Input → Additional Dependencies:** Add:
     ```
     glfw3.lib
     glew32.lib
     opengl32.lib
     cudart_static.lib
     ```
4. Copy `glew32.dll` to output directory (`x64/Release/`)
5. Build in **Release** mode for accurate profiling

#### Linux (Command Line)

```bash
nvcc -O3 -arch=sm_86 -o heat_sim \
  main.cu heat_kernel.cu \
  -lGL -lglfw -lGLEW \
  -I/usr/include \
  -L/usr/lib/x86_64-linux-gnu
```

### Controls

| Key | Action |
|-----|--------|
| **Space** | Pause / Resume simulation |
| **R** | Reset temperature grid to 0 |
| **1** | Interactive Mode (Click to add heat) |
| **2** | Corner Radiator Mode |
| **3** | Central Heating Mode |
| **4** | Hot Walls Mode |
| **5** | Multiple Heat Sources Mode |
| **[** / **]** | Decrease / Increase heat source radius |
| **-** / **+** | Decrease / Increase diffusion rate (α) |
| **ESC** | Exit |

### Implementation Details

#### Finite Difference Stencil

The simulation uses a 5-point stencil. For each cell $(i, j)$, the new temperature is:

$$T_{\text{new}} = T_{\text{center}} + \alpha \left( T_{\text{up}} + T_{\text{down}} + T_{\text{left}} + T_{\text{right}} - 4T_{\text{center}} \right)$$

Where $\alpha$ is the thermal diffusivity coefficient (user-adjustable via keyboard).

#### OpenGL–CUDA Interoperability

To avoid copying GPU memory to CPU before rendering:

```cuda
// 1. Map OpenGL buffer to CUDA
cudaGraphicsMapResources(1, &cuda_pbo, 0);
cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &size, cuda_pbo);

// 2. Kernel writes colors directly to d_pixels
visualizeHeat<<<grid, block>>>(d_temp, d_pixels, WIDTH, HEIGHT, MIN_TEMP, MAX_TEMP);

// 3. Unmap and render
cudaGraphicsUnmapResources(1, &cuda_pbo, 0);
```

This allows zero-copy rendering, keeping all computations and visualization on the GPU.

#### Memory Architecture

**Global Memory Kernel (Recommended):**
- Direct memory access with coalesced reads
- L1 cache automatically optimizes neighbor accesses
- 84% memory bandwidth utilization at 2048×1536

**Shared Memory Kernel (Research):**
- 18×18 tiles (16×16 threads + 2-cell halo)
- Reduces global loads by 65% (2.35M → 831K sectors)
- Bank conflicts (75,800 per launch) negate benefits
- Demonstrates that hardware optimizations can outperform manual tiling

---

## 📂 Project 2: Parallel Topological Sort 📊

A high-performance implementation of parallel topological sorting using Kahn's algorithm for large directed acyclic graphs (DAGs), achieving up to **2.25× speedup** over optimized serial implementations.

### Overview

This project implements both serial and parallel versions of Kahn's algorithm for topological sorting, optimized for large-scale graphs with millions of nodes and edges. The parallel implementation uses **OpenMP** for thread-level parallelism with adaptive frontier processing.

### Key Features

- **Parallel Kahn's Algorithm** with adaptive frontier processing
- **Optimized Serial Baseline** for performance comparison
- **CSR (Compressed Sparse Row)** graph representation for cache efficiency
- **Memory prefetching** to reduce cache misses
- **Binary graph format** for fast I/O
- **Graph generator** with multiple DAG topologies for benchmarking
- **Thread-safe atomic operations** for concurrent indegree updates

### Performance Results

Performance varies significantly based on graph topology:

| Graph Type | Speedup | Characteristics |
|------------|---------|-----------------|
| **Easy (Sequential)** | 0.8× | Low parallelism, serial performs better |
| **Narrow DAG** | 1.5-1.8× | Moderate parallelism, random structure |
| **Wide DAG** | **2.25×** | High parallelism, broad frontier |
| **Disjoint Clusters** | 2.0-2.2× | Independent subgraphs, excellent parallelism |

The parallel implementation excels on graphs with wide frontiers (many nodes processable simultaneously), random or complex topology, and multiple independent components.

### Directory Structure

```
TopologicalSort/
├── graph_generator.cpp    # Generates test DAGs in binary format
├── serial_kahn.cpp        # Optimized serial baseline (includes I/O)
├── parallel_kahn.cpp      # Parallel implementation (includes I/O)
├── serial_kahn            # Compiled serial binary
├── parallel_kahn          # Compiled parallel binary
└── graph_gen              # Compiled graph generator
```

### Build Instructions

#### Requirements

- C++ Compiler with C++11 support (GCC 7+ or Clang 6+ recommended)
- OpenMP support
- x86-64 CPU (for prefetching intrinsics)

#### Compilation

```bash
# Graph generator
g++ -O3 -fopenmp -march=native graph_generator.cpp -o graph_gen

# Serial version (includes I/O)
g++ -O3 -march=native -g serial_kahn.cpp -o serial_topo

# Parallel version (includes I/O)
g++ -O3 -march=native -fopenmp -g parallel_kahn.cpp -o parallel_topo
```

### Usage

#### 1. Generate Test Graphs

```bash
./graph_gen <type> <nodes> <edges_per_node> <output_file>
```

**Graph Types:**
- `easy`: Sequential DAG (worst-case for parallelism)
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

#### 2. Run Serial Baseline

```bash
./serial_kahn <graph_file.bin>
```

#### 3. Run Parallel Version

```bash
# Set thread count (default: all available cores)
export OMP_NUM_THREADS=8
./parallel_kahn <graph_file.bin>

# Or inline
OMP_NUM_THREADS=16 ./parallel_kahn graph_wide.bin
```

### Algorithm Details

#### Kahn's Algorithm

1. **Initialize:** Compute indegree for all nodes
2. **Find sources:** Identify all nodes with indegree = 0
3. **Process frontier:** For each node in the current frontier:
   - Add to output ordering
   - Decrement indegrees of all neighbors
   - Add neighbors with indegree = 0 to next frontier
4. **Repeat:** Continue until frontier is empty

#### Parallel Optimizations

**Adaptive Parallelization:**
- Frontiers ≥ 2048 nodes: Parallel processing with dynamic scheduling
- Frontiers < 2048 nodes: Serial processing to avoid overhead

**Memory Optimizations:**
- CSR format reduces memory footprint by ~40%
- Thread-local buffers reduce contention
- Atomic operations only where necessary

**Cache Optimizations:**
- Prefetching upcoming indegree values (`_mm_prefetch` with 8-element lookahead)
- Static scheduling for initial parallel sections
- Dynamic scheduling (chunk size 32) for load balancing

---

## 📊 Comparative Analysis

| Aspect | GPU Heat Diffusion | Parallel Topological Sort |
|--------|-------------------|---------------------------|
| **Parallelism Type** | Massive data parallelism (3,584 CUDA cores) | Thread-level parallelism (OpenMP) |
| **Bottleneck** | Memory bandwidth (84% utilization) | Topology-dependent (adaptive) |
| **Speedup** | 42.7× on 2048×1536 grid | 2.25× on wide DAGs |
| **Scalability** | Excellent (6.25× problem → ~4× faster) | Variable (depends on frontier width) |
| **Memory Pattern** | Regular, coalesced | Irregular, sparse |
| **Best Use Case** | Large, regular grids | Complex, wide DAGs |

**Key Insight:** GPU excels at regular, memory-bound problems with massive parallelism, while OpenMP is effective for irregular, compute-bound tasks with moderate parallelism.

---

## 🧪 Profiling and Analysis

### GPU Project (NVIDIA Nsight Compute)

```bash
# Profile memory bandwidth utilization
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --launch-skip 50 --launch-count 10 \
    --kernel-name heatStep heat_sim.exe

# Full analysis with export
ncu --set full --export profile --kernel-name heatStep heat_sim.exe
```

**Open in GUI:** `nsight-compute profile.ncu-rep`

### CPU Project (Intel VTune / perf)

```bash
# VTune hotspots analysis
vtune -collect hotspots -- env OMP_NUM_THREADS=8 ./parallel_kahn graph_wide.bin

# Linux perf
perf record -g ./parallel_kahn graph_wide.bin
perf report
```

---

## 📝 Publications

This work is documented in an IEEE-format paper:

**"Parallel Heat Diffusion Simulation on the GPU Using CUDA"**
- Authors: Anes Ćenanović, Faris Crnčalo, Esma Dizdarević
- Institution: University of Sarajevo, Faculty of Electrical Engineering
- [View Paper](docs/IEEE_Paper.pdf) | [LaTeX Source](docs/IEEE_Paper.tex)

Key contributions:
- Empirical validation of Roofline performance model (84% bandwidth vs 85% theoretical)
- Analysis of shared memory trade-offs on modern GPU architectures
- Demonstration that hardware optimizations can outperform manual tiling for regular stencils

---

## 🔧 Future Work

### GPU Heat Diffusion
- Extend to 3D simulation with volume rendering
- Implement higher-order stencils (4th/6th order accuracy)
- Multi-GPU support via domain decomposition
- Adaptive time-stepping based on CFL condition

### Topological Sort
- GPU acceleration using CUDA
- Work-stealing scheduler for better load balancing
- NUMA-aware memory allocation
- Lock-free queue for frontier management

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NVIDIA** for CUDA Toolkit and Nsight profiling tools
- **University of Sarajevo** for computational resources
- **OpenMP Architecture Review Board** for parallel programming standards

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional graph generators (scale-free, small-world topologies)
- GPU topological sort implementation
- 3D heat diffusion with volume rendering
- Performance comparisons with other parallel frameworks

---

## 📧 Contact

For questions or collaborations:
- Anes Ćenanović: acenanovi1@etf.unsa.ba
- Faris Crnčalo: fcrncal1@etf.unsa.ba
- Esma Dizdarević: edizdarevi1@etf.unsa.ba

**Repository:** [https://github.com/AnesCenanovic/Parallel-Computer-Systems](https://github.com/AnesCenanovic/Parallel-Computer-Systems)
