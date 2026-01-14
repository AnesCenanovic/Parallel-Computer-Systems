// parallel_topo_vtune.cpp
// VTUNE-READY VERSION: Separates I/O from algorithm for clean profiling
// Compile: g++ -O3 -g -march=native -fopenmp parallel_topo_vtune.cpp -o parallel_topo_vtune
// Usage: OMP_NUM_THREADS=8 ./parallel_topo_vtune <graph_file>
//
// PROFILING WORKFLOW:
// 1. Run: OMP_NUM_THREADS=8 ./parallel_topo_vtune graph_wide.bin
// 2. Program loads data and prints "READY FOR VTUNE - PID: xxxxx"
// 3. Attach VTune: vtune -collect hotspots -target-pid xxxxx
// 4. Press Enter in the original terminal to start algorithm
// 5. VTune captures ONLY pure algorithm performance (no I/O pollutants)

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <atomic>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <unistd.h>  // for getpid()

// =============================================================================
// CSR STRUCTURE
// =============================================================================
struct CSRGraph {
    std::vector<int> row_offsets;
    std::vector<int> flat_adj;
    int n;
};

// =============================================================================
// GRAPH LOADING (NOT PROFILED)
// =============================================================================
std::vector<std::vector<int>> load_graph_binary(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "ERROR: Cannot open file: " << filename << std::endl;
        exit(1);
    }
    
    int n;
    in.read(reinterpret_cast<char*>(&n), sizeof(int));
    
    std::vector<std::vector<int>> adj(n);
    
    for (int u = 0; u < n; ++u) {
        int degree;
        in.read(reinterpret_cast<char*>(&degree), sizeof(int));
        
        if (degree > 0) {
            adj[u].resize(degree);
            in.read(reinterpret_cast<char*>(adj[u].data()), degree * sizeof(int));
        }
    }
    
    in.close();
    return adj;
}

// =============================================================================
// CSR CONVERSION (NOT PROFILED)
// =============================================================================
CSRGraph convert_to_csr(const std::vector<std::vector<int>>& adj) {
    int n = adj.size();
    CSRGraph g;
    g.n = n;
    g.row_offsets.resize(n + 1);
    
    g.row_offsets[0] = 0;
    for (int i = 0; i < n; ++i) {
        g.row_offsets[i+1] = g.row_offsets[i] + adj[i].size();
    }
    
    g.flat_adj.resize(g.row_offsets[n]);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        if (!adj[i].empty()) {
            int start = g.row_offsets[i];
            std::copy(adj[i].begin(), adj[i].end(), g.flat_adj.begin() + start);
        }
    }
    
    return g;
}

// =============================================================================
// PARALLEL TOPOLOGICAL SORT (THIS IS WHAT VTUNE WILL PROFILE)
// =============================================================================
std::vector<int> topological_sort_parallel(const CSRGraph& g) {
    int n = g.n;
    
    // Use atomics for thread-safe indegree updates
    std::vector<std::atomic<int>> indegree(n);
    
    // Initialize indegrees in parallel
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        indegree[i].store(0, std::memory_order_relaxed);
    }
    
    // Compute indegrees in parallel
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < g.flat_adj.size(); ++i) {
        indegree[g.flat_adj[i]].fetch_add(1, std::memory_order_relaxed);
    }
    
    // Find initial frontier in parallel
    std::vector<int> frontier;
    frontier.reserve(n);
    
    #pragma omp parallel
    {
        std::vector<int> local_frontier;
        
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i) {
            if (indegree[i].load(std::memory_order_relaxed) == 0) {
                local_frontier.push_back(i);
            }
        }
        
        #pragma omp critical
        {
            frontier.insert(frontier.end(), local_frontier.begin(), local_frontier.end());
        }
    }
    
    std::vector<int> order;
    order.reserve(n);
    
    const int PARALLEL_THRESHOLD = 2048;
    
    // BFS with Kahn's algorithm
    while (!frontier.empty()) {
        order.insert(order.end(), frontier.begin(), frontier.end());
        
        if (frontier.size() >= PARALLEL_THRESHOLD) {
            // PARALLEL PATH
            const int num_threads = omp_get_max_threads();
            std::vector<std::vector<int>> thread_next(num_threads);
            
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                thread_next[tid].reserve(frontier.size() / num_threads + 128);
                
                #pragma omp for schedule(dynamic, 32) nowait
                for (size_t i = 0; i < frontier.size(); ++i) {
                    int u = frontier[i];
                    int start = g.row_offsets[u];
                    int end = g.row_offsets[u+1];
                    
                    for (int j = start; j < end; ++j) {
                        int v = g.flat_adj[j];
                        
                        // Prefetch ahead
                        if (j + 4 < end) {
                            _mm_prefetch((const char*)&indegree[g.flat_adj[j+4]], _MM_HINT_T0);
                        }
                        
                        if (indegree[v].fetch_sub(1, std::memory_order_relaxed) == 1) {
                            thread_next[tid].push_back(v);
                        }
                    }
                }
            }
            
            // Merge thread-local results
            std::vector<int> next_frontier;
            for (const auto& local : thread_next) {
                next_frontier.insert(next_frontier.end(), local.begin(), local.end());
            }
            frontier = std::move(next_frontier);
            
        } else {
            // SERIAL PATH (small frontiers)
            std::vector<int> next_frontier;
            
            for (int u : frontier) {
                int start = g.row_offsets[u];
                int end = g.row_offsets[u+1];
                
                for (int j = start; j < end; ++j) {
                    int v = g.flat_adj[j];
                    
                    if (j + 8 < end) {
                        _mm_prefetch((const char*)&indegree[g.flat_adj[j+8]], _MM_HINT_T0);
                    }
                    
                    if (indegree[v].fetch_sub(1, std::memory_order_relaxed) == 1) {
                        next_frontier.push_back(v);
                    }
                }
            }
            
            frontier = std::move(next_frontier);
        }
    }
    
    return order;
}

// =============================================================================
// MAIN - VTUNE-READY: SEPARATES SETUP FROM ALGORITHM
// =============================================================================
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file.bin>\n";
        std::cerr << "Set OMP_NUM_THREADS before running\n";
        return 1;
    }
    
    std::string input_file = argv[1];
    int num_threads = omp_get_max_threads();
    
    std::cout << "=================================================\n";
    std::cout << "PARALLEL TOPOLOGICAL SORT (VTUNE-READY)\n";
    std::cout << "=================================================\n";
    std::cout << "Input: " << input_file << "\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "=================================================\n\n";
    
    // =========================================================================
    // PHASE 1: SETUP (NOT PROFILED) - Load and prepare data
    // =========================================================================
    std::cout << "[SETUP PHASE - NOT PROFILED]\n";
    std::cout << "Loading graph..." << std::flush;
    auto load_start = std::chrono::high_resolution_clock::now();
    auto adj = load_graph_binary(input_file);
    auto load_end = std::chrono::high_resolution_clock::now();
    std::cout << " Done (" 
              << std::chrono::duration<double>(load_end - load_start).count() 
              << "s)\n";
    
    std::cout << "Converting to CSR..." << std::flush;
    auto csr_start = std::chrono::high_resolution_clock::now();
    auto g = convert_to_csr(adj);
    auto csr_end = std::chrono::high_resolution_clock::now();
    std::cout << " Done (" 
              << std::chrono::duration<double>(csr_end - csr_start).count() 
              << "s)\n";
    
    std::cout << "  Nodes: " << g.n << "\n";
    std::cout << "  Edges: " << g.flat_adj.size() << "\n\n";
    
    // Free original adjacency list
    adj.clear();
    adj.shrink_to_fit();
    
    // =========================================================================
    // WAIT FOR VTUNE ATTACHMENT
    // =========================================================================
    std::cout << "=================================================\n";
    std::cout << "✓ READY FOR VTUNE PROFILING\n";
    std::cout << "=================================================\n";
    std::cout << "Process ID: " << getpid() << "\n\n";
    std::cout << "To attach VTune (in another terminal):\n";
    std::cout << "  vtune -collect hotspots -target-pid " << getpid() << "\n";
    std::cout << "  vtune -collect threading -target-pid " << getpid() << "\n\n";
    std::cout << "Press ENTER when ready to start algorithm...\n";
    std::cout << "=================================================\n";
    std::cin.get();
    
    // =========================================================================
    // PHASE 2: ALGORITHM (VTUNE PROFILES THIS)
    // =========================================================================
    std::cout << "\n[ALGORITHM PHASE - VTUNE PROFILES THIS]\n";
    std::cout << "Running topological sort (5 iterations)...\n";
    std::cout << "------------------------------------------------\n";
    
    const int NUM_RUNS = 5;
    std::vector<double> times;
    
    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // =====================================================================
        // THE PURE ALGORITHM - THIS IS WHAT VTUNE SEES
        // =====================================================================
        auto result = topological_sort_parallel(g);
        // =====================================================================
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        times.push_back(duration.count());
        
        std::cout << "  Run " << (run + 1) << ": " 
                  << std::fixed << std::setprecision(4) 
                  << duration.count() << "s"
                  << " (sorted " << result.size() << " nodes)\n";
    }
    
    std::cout << "------------------------------------------------\n";
    
    // Statistics
    double min_time = *std::min_element(times.begin(), times.end());
    double max_time = *std::max_element(times.begin(), times.end());
    double avg_time = 0;
    for (double t : times) avg_time += t;
    avg_time /= times.size();
    
    std::cout << "\nRESULTS:\n";
    std::cout << "  Best:    " << std::fixed << std::setprecision(4) << min_time << "s\n";
    std::cout << "  Worst:   " << max_time << "s\n";
    std::cout << "  Average: " << avg_time << "s\n";
    std::cout << "  Std dev: " << (max_time - min_time) / 2.0 << "s\n";
    std::cout << "=================================================\n";
    
    return 0;
}