#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <random>
#include <algorithm>
#include <string>
#include <iomanip>
#include <atomic>
#include <omp.h>
#include <immintrin.h> 

// -----------------------------------------------------------------------------
// HELPER: Compressed Sparse Row (CSR) Structure
// -----------------------------------------------------------------------------
struct CSRGraph {
    std::vector<int> row_offsets; // Where each node's edges start
    std::vector<int> flat_adj;    // All edges in one contiguous array
    int n;
};

// Converts vector<vector> to Flat Arrays (CSR)
// This fixes the "Cache usage" and "DRAM" issues by creating spatial locality.
CSRGraph convert_to_csr(const std::vector<std::vector<int>>& adj, bool parallel) {
    int n = adj.size();
    CSRGraph g;
    g.n = n;
    g.row_offsets.resize(n + 1);cd 

    // 1. Calculate offsets (Prefix Sum)
    g.row_offsets[0] = 0;
    for (int i = 0; i < n; ++i) {
        g.row_offsets[i+1] = g.row_offsets[i] + adj[i].size();
    }

    // 2. Flatten the data
    // Parallel copy helps significantly when N is large (20M+)
    g.flat_adj.resize(g.row_offsets[n]);
    
    if (parallel) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            if (!adj[i].empty()) {
                int start = g.row_offsets[i];
                std::copy(adj[i].begin(), adj[i].end(), g.flat_adj.begin() + start);
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            if (!adj[i].empty()) {
                int start = g.row_offsets[i];
                std::copy(adj[i].begin(), adj[i].end(), g.flat_adj.begin() + start);
            }
        }
    }
    
    return g;
}

// -----------------------------------------------------------------------------
// OPTIMIZED SERIAL VERSION (BASELINE)
// -----------------------------------------------------------------------------
std::vector<int> topological_sort_kahn_serial(const std::vector<std::vector<int>>& raw_adj) {
    // Optimization: Convert to Flat Array even for Serial to make comparison fair
    // and to speed up the baseline using cache locality.
    CSRGraph g = convert_to_csr(raw_adj, false);
    int n = g.n;

    std::vector<int> indegree(n, 0);

    // Linear memory access pattern (Friendly to Hardware Prefetcher)
    for (int i = 0; i < g.flat_adj.size(); ++i) {
        indegree[g.flat_adj[i]]++;
    }

    std::vector<int> frontier;
    frontier.reserve(n);

    for (int i = 0; i < n; ++i)
        if (indegree[i] == 0)
            frontier.push_back(i);

    std::vector<int> next_frontier;
    next_frontier.reserve(n);

    std::vector<int> order;
    order.reserve(n);

    // Standard pointer-chasing queue simulation
    int head = 0;
    while(head < frontier.size()) {
        int u = frontier[head++];
        order.push_back(u);

        int start = g.row_offsets[u];
        int end = g.row_offsets[u+1];

        // Accessing flat_adj[start...end] is sequential -> L1 Cache Hit
        for (int j = start; j < end; ++j) {
            int v = g.flat_adj[j];
            if (--indegree[v] == 0) {
                frontier.push_back(v);
            }
        }
    }

    return order;
}

// -----------------------------------------------------------------------------
// OPTIMIZED PARALLEL VERSION
// -----------------------------------------------------------------------------
std::vector<int> topological_sort_kahn_parallel(const std::vector<std::vector<int>>& raw_adj) {
    // 1. Flatten Matrix -> Array (Solves "Memory Bound" and L1 Cache issues)
    CSRGraph g = convert_to_csr(raw_adj, true);
    int n = g.n;

    std::vector<std::atomic<int>> indegree(n);
    
    // 2. Vectorized Initialization (SIMD friendly)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        indegree[i].store(0, std::memory_order_relaxed);
    }

    // 3. Parallel Indegree Calculation
    // We iterate the EDGE array, not the node array. This is perfectly linear memory access.
    const size_t total_edges = g.flat_adj.size();
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total_edges; ++i) {
        // atomic fetch_add is required, but input data read is sequential
        indegree[g.flat_adj[i]].fetch_add(1, std::memory_order_relaxed);
    }

    std::vector<int> frontier;
    frontier.reserve(n);

    // 4. Find initial frontier (Parallel Scan)
    #pragma omp parallel
    {
        std::vector<int> local_frontier;
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i) {
            if (indegree[i].load(std::memory_order_relaxed) == 0)
                local_frontier.push_back(i);
        }
        #pragma omp critical
        frontier.insert(frontier.end(), local_frontier.begin(), local_frontier.end());
    }

    std::vector<int> next_frontier;
    next_frontier.reserve(n);
    std::vector<int> order;
    order.reserve(n);

    const int PARALLEL_THRESHOLD = 4096; 

    while (!frontier.empty()) {
        // Append current frontier to order
        order.insert(order.end(), frontier.begin(), frontier.end());
        next_frontier.clear();

        if (frontier.size() >= PARALLEL_THRESHOLD) {
            // --- PARALLEL EXECUTION ---
            
            int num_threads = omp_get_max_threads();
            std::vector<std::vector<int>> thread_local_results(num_threads);
            
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                thread_local_results[tid].reserve(frontier.size() / num_threads + 512);
                
                #pragma omp for schedule(dynamic, 128)
                for (size_t i = 0; i < frontier.size(); i++) {
                    int u = frontier[i];

                    int start = g.row_offsets[u];
                    int end = g.row_offsets[u+1];
                    const int* neighbors = &g.flat_adj[start];
                    int len = end - start;

                    // Manual Unrolling + Prefetching Loop
                    // Helps hide the latency of fetching atomic indegrees from DRAM
                    int j = 0;
                    for (; j <= len - 4; j += 4) {
                        int v0 = neighbors[j];
                        int v1 = neighbors[j+1];
                        int v2 = neighbors[j+2];
                        int v3 = neighbors[j+3];

                        // Software Prefetch: Look ahead 8 elements
                        if (j + 8 < len) {
                            _mm_prefetch((const char*)&indegree[neighbors[j+8]], _MM_HINT_T0);
                        }

                        if (indegree[v0].fetch_sub(1, std::memory_order_relaxed) == 1) thread_local_results[tid].push_back(v0);
                        if (indegree[v1].fetch_sub(1, std::memory_order_relaxed) == 1) thread_local_results[tid].push_back(v1);
                        if (indegree[v2].fetch_sub(1, std::memory_order_relaxed) == 1) thread_local_results[tid].push_back(v2);
                        if (indegree[v3].fetch_sub(1, std::memory_order_relaxed) == 1) thread_local_results[tid].push_back(v3);
                    }

                    // Handle remaining
                    for (; j < len; ++j) {
                        int v = neighbors[j];
                        if (indegree[v].fetch_sub(1, std::memory_order_relaxed) == 1) {
                            thread_local_results[tid].push_back(v);
                        }
                    }
                }
            }
            
            // Merge results
            for (const auto& local : thread_local_results) {
                next_frontier.insert(next_frontier.end(), local.begin(), local.end());
            }
            
        } else {
            // --- SERIAL EXECUTION (Optimized) ---
            
            for (int u : frontier) {
                int start = g.row_offsets[u];
                int end = g.row_offsets[u+1];

                for (int j = start; j < end; ++j) {
                    int v = g.flat_adj[j];
                    
                    // Simple prefetch helps serial too
                    if (j + 4 < end) _mm_prefetch((const char*)&indegree[g.flat_adj[j+4]], _MM_HINT_T0);

                    if (indegree[v].fetch_sub(1, std::memory_order_relaxed) == 1) {
                        next_frontier.push_back(v);
                    }
                }
            }
        }

        frontier.swap(next_frontier);
    }

    return order;
}

// -----------------------------------------------------------------------------
// DAG GENERATORS (Unchanged)
// -----------------------------------------------------------------------------
std::vector<std::vector<int>> generate_easy_dag(int n, int edges_per_node) {
    std::vector<std::vector<int>> adj(n);
    #pragma omp parallel for schedule(static)
    for (int u = 0; u < n; ++u)
        for (int i = 1; i <= edges_per_node; ++i)
            if (u + i < n)
                adj[u].push_back(u + i);
    return adj;
}

std::vector<std::vector<int>> generate_challenging_dag(int n, int avg_edges_per_node) {
    std::vector<std::vector<int>> adj(n);
    // Use a deterministic seed but thread-safe generation is tricky.
    // We generate serially for correctness of the benchmark setup provided.
    std::mt19937 gen(1337);

    for (int u = 0; u < n; ++u) {
        if (u + 1 >= n) continue;

        std::uniform_int_distribution<int> dist(u + 1, n - 1);
        std::vector<int> neighbors;
        neighbors.reserve(avg_edges_per_node);

        for (int i = 0; i < avg_edges_per_node; ++i)
            neighbors.push_back(dist(gen));

        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

        adj[u] = neighbors;
    }
    return adj;
}

std::vector<std::vector<int>> generate_wide_frontier_dag(int n, int avg_edges_per_node, double source_node_ratio = 0.1) {
    std::vector<std::vector<int>> adj(n);
    std::mt19937 gen(1337);

    int first_non_source_node = static_cast<int>(n * source_node_ratio);
    if (first_non_source_node <= 0) first_non_source_node = 1;

    for (int u = first_non_source_node; u < n; ++u) {
        if (u + 1 >= n) continue;

        std::uniform_int_distribution<int> dist(u + 1, n - 1);
        std::vector<int> neighbors;
        neighbors.reserve(avg_edges_per_node);

        for (int i = 0; i < avg_edges_per_node; ++i)
            neighbors.push_back(dist(gen));

        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

        adj[u] = neighbors;
    }
    return adj;
}

void run_benchmark(const std::string& version,
                   const std::string& graph_name,
                   const std::string& description,
                   const std::vector<std::vector<int>>& adj,
                   bool parallel) {

    std::cout << "---------------------------------------------------------\n";
    std::cout << "--- " << version << " - " << graph_name << " ---\n";
    std::cout << "--- Description: " << description << " ---\n";
    std::cout << "---------------------------------------------------------\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    auto result = parallel ? topological_sort_kahn_parallel(adj) 
                          : topological_sort_kahn_serial(adj);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << version << " completed in: "
              << std::fixed << std::setprecision(4)
              << duration.count() << " seconds.\n";
    std::cout << "Sorted " << result.size() << " nodes.\n\n";
}

int main() { 
    // Setting thread count. 
    // Optimization: Ensure OMP doesn't oversubscribe if hyperthreading is slow for memory bound tasks.
    omp_set_num_threads(8);

    // Note: 20M nodes is very large (approx 1.5GB - 2GB RAM for graph). 
    // Ensure you compile in Release mode (-O3).
    int num_nodes = 20000000;  
    int edges_per_node = 15;

    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    // Test 1: Easy DAG
    {
        std::cout << "Generating Easy DAG..." << std::endl;
        auto adj_easy = generate_easy_dag(num_nodes, edges_per_node);
        run_benchmark("SERIAL", "Best Case", "Cache-friendly DAG", adj_easy, false);
        run_benchmark("PARALLEL", "Best Case", "Cache-friendly DAG", adj_easy, true);
    }

    // Test 2: Challenging DAG
    {
        std::cout << "Generating Challenging DAG..." << std::endl;
        auto adj_challenge = generate_challenging_dag(num_nodes, edges_per_node);
        run_benchmark("SERIAL", "Narrow Frontier", "Serial workflow DAG", adj_challenge, false);
        run_benchmark("PARALLEL", "Narrow Frontier", "Serial workflow DAG", adj_challenge, true);
    }

    // Test 3: Wide Frontier DAG
    {
        std::cout << "Generating Wide DAG..." << std::endl;
        auto adj_wide = generate_wide_frontier_dag(num_nodes, edges_per_node);
        run_benchmark("SERIAL", "Wide Frontier", "High parallelism DAG", adj_wide, false);
        run_benchmark("PARALLEL", "Wide Frontier", "High parallelism DAG", adj_wide, true);
    }

    return 0;
}
