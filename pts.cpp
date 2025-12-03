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
#include <cstring>
#include <unordered_map>

// =============================================================================
// COMPRESSED SPARSE ROW (CSR) STRUCTURE
// =============================================================================
struct CSRGraph {
    std::vector<int> row_offsets;
    std::vector<int> flat_adj;
    int n;
};

CSRGraph convert_to_csr(const std::vector<std::vector<int>>& adj, bool parallel) {
    int n = adj.size();
    CSRGraph g;
    g.n = n;
    g.row_offsets.resize(n + 1);

    g.row_offsets[0] = 0;
    for (int i = 0; i < n; ++i) {
        g.row_offsets[i+1] = g.row_offsets[i] + adj[i].size();
    }

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

// =============================================================================
// SERIAL BASELINE (Your Best Sequential)
// =============================================================================
std::vector<int> topological_sort_serial(const std::vector<std::vector<int>>& raw_adj) {
    CSRGraph g = convert_to_csr(raw_adj, false);
    int n = g.n;

    std::vector<int> indegree(n, 0);

    for (size_t i = 0; i < g.flat_adj.size(); ++i) {
        indegree[g.flat_adj[i]]++;
    }

    std::vector<int> frontier;
    frontier.reserve(n);

    for (int i = 0; i < n; ++i) {
        if (indegree[i] == 0) {
            frontier.push_back(i);
        }
    }

    std::vector<int> order;
    order.reserve(n);

    size_t head = 0;
    while (head < frontier.size()) {
        int u = frontier[head++];
        order.push_back(u);

        int start = g.row_offsets[u];
        int end = g.row_offsets[u+1];

        for (int j = start; j < end; ++j) {
            int v = g.flat_adj[j];
            
            if (j + 8 < end) {
                _mm_prefetch((const char*)&indegree[g.flat_adj[j+8]], _MM_HINT_T0);
            }
            
            if (--indegree[v] == 0) {
                frontier.push_back(v);
            }
        }
    }

    return order;
}

// =============================================================================
// PARALLEL V1: SPARSE DELTA (Fixed Memory Issue)
// Key: Only store deltas for nodes actually touched
// =============================================================================
std::vector<int> topological_sort_parallel_sparse(const std::vector<std::vector<int>>& raw_adj) {
    CSRGraph g = convert_to_csr(raw_adj, true);
    int n = g.n;
    
    const int num_threads = omp_get_max_threads();
    
    // Parallel indegree calculation
    std::vector<int> indegree(n, 0);
    
    #pragma omp parallel
    {
        std::vector<int> local_indegree(n, 0);
        
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < g.flat_adj.size(); ++i) {
            local_indegree[g.flat_adj[i]]++;
        }
        
        #pragma omp critical
        for (int v = 0; v < n; ++v) {
            indegree[v] += local_indegree[v];
        }
    }
    
    // Find initial frontier
    std::vector<int> frontier;
    frontier.reserve(n);
    
    #pragma omp parallel
    {
        std::vector<int> local_frontier;
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i) {
            if (indegree[i] == 0) {
                local_frontier.push_back(i);
            }
        }
        #pragma omp critical
        frontier.insert(frontier.end(), local_frontier.begin(), local_frontier.end());
    }
    
    std::vector<int> order;
    order.reserve(n);
    
    const int PARALLEL_THRESHOLD = 4096;
    
    // BFS with SPARSE delta tracking
    while (!frontier.empty()) {
        order.insert(order.end(), frontier.begin(), frontier.end());
        
        if (frontier.size() >= PARALLEL_THRESHOLD) {
            // Use SPARSE storage: only store deltas for touched nodes
            std::vector<std::unordered_map<int, int>> thread_deltas(num_threads);
            std::vector<std::vector<int>> thread_next(num_threads);
            
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& local_deltas = thread_deltas[tid];
                auto& local_next = thread_next[tid];
                
                #pragma omp for schedule(dynamic, 64) nowait
                for (size_t i = 0; i < frontier.size(); ++i) {
                    int u = frontier[i];
                    int start = g.row_offsets[u];
                    int end = g.row_offsets[u+1];
                    
                    for (int j = start; j < end; ++j) {
                        int v = g.flat_adj[j];
                        local_deltas[v]++;  // Sparse: only touched nodes
                    }
                }
                
                #pragma omp barrier
                
                // Each thread processes a partition
                #pragma omp for schedule(static)
                for (int v = 0; v < n; ++v) {
                    int total_delta = 0;
                    for (int t = 0; t < num_threads; ++t) {
                        auto it = thread_deltas[t].find(v);
                        if (it != thread_deltas[t].end()) {
                            total_delta += it->second;
                        }
                    }
                    
                    if (total_delta > 0) {
                        indegree[v] -= total_delta;
                        if (indegree[v] == 0) {
                            local_next.push_back(v);
                        }
                    }
                }
            }
            
            std::vector<int> next_frontier;
            for (const auto& local : thread_next) {
                next_frontier.insert(next_frontier.end(), local.begin(), local.end());
            }
            frontier = std::move(next_frontier);
            
        } else {
            // Serial for small frontiers
            std::vector<int> next_frontier;
            
            for (int u : frontier) {
                int start = g.row_offsets[u];
                int end = g.row_offsets[u+1];
                
                for (int j = start; j < end; ++j) {
                    int v = g.flat_adj[j];
                    
                    if (j + 8 < end) {
                        _mm_prefetch((const char*)&indegree[g.flat_adj[j+8]], _MM_HINT_T0);
                    }
                    
                    if (--indegree[v] == 0) {
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
// PARALLEL V2: OPTIMIZED ATOMICS (Your Current Best)
// Improvements: Better scheduling, reduced padding, smarter thresholds
// =============================================================================
std::vector<int> topological_sort_parallel_atomic(const std::vector<std::vector<int>>& raw_adj) {
    CSRGraph g = convert_to_csr(raw_adj, true);
    int n = g.n;
    
    // Regular atomics (no padding - your V2 proves padding isn't the bottleneck)
    std::vector<std::atomic<int>> indegree(n);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        indegree[i].store(0, std::memory_order_relaxed);
    }
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < g.flat_adj.size(); ++i) {
        indegree[g.flat_adj[i]].fetch_add(1, std::memory_order_relaxed);
    }
    
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
        frontier.insert(frontier.end(), local_frontier.begin(), local_frontier.end());
    }
    
    std::vector<int> order;
    order.reserve(n);
    
    const int PARALLEL_THRESHOLD = 2048;  // Lower threshold
    
    while (!frontier.empty()) {
        order.insert(order.end(), frontier.begin(), frontier.end());
        
        if (frontier.size() >= PARALLEL_THRESHOLD) {
            const int num_threads = omp_get_max_threads();
            std::vector<std::vector<int>> thread_next(num_threads);
            
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                thread_next[tid].reserve(frontier.size() / num_threads + 128);
                
                // Dynamic scheduling with smaller chunks for better load balance
                #pragma omp for schedule(dynamic, 32) nowait
                for (size_t i = 0; i < frontier.size(); ++i) {
                    int u = frontier[i];
                    int start = g.row_offsets[u];
                    int end = g.row_offsets[u+1];
                    
                    // Process in small batches to improve locality
                    for (int j = start; j < end; ++j) {
                        int v = g.flat_adj[j];
                        
                        // Aggressive prefetching
                        if (j + 4 < end) {
                            _mm_prefetch((const char*)&indegree[g.flat_adj[j+4]], _MM_HINT_T0);
                        }
                        
                        if (indegree[v].fetch_sub(1, std::memory_order_relaxed) == 1) {
                            thread_next[tid].push_back(v);
                        }
                    }
                }
            }
            
            std::vector<int> next_frontier;
            for (const auto& local : thread_next) {
                next_frontier.insert(next_frontier.end(), local.begin(), local.end());
            }
            frontier = std::move(next_frontier);
            
        } else {
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
// PARALLEL V3: BATCHED UPDATES (New Approach)
// Batch atomic updates to reduce contention
// =============================================================================
std::vector<int> topological_sort_parallel_batched(const std::vector<std::vector<int>>& raw_adj) {
    CSRGraph g = convert_to_csr(raw_adj, true);
    int n = g.n;
    
    std::vector<std::atomic<int>> indegree(n);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        indegree[i].store(0, std::memory_order_relaxed);
    }
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < g.flat_adj.size(); ++i) {
        indegree[g.flat_adj[i]].fetch_add(1, std::memory_order_relaxed);
    }
    
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
        frontier.insert(frontier.end(), local_frontier.begin(), local_frontier.end());
    }
    
    std::vector<int> order;
    order.reserve(n);
    
    const int PARALLEL_THRESHOLD = 2048;
    const int BATCH_SIZE = 256;  // Process in batches
    
    while (!frontier.empty()) {
        order.insert(order.end(), frontier.begin(), frontier.end());
        
        if (frontier.size() >= PARALLEL_THRESHOLD) {
            const int num_threads = omp_get_max_threads();
            std::vector<std::vector<int>> thread_next(num_threads);
            
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                thread_next[tid].reserve(frontier.size() / num_threads + 128);
                
                // Process frontier in batches
                #pragma omp for schedule(dynamic, 1)
                for (size_t batch_start = 0; batch_start < frontier.size(); batch_start += BATCH_SIZE) {
                    size_t batch_end = std::min(batch_start + BATCH_SIZE, frontier.size());
                    
                    for (size_t i = batch_start; i < batch_end; ++i) {
                        int u = frontier[i];
                        int start = g.row_offsets[u];
                        int end = g.row_offsets[u+1];
                        
                        for (int j = start; j < end; ++j) {
                            int v = g.flat_adj[j];
                            
                            if (j + 4 < end) {
                                _mm_prefetch((const char*)&indegree[g.flat_adj[j+4]], _MM_HINT_T0);
                            }
                            
                            if (indegree[v].fetch_sub(1, std::memory_order_relaxed) == 1) {
                                thread_next[tid].push_back(v);
                            }
                        }
                    }
                }
            }
            
            std::vector<int> next_frontier;
            for (const auto& local : thread_next) {
                next_frontier.insert(next_frontier.end(), local.begin(), local.end());
            }
            frontier = std::move(next_frontier);
            
        } else {
            std::vector<int> next_frontier;
            
            for (int u : frontier) {
                int start = g.row_offsets[u];
                int end = g.row_offsets[u+1];
                
                for (int j = start; j < end; ++j) {
                    int v = g.flat_adj[j];
                    
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
// GRAPH GENERATORS
// =============================================================================
std::vector<std::vector<int>> generate_easy_dag(int n, int edges_per_node) {
    std::vector<std::vector<int>> adj(n);
    #pragma omp parallel for schedule(static)
    for (int u = 0; u < n; ++u) {
        for (int i = 1; i <= edges_per_node; ++i) {
            if (u + i < n) {
                adj[u].push_back(u + i);
            }
        }
    }
    return adj;
}

std::vector<std::vector<int>> generate_challenging_dag(int n, int avg_edges_per_node) {
    std::vector<std::vector<int>> adj(n);
    std::mt19937 gen(1337);

    for (int u = 0; u < n; ++u) {
        if (u + 1 >= n) continue;

        std::uniform_int_distribution<int> dist(u + 1, n - 1);
        std::vector<int> neighbors;
        neighbors.reserve(avg_edges_per_node);

        for (int i = 0; i < avg_edges_per_node; ++i) {
            neighbors.push_back(dist(gen));
        }

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

        for (int i = 0; i < avg_edges_per_node; ++i) {
            neighbors.push_back(dist(gen));
        }

        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

        adj[u] = neighbors;
    }
    return adj;
}

// =============================================================================
// BENCHMARKING
// =============================================================================
void run_benchmark(const std::string& version,
                   const std::string& graph_name,
                   const std::string& description,
                   const std::vector<std::vector<int>>& adj,
                   std::vector<int>(*func)(const std::vector<std::vector<int>>&)) {

    std::cout << "---------------------------------------------------------\n";
    std::cout << "--- " << version << " - " << graph_name << " ---\n";
    std::cout << "--- " << description << " ---\n";
    std::cout << "---------------------------------------------------------\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = func(adj);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << version << " completed in: "
              << std::fixed << std::setprecision(4)
              << duration.count() << " seconds.\n";
    std::cout << "Sorted " << result.size() << " nodes.\n\n";
}

int main() { 
    omp_set_num_threads(8);
    
    int num_nodes = 20000000;  
    int edges_per_node = 15;

    std::cout << "=============================================================\n";
    std::cout << "TOPOLOGICAL SORT - REVISED COMPARISON\n";
    std::cout << "=============================================================\n";
    std::cout << "Running with " << omp_get_max_threads() << " threads\n";
    std::cout << "Nodes: " << num_nodes << ", Avg edges/node: " << edges_per_node << "\n\n";

    // Test 1: Sequential Graph
    {
        std::cout << "\n### TEST 1: SEQUENTIAL GRAPH ###\n\n";
        
        auto adj_easy = generate_easy_dag(num_nodes, edges_per_node);
        run_benchmark("SERIAL", "Sequential", "Baseline", adj_easy, topological_sort_serial);
        run_benchmark("PARALLEL (Atomic)", "Sequential", "Optimized atomics", adj_easy, topological_sort_parallel_atomic);
        run_benchmark("PARALLEL (Batched)", "Sequential", "Batched updates", adj_easy, topological_sort_parallel_batched);
    }

    // Test 2: Narrow Frontier
    {
        std::cout << "\n### TEST 2: NARROW FRONTIER ###\n\n";
        
        auto adj_challenge = generate_challenging_dag(num_nodes, edges_per_node);
        run_benchmark("SERIAL", "Narrow", "Baseline", adj_challenge, topological_sort_serial);
        run_benchmark("PARALLEL (Atomic)", "Narrow", "Optimized atomics", adj_challenge, topological_sort_parallel_atomic);
        run_benchmark("PARALLEL (Batched)", "Narrow", "Batched updates", adj_challenge, topological_sort_parallel_batched);
    }

    // Test 3: Wide Frontier
    {
        std::cout << "\n### TEST 3: WIDE FRONTIER ###\n\n";
        
        auto adj_wide = generate_wide_frontier_dag(num_nodes, edges_per_node);
        run_benchmark("SERIAL", "Wide", "Baseline", adj_wide, topological_sort_serial);
        run_benchmark("PARALLEL (Atomic)", "Wide", "Optimized atomics", adj_wide, topological_sort_parallel_atomic);
        run_benchmark("PARALLEL (Batched)", "Wide", "Batched updates", adj_wide, topological_sort_parallel_batched);
    }
    
    std::cout << "\n=============================================================\n";
    std::cout << "RECOMMENDATION:\n";
    std::cout << "=============================================================\n";
    std::cout << "BEST SERIAL: topological_sort_serial\n";
    std::cout << "BEST PARALLEL: topological_sort_parallel_atomic or batched\n";
    std::cout << "(whichever performs better on your wide frontier test)\n";
    std::cout << "=============================================================\n";

    return 0;
}
