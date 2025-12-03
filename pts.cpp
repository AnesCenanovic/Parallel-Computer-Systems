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

struct CSRGraph {
    std::vector<int> row_offsets;
    std::vector<int> flat_adj;
    int n;
};

// Converts vector<vector> to Flat Array (CSR) to fix Cache Misses
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
            
            // Prefetch next iterations to hide DRAM latency
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


std::vector<int> topological_sort_parallel(const std::vector<std::vector<int>>& raw_adj) {

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
    
    while (!frontier.empty()) {
        order.insert(order.end(), frontier.begin(), frontier.end());
        
        if (frontier.size() >= PARALLEL_THRESHOLD) {
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
                    if (j + 8 < end) _mm_prefetch((const char*)&indegree[g.flat_adj[j+8]], _MM_HINT_T0);
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
    for (int u = 0; u < n; ++u)
        for (int i = 1; i <= edges_per_node; ++i)
            if (u + i < n)
                adj[u].push_back(u + i);
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
        for (int i = 0; i < avg_edges_per_node; ++i) neighbors.push_back(dist(gen));
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
        for (int i = 0; i < avg_edges_per_node; ++i) neighbors.push_back(dist(gen));
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        adj[u] = neighbors;
    }
    return adj;
}

std::vector<std::vector<int>> generate_disjoint_clusters(int n, int edges_per_node) {
    std::vector<std::vector<int>> adj(n);
    int num_clusters = 8; // Match thread count approx
    int nodes_per_cluster = n / num_clusters;

    #pragma omp parallel for
    for (int c = 0; c < num_clusters; ++c) {
        int start = c * nodes_per_cluster;
        int end = start + nodes_per_cluster;
        std::mt19937 gen(1337 + c); 

        for (int u = start; u < end; ++u) {
            if (u + 1 >= end) continue;
            std::uniform_int_distribution<int> dist(u + 1, end - 1);
            std::vector<int> neighbors;
            for (int i = 0; i < edges_per_node; ++i) neighbors.push_back(dist(gen));
            
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
            adj[u] = neighbors;
        }
    }
    return adj;
}

// =============================================================================
// BENCHMARKING HARNESS
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

    // 1. Warmup Run (Executes but doesn't count time)
    std::cout << "Warming up...\r" << std::flush;
    volatile size_t dummy = func(adj).size(); 

    // 2. Average of 3 Runs
    int iterations = 3;
    double total_time = 0;
    std::cout << "Benchmarking... (" << iterations << " iterations)\n";

    for(int i=0; i<iterations; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto result = func(adj);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        total_time += duration.count();
    }

    std::cout << version << " Average Time: "
              << std::fixed << std::setprecision(4)
              << (total_time / iterations) << " seconds.\n\n";
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <algo> <graph>\n";
    std::cout << "  <algo>:  serial, parallel\n";
    std::cout << "  <graph>: easy, narrow, wide, disjoint, dense\n";
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string algo = argv[1];
    std::string graph_type = argv[2];

    // Optimized for i7-10510U (4 Physical Cores). 
    // Remove or change this if running on a high-core-count server.
    omp_set_num_threads(4);
    
    // Default Parameters (20M nodes)
    int num_nodes = 20000000;
    int edges = 15;

    // Special Parameter Logic
    if (graph_type == "dense") {
        num_nodes = 2000000; // Less nodes
        edges = 100;         // More edges (Compute heavy)
    }

    std::vector<std::vector<int>> adj;
    std::string desc;

    std::cout << "Generating " << graph_type << " graph with " << num_nodes 
              << " nodes and " << edges << " edges/node...\n";

    // Graph Generation Selection
    if (graph_type == "easy") {
        adj = generate_easy_dag(num_nodes, edges);
        desc = "Linear (Best Case)";
    } else if (graph_type == "narrow") {
        adj = generate_challenging_dag(num_nodes, edges);
        desc = "Random (Avg Case)";
    } else if (graph_type == "wide") {
        adj = generate_wide_frontier_dag(num_nodes, edges);
        desc = "Random (Wide Frontier)";
    } else if (graph_type == "disjoint") {
        adj = generate_disjoint_clusters(num_nodes, edges);
        desc = "Clusters (No False Sharing)";
    } else if (graph_type == "dense") {
        adj = generate_challenging_dag(num_nodes, edges);
        desc = "Dense (Atomic Stress Test)";
    } else {
        std::cerr << "Unknown graph type: " << graph_type << "\n";
        return 1;
    }

    // Algorithm Execution
    if (algo == "serial") {
        run_benchmark("SERIAL", graph_type, desc, adj, topological_sort_serial);
    } 
    else if (algo == "parallel") {
        run_benchmark("PARALLEL", graph_type, desc, adj, topological_sort_parallel);
    } 
    else {
        std::cerr << "Unknown algorithm: " << algo << "\n";
        return 1;
    }

    return 0;
}
