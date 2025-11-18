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

// Serial version for comparison
std::vector<int> topological_sort_kahn_serial(const std::vector<std::vector<int>>& adj) {
    int n = adj.size();
    std::vector<int> indegree(n);

    for (int u = 0; u < n; ++u) {
        for (int v : adj[u]) {
            indegree[v]++;
        }
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

    while (!frontier.empty()) {
        for (int u : frontier)
            order.push_back(u);

        next_frontier.clear();

        for (size_t i = 0; i < frontier.size(); i++) {
            int u = frontier[i];

            for (int v : adj[u]) {
                int new_val = --indegree[v];

                if (new_val == 0) {
                    next_frontier.push_back(v);
                }
            }
        }

        frontier.swap(next_frontier);
    }

    return order;
}

// Optimized parallel version
std::vector<int> topological_sort_kahn_parallel(const std::vector<std::vector<int>>& adj) {
    int n = adj.size();
    std::vector<std::atomic<int>> indegree(n);
    
    // Initialize atomics to 0
    for (int i = 0; i < n; ++i) {
        indegree[i].store(0, std::memory_order_relaxed);
    }

    // 1) Parallel indegree computation with padding to avoid false sharing
    #pragma omp parallel for schedule(static)
    for (int u = 0; u < n; ++u) {
        for (int v : adj[u]) {
            indegree[v].fetch_add(1, std::memory_order_relaxed);
        }
    }

    // 2) Initial frontier
    std::vector<int> frontier;
    frontier.reserve(n);

    for (int i = 0; i < n; ++i)
        if (indegree[i].load(std::memory_order_relaxed) == 0)
            frontier.push_back(i);

    std::vector<int> next_frontier;
    next_frontier.reserve(n);

    std::vector<int> order;
    order.reserve(n);

    const int PARALLEL_THRESHOLD = 5000; // Only parallelize if frontier is large enough

    // 3) Process level-by-level
    while (!frontier.empty()) {
        // Append current layer to order
        for (int u : frontier)
            order.push_back(u);

        next_frontier.clear();

        // Decide whether to parallelize based on frontier size
        if (frontier.size() >= PARALLEL_THRESHOLD) {
            // PARALLEL PATH: Large frontier, worth the overhead
            
            // Pre-allocate thread-local storage
            int num_threads = omp_get_max_threads();
            std::vector<std::vector<int>> thread_local_results(num_threads);
            
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                thread_local_results[tid].reserve(frontier.size() / num_threads + 1024);
                
                #pragma omp for schedule(dynamic, 512) nowait
                for (size_t i = 0; i < frontier.size(); i++) {
                    int u = frontier[i];

                    for (int v : adj[u]) {
                        int new_val = indegree[v].fetch_sub(1, std::memory_order_relaxed) - 1;

                        if (new_val == 0) {
                            thread_local_results[tid].push_back(v);
                        }
                    }
                }
            }
            
            // Merge thread-local results (serial, but fast)
            size_t total_size = 0;
            for (const auto& local : thread_local_results) {
                total_size += local.size();
            }
            next_frontier.reserve(total_size);
            
            for (auto& local : thread_local_results) {
                next_frontier.insert(next_frontier.end(), local.begin(), local.end());
            }
            
        } else {
            // SERIAL PATH: Small frontier, avoid parallel overhead
            
            for (size_t i = 0; i < frontier.size(); i++) {
                int u = frontier[i];

                for (int v : adj[u]) {
                    int new_val = indegree[v].fetch_sub(1, std::memory_order_relaxed) - 1;

                    if (new_val == 0) {
                        next_frontier.push_back(v);
                    }
                }
            }
        }

        frontier.swap(next_frontier);
    }

    return order;
}

// DAG generators
std::vector<std::vector<int>> generate_easy_dag(int n, int edges_per_node) {
    std::vector<std::vector<int>> adj(n);
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
    omp_set_num_threads(8);

    int num_nodes = 20000000;  
    int edges_per_node = 15;

    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    // Test 1: Easy DAG
    {
        auto adj_easy = generate_easy_dag(num_nodes, edges_per_node);
        run_benchmark("SERIAL", "Best Case", "Cache-friendly DAG", adj_easy, false);
        run_benchmark("PARALLEL", "Best Case", "Cache-friendly DAG", adj_easy, true);
    }

    // Test 2: Challenging DAG
    {
        auto adj_challenge = generate_challenging_dag(num_nodes, edges_per_node);
        run_benchmark("SERIAL", "Narrow Frontier", "Serial workflow DAG", adj_challenge, false);
        run_benchmark("PARALLEL", "Narrow Frontier", "Serial workflow DAG", adj_challenge, true);
    }

    // Test 3: Wide Frontier DAG
    {
        auto adj_wide = generate_wide_frontier_dag(num_nodes, edges_per_node);
        run_benchmark("SERIAL", "Wide Frontier", "High parallelism DAG", adj_wide, false);
        run_benchmark("PARALLEL", "Wide Frontier", "High parallelism DAG", adj_wide, true);
    }

    return 0;
}
