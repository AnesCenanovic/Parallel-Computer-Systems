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

std::vector<int> topological_sort_kahn_parallel(const std::vector<std::vector<int>>& adj) {
    int n = adj.size();
    std::vector<std::atomic<int>> indegree(n);

    // 1) PARALLEL indegree computation
    #pragma omp parallel for schedule(static)
    for (int u = 0; u < n; ++u) {
        for (int v : adj[u]) {
            indegree[v].fetch_add(1, std::memory_order_relaxed);
        }
    }

    // 2) Initial frontier (all nodes with 0 indegree)
    std::vector<int> frontier;
    frontier.reserve(n);

    for (int i = 0; i < n; ++i)
        if (indegree[i].load() == 0)
            frontier.push_back(i);

    std::vector<int> next_frontier;
    next_frontier.reserve(n);

    std::vector<int> order;
    order.reserve(n);

    // 3) Process level-by-level
    while (!frontier.empty()) {

        // append current layer to order
        for (int u : frontier)
            order.push_back(u);

        // clear next frontier
        next_frontier.clear();

        // 4) PARALLEL expand all nodes in current frontier
        #pragma omp parallel
        {
            std::vector<int> local_new; // per-thread results
            local_new.reserve(1024);

            #pragma omp for schedule(dynamic, 1024)
            for (size_t i = 0; i < frontier.size(); i++) {
                int u = frontier[i];

                for (int v : adj[u]) {
                    // atomic decrement
                    int new_val = indegree[v].fetch_sub(1) - 1;

                    // if indegree hits zero -> new frontier node
                    if (new_val == 0) {
                        local_new.push_back(v);
                    }
                }
            }

            // merge thread-local results into shared next_frontier
            #pragma omp critical
            next_frontier.insert(next_frontier.end(), local_new.begin(), local_new.end());
        }

        frontier.swap(next_frontier);
    }

    return order;
}

//DAG generators

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

void run_benchmark_parallel(const std::string& graph_name,
                            const std::string& description,
                            const std::vector<std::vector<int>>& adj) {

    std::cout << "---------------------------------------------------------\n";
    std::cout << "--- Parallel Test Case: " << graph_name << " ---\n";
    std::cout << "--- Description: " << description << " ---\n";
    std::cout << "---------------------------------------------------------\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    auto result = topological_sort_kahn_parallel(adj);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Parallel sort completed in: "
              << std::fixed << std::setprecision(4)
              << duration.count() << " seconds.\n";
    std::cout << "Sorted " << result.size() << " nodes.\n\n";
}

int main() {
    omp_set_num_threads(8);   

    int num_nodes = 2'000'000;
    int edges_per_node = 15;

    // Test 1
    {
        auto adj_easy = generate_easy_dag(num_nodes, edges_per_node);
        run_benchmark_parallel("Best Case (Parallel)", "Cache-friendly DAG", adj_easy);
    }

    // Test 2
    {
        auto adj_challenge = generate_challenging_dag(num_nodes, edges_per_node);
        run_benchmark_parallel("Narrow Frontier (Parallel)", "Serial workflow DAG", adj_challenge);
    }

    // Test 3
    {
        auto adj_wide = generate_wide_frontier_dag(num_nodes, edges_per_node);
        run_benchmark_parallel("Wide Frontier (Parallel)", "High parallelism DAG", adj_wide);
    }

    return 0;
}
