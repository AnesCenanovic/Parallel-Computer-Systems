#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <random>
#include <algorithm>
#include <string>
#include <iomanip>

std::vector<int> topological_sort_kahn(const std::vector<std::vector<int>>& adj) {
    int n = adj.size();
    std::vector<int> indegree(n, 0);
    for (const auto& neighbors : adj) {
        for (int v : neighbors) {
            indegree[v]++;
        }
    }
    std::queue<int> q;
    for (int i = 0; i < n; ++i) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }
    std::vector<int> order;
    order.reserve(n);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        order.push_back(u);
        for (int v : adj[u]) {
            indegree[v]--;
            if (indegree[v] == 0) {
                q.push(v);
            }
        }
    }
    return order;
}

// Best Case - Sequential cache-friendly DAG
std::vector<std::vector<int>> generate_easy_dag(int n, int edges_per_node) {
    std::vector<std::vector<int>> adj(n);
    for (int u = 0; u < n; ++u) {
        for (int i = 1; i <= edges_per_node; ++i) {
            int v = u + i;
            if (v < n) {
                adj[u].push_back(v);
            }
        }
    }
    return adj;
}

// Common Case - Cache-hostile DAG with a narrow workflow
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

// Parallel-Ready Case - Cache-hostile DAG with a wide workflow
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

void run_benchmark(const std::string& graph_name, const std::string& description, const std::vector<std::vector<int>>& adj) {
    std::cout << "---------------------------------------------------------\n";
    std::cout << "--- Test Case: " << graph_name << " ---\n";
    std::cout << "--- Description: " << description << " ---\n";
    std::cout << "---------------------------------------------------------\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    auto result = topological_sort_kahn(adj);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Sort completed in: " << std::fixed << std::setprecision(4) << duration.count() << " seconds.\n";
    std::cout << "Sorted " << result.size() << " nodes.\n\n";
}

int main() {
    int num_nodes = 20000000;
    int edges_per_node = 15;

    // Test 1
    {
        auto adj_easy = generate_easy_dag(num_nodes, edges_per_node);
        run_benchmark("Best Case (Sequential)", "A cache-friendly DAG with perfect data locality", adj_easy);
    }

    // Test 2
    {
        auto adj_challenging = generate_challenging_dag(num_nodes, edges_per_node);
        run_benchmark("Common Case (Narrow Frontier)", "A cache-hostile DAG with a serial workflow", adj_challenging);
    }

    // Test 3
    {
        auto adj_wide = generate_wide_frontier_dag(num_nodes, edges_per_node);
        run_benchmark("Parallel-Ready Case (Wide Frontier)", "A cache-hostile DAG with high initial parallelism", adj_wide);
    }
    
    return 0;
}
