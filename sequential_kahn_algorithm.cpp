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
    std::vector<int> indegree(n, 0); // vector of indegrees for each node
    for (const auto& neighbors : adj) { // iterate through adjacencies for each node
        for (int v : neighbors) { // for each individidual adjacency of that node
            indegree[v]++; // increment indegree of that neighbor node
        }
    }
    std::queue<int> q; // queue of all indegree 0 nodes
    for (int i = 0; i < n; ++i) { // for each of those nodes
        if (indegree[i] == 0) { // if indegree zero
            q.push(i); // push it to processing queue
        }
    }
    std::vector<int> order; // final order vector
    order.reserve(n); // reserve space for n nodes to avoid resizing later
    while (!q.empty()) { // process queue until no more nodes
        int u = q.front(); // take node in front of queue
        q.pop();
        order.push_back(u); // put it in the order
        for (int v : adj[u]) { // for each outbound for that node
            indegree[v]--; // decrease indegree for all other nodes
            if (indegree[v] == 0) { // if they now have no indegrees
                q.push(v); // push them to processing queue 
            }
        }
    }
    return order; // final topological order of n nodes
}

// Best Case - Sequential cache-friendly DAG
std::vector<std::vector<int>> generate_easy_dag(int n, int edges_per_node) {
    std::vector<std::vector<int>> adj(n); // adjacency list for n nodes
    for (int u = 0; u < n; ++u) { // for each node
        for (int i = 1; i <= edges_per_node; ++i) { // create fixed number of outgoing edges
            int v = u + i; // destination node 'v' is i offset from source node 'u'
            if (v < n) { // ensure graph bound
                adj[u].push_back(v); // add directed edge from source to destination node
            }
        }
    }
    return adj;
}

// Common Case - Cache-hostile DAG with a narrow workflow
std::vector<std::vector<int>> generate_challenging_dag(int n, int avg_edges_per_node) {
    std::vector<std::vector<int>> adj(n); // adjacency list for n nodes
    std::mt19937 gen(1337); // RNG with fixed seed for reproducible results

    for (int u = 0; u < n; ++u) { // iterate through each node
        if (u + 1 >= n) continue; // node must connected to node with higher index, skip last node
        std::uniform_int_distribution<int> dist(u + 1, n - 1); // distribution to generate random integers in u+1, n-1 range
        
        std::vector<int> neighbors; // list for radnomly selected neighbors
        neighbors.reserve(avg_edges_per_node); // reserve space for maximum amount of edges
        for (int i = 0; i < avg_edges_per_node; ++i) { // generate list of random neighbors for node
            neighbors.push_back(dist(gen)); // pick a random node and add it to the list
        }
        std::sort(neighbors.begin(), neighbors.end()); // sort list of neighbors
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end()); // shuffle duplicates to end and delete them
        adj[u] = neighbors; // assign final clean list to current node
    }
    return adj;
}

// Parallel-Ready Case - Cache-hostile DAG with a wide workflow
std::vector<std::vector<int>> generate_wide_frontier_dag(int n, int avg_edges_per_node, double source_node_ratio = 0.1) {
    std::vector<std::vector<int>> adj(n); // adjacency list for n nodes
    std::mt19937 gen(1337); // RNG with fixed seed for reproducible results

    int first_non_source_node = static_cast<int>(n * source_node_ratio); // calculate amount of starting nodes
    if (first_non_source_node <= 0) first_non_source_node = 1; // safety in case of small n

    for (int u = first_non_source_node; u < n; ++u) { // same as previous DAG
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
