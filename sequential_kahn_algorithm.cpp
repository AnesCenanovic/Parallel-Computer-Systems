#include <iostream>
#include <vector>
#include <queue>
#include <chrono>

std::vector<int> topological_sort_kahn(const std::vector<std::vector<int>>& adj) {
    int n = adj.size();
    std::vector<int> indegree(n, 0);

    // Step 1: Compute in-degrees
    for (int u = 0; u < n; ++u) {
        for (int v : adj[u]) {
            indegree[v]++;
        }
    }

    // Step 2: Queue for vertices with indegree 0
    std::queue<int> q;
    for (int i = 0; i < n; ++i) {
        if (indegree[i] == 0)
            q.push(i);
    }

    // Step 3: Process queue
    std::vector<int> order;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        order.push_back(u);

        for (int v : adj[u]) {
            indegree[v]--;
            if (indegree[v] == 0)
                q.push(v);
        }
    }

    // Step 4: Check for cycles
    if ((int)order.size() != n) {
        std::cerr << "Error: Graph contains a cycle, no topological ordering exists.\n";
        return {};
    }

    return order;
}

std::vector<std::vector<int>> generate_large_dag(int n, int edges_per_node = 10) {
    std::vector<std::vector<int>> adj(n);
    for (int u = 0; u < n; ++u) {
        for (int i = 1; i <= edges_per_node; ++i) {
            int v = u + i;
            if (v < n)
                adj[u].push_back(v);
        }
    }
    return adj;
}

int main() {
    int num_nodes = 10000000;
    auto adj = generate_large_dag(num_nodes);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = topological_sort_kahn(adj);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Topological sort completed in " << duration.count() << " seconds.\n";
    std::cout << "Sorted " << result.size() << " nodes.\n";

    return 0;
}
