#include <iostream>
#include <vector>
#include <queue>

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

int main() {
    // Example: 6 nodes (0–5)
    // 5 → 0, 5 → 2, 4 → 0, 4 → 1, 2 → 3, 3 → 1
    std::vector<std::vector<int>> adj = {
        {},       // 0
        {},       // 1
        {3},      // 2
        {1},      // 3
        {0, 1},   // 4
        {0, 2}    // 5
    };

    std::vector<int> result = topological_sort_kahn(adj);

    std::cout << "Topological order: ";
    for (int v : result)
        std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
