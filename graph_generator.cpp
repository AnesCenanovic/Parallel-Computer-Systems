// graph_generator.cpp
// Compile: g++ -O3 -fopenmp -g graph_generator.cpp -o graph_gen
// Usage: ./graph_gen <type> <nodes> <edges> <output_file>
//   type: easy|narrow|wide|disjoint

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <string>
#include <cstring>

// =============================================================================
// GRAPH GENERATORS
// =============================================================================

std::vector<std::vector<int>> generate_easy_dag(int n, int edges_per_node) {
    std::cout << "Generating EASY DAG (sequential)..." << std::endl;
    std::vector<std::vector<int>> adj(n);
    
    for (int u = 0; u < n; ++u) {
        for (int i = 1; i <= edges_per_node; ++i) {
            if (u + i < n) {
                adj[u].push_back(u + i);
            }
        }
    }
    return adj;
}

std::vector<std::vector<int>> generate_narrow_dag(int n, int avg_edges_per_node) {
    std::cout << "Generating NARROW DAG (random, narrow frontier)..." << std::endl;
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

std::vector<std::vector<int>> generate_wide_dag(int n, int avg_edges_per_node) {
    std::cout << "Generating WIDE DAG (10% sources, wide frontier)..." << std::endl;
    std::vector<std::vector<int>> adj(n);
    std::mt19937 gen(1337);
    
    int first_non_source_node = static_cast<int>(n * 0.1);
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

std::vector<std::vector<int>> generate_disjoint_dag(int n, int edges_per_node) {
    std::cout << "Generating DISJOINT DAG (8 independent clusters)..." << std::endl;
    std::vector<std::vector<int>> adj(n);
    
    int num_clusters = 8;
    int nodes_per_cluster = n / num_clusters;
    
    for (int c = 0; c < num_clusters; ++c) {
        int start = c * nodes_per_cluster;
        int end = start + nodes_per_cluster;
        std::mt19937 gen(1337 + c);
        
        for (int u = start; u < end; ++u) {
            if (u + 1 >= end) continue;
            
            std::uniform_int_distribution<int> dist(u + 1, end - 1);
            std::vector<int> neighbors;
            
            for (int i = 0; i < edges_per_node; ++i) {
                neighbors.push_back(dist(gen));
            }
            
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
            adj[u] = neighbors;
        }
    }
    return adj;
}

// =============================================================================
// BINARY FORMAT I/O (Compact & Fast)
// =============================================================================

void save_graph_binary(const std::vector<std::vector<int>>& adj, const std::string& filename) {
    std::cout << "Saving graph to " << filename << "..." << std::endl;
    
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "ERROR: Cannot open file for writing: " << filename << std::endl;
        exit(1);
    }
    
    // Header: num_nodes
    int n = adj.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(int));
    
    // For each node: num_neighbors, then neighbor_ids
    for (int u = 0; u < n; ++u) {
        int degree = adj[u].size();
        out.write(reinterpret_cast<const char*>(&degree), sizeof(int));
        
        if (degree > 0) {
            out.write(reinterpret_cast<const char*>(adj[u].data()), degree * sizeof(int));
        }
    }
    
    out.close();
    
    // Print statistics
    long long total_edges = 0;
    for (const auto& neighbors : adj) {
        total_edges += neighbors.size();
    }
    
    std::cout << "Graph saved successfully!" << std::endl;
    std::cout << "  Nodes: " << n << std::endl;
    std::cout << "  Edges: " << total_edges << std::endl;
    std::cout << "  Avg degree: " << (double)total_edges / n << std::endl;
    std::cout << "  File size: " << (n * 4 + total_edges * 4) / (1024.0 * 1024.0) << " MB" << std::endl;
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <type> <nodes> <edges_per_node> <output_file>\n";
        std::cerr << "  type: easy | narrow | wide | disjoint\n";
        std::cerr << "Example: " << argv[0] << " wide 20000000 15 graph_wide.bin\n";
        return 1;
    }
    
    std::string type = argv[1];
    int n = std::stoi(argv[2]);
    int edges = std::stoi(argv[3]);
    std::string output = argv[4];
    
    std::cout << "=================================================\n";
    std::cout << "GRAPH GENERATOR\n";
    std::cout << "=================================================\n";
    std::cout << "Type: " << type << "\n";
    std::cout << "Nodes: " << n << "\n";
    std::cout << "Edges per node: " << edges << "\n";
    std::cout << "Output: " << output << "\n";
    std::cout << "=================================================\n\n";
    
    std::vector<std::vector<int>> adj;
    
    if (type == "easy") {
        adj = generate_easy_dag(n, edges);
    } else if (type == "narrow") {
        adj = generate_narrow_dag(n, edges);
    } else if (type == "wide") {
        adj = generate_wide_dag(n, edges);
    } else if (type == "disjoint") {
        adj = generate_disjoint_dag(n, edges);
    } else {
        std::cerr << "ERROR: Unknown graph type: " << type << "\n";
        return 1;
    }
    
    save_graph_binary(adj, output);
    
    std::cout << "\nGraph generation complete!\n";
    return 0;
}