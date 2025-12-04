// serial_topo.cpp
// Compile: g++ -O3 -g -march=native serial_topo.cpp -o serial_topo
// Usage: ./serial_topo <graph_file>
// VTune: vtune -collect hotspots ./serial_topo graph_wide.bin

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <immintrin.h>
#include <algorithm>

// =============================================================================
// CSR STRUCTURE
// =============================================================================
struct CSRGraph {
    std::vector<int> row_offsets;
    std::vector<int> flat_adj;
    int n;
};

// =============================================================================
// GRAPH LOADING
// =============================================================================
std::vector<std::vector<int>> load_graph_binary(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "ERROR: Cannot open file: " << filename << std::endl;
        exit(1);
    }
    
    int n;
    in.read(reinterpret_cast<char*>(&n), sizeof(int));
    
    std::vector<std::vector<int>> adj(n);
    
    for (int u = 0; u < n; ++u) {
        int degree;
        in.read(reinterpret_cast<char*>(&degree), sizeof(int));
        
        if (degree > 0) {
            adj[u].resize(degree);
            in.read(reinterpret_cast<char*>(adj[u].data()), degree * sizeof(int));
        }
    }
    
    in.close();
    return adj;
}

// =============================================================================
// CSR CONVERSION
// =============================================================================
CSRGraph convert_to_csr(const std::vector<std::vector<int>>& adj) {
    int n = adj.size();
    CSRGraph g;
    g.n = n;
    g.row_offsets.resize(n + 1);
    
    g.row_offsets[0] = 0;
    for (int i = 0; i < n; ++i) {
        g.row_offsets[i+1] = g.row_offsets[i] + adj[i].size();
    }
    
    g.flat_adj.resize(g.row_offsets[n]);
    
    for (int i = 0; i < n; ++i) {
        if (!adj[i].empty()) {
            int start = g.row_offsets[i];
            std::copy(adj[i].begin(), adj[i].end(), g.flat_adj.begin() + start);
        }
    }
    
    return g;
}

// =============================================================================
// SERIAL TOPOLOGICAL SORT (OPTIMIZED)
// =============================================================================
std::vector<int> topological_sort_serial(const CSRGraph& g) {
    int n = g.n;
    
    // Compute indegrees
    std::vector<int> indegree(n, 0);
    
    for (size_t i = 0; i < g.flat_adj.size(); ++i) {
        indegree[g.flat_adj[i]]++;
    }
    
    // Find initial frontier
    std::vector<int> frontier;
    frontier.reserve(n);
    
    for (int i = 0; i < n; ++i) {
        if (indegree[i] == 0) {
            frontier.push_back(i);
        }
    }
    
    // BFS with Kahn's algorithm
    std::vector<int> order;
    order.reserve(n);
    
    size_t head = 0;
    while (head < frontier.size()) {
        int u = frontier[head++];
        order.push_back(u);
        
        int start = g.row_offsets[u];
        int end = g.row_offsets[u+1];
        
        // Process neighbors with prefetching
        for (int j = start; j < end; ++j) {
            int v = g.flat_adj[j];
            
            // Prefetch indegree array ahead
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
// MAIN - PURE ALGORITHM BENCHMARK
// =============================================================================
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file.bin>\n";
        return 1;
    }
    
    std::string input_file = argv[1];
    
    std::cout << "=================================================\n";
    std::cout << "SERIAL TOPOLOGICAL SORT\n";
    std::cout << "=================================================\n";
    std::cout << "Input: " << input_file << "\n";
    std::cout << "=================================================\n\n";
    
    // PHASE 1: Load graph (not timed)
    std::cout << "[1/3] Loading graph..." << std::flush;
    auto load_start = std::chrono::high_resolution_clock::now();
    auto adj = load_graph_binary(input_file);
    auto load_end = std::chrono::high_resolution_clock::now();
    std::cout << " Done (" 
              << std::chrono::duration<double>(load_end - load_start).count() 
              << "s)\n";
    
    std::cout << "  Loaded " << adj.size() << " nodes\n";
    
    // PHASE 2: Convert to CSR (not timed)
    std::cout << "[2/3] Converting to CSR..." << std::flush;
    auto csr_start = std::chrono::high_resolution_clock::now();
    auto g = convert_to_csr(adj);
    auto csr_end = std::chrono::high_resolution_clock::now();
    std::cout << " Done (" 
              << std::chrono::duration<double>(csr_end - csr_start).count() 
              << "s)\n";
    
    std::cout << "  Total edges: " << g.flat_adj.size() << "\n\n";
    
    // Free original adjacency list to save memory
    adj.clear();
    adj.shrink_to_fit();
    
    // PHASE 3: Run algorithm (TIMED - This is what VTune should profile)
    std::cout << "[3/3] Running topological sort...\n";
    std::cout << "------------------------------------------------\n";
    
    const int NUM_RUNS = 5;
    std::vector<double> times;
    
    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // THE ACTUAL ALGORITHM - VTUNE PROFILE THIS
        auto result = topological_sort_serial(g);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        times.push_back(duration.count());
        
        std::cout << "  Run " << (run + 1) << ": " 
                  << std::fixed << std::setprecision(4) 
                  << duration.count() << "s"
                  << " (sorted " << result.size() << " nodes)\n";
    }
    
    std::cout << "------------------------------------------------\n";
    
    // Statistics
    double min_time = *std::min_element(times.begin(), times.end());
    double max_time = *std::max_element(times.begin(), times.end());
    double avg_time = 0;
    for (double t : times) avg_time += t;
    avg_time /= times.size();
    
    std::cout << "RESULTS:\n";
    std::cout << "  Best:    " << std::fixed << std::setprecision(4) << min_time << "s\n";
    std::cout << "  Worst:   " << max_time << "s\n";
    std::cout << "  Average: " << avg_time << "s\n";
    std::cout << "  Std dev: " << (max_time - min_time) / 2.0 << "s\n";
    std::cout << "=================================================\n";
    
    return 0;
}