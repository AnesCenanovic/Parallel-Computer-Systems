#include <iostream>
#include <vector>
#include <cmath>
#include <chrono> 

// cube size, number of time steps
const int N = 128; 
const int STEPS = 100; 

// diffusion constant
const float ALPHA = 0.1f; 

// helper function to convert 3D indices to 1D
int getIdx(int x, int y, int z) {
    return z * N * N + y * N + x;
}

void cpuHeatDissipation(float* input, float* output) {
    for (int z = 1; z < N - 1; z++) {
        for (int y = 1; y < N - 1; y++) {
            for (int x = 1; x < N - 1; x++) {
                
                int c = getIdx(x, y, z); 
                
                // get neighbor indices
                int up = getIdx(x, y - 1, z);
                int down = getIdx(x, y + 1, z);
                int left = getIdx(x - 1, y, z);
                int right = getIdx(x + 1, y, z);
                int front = getIdx(x, y, z - 1);
                int back = getIdx(x, y, z + 1);

                // Formula: T_new = T_old + alpha * (Sum_Neighbors - 6 * T_old)
                float sumNeighbors = input[up] + input[down] + input[left] + 
                                     input[right] + input[front] + input[back];
                
                output[c] = input[c] + ALPHA * (sumNeighbors - 6.0f * input[c]);
            }
        }
    }
}

//int main() {
//    size_t size = N * N * N * sizeof(float);
//    int numElements = N * N * N;
//
//    std::vector<float> h_in(numElements);
//    std::vector<float> h_out(numElements);
//
//    // initialize input array to zero
//    for (int i = 0; i < numElements; i++) h_in[i] = 0.0f;
//    
//    // set a hot spot in the center
//    int center = getIdx(N/2, N/2, N/2);
//    h_in[center] = 1000.0f; 
//    
//    std::cout << "Beginning cpu simulation: " << N << "x" << N << "x" << N << std::endl;
//
//    auto start = std::chrono::high_resolution_clock::now();
//
//    float* inPtr = h_in.data();
//    float* outPtr = h_out.data();
//
//    for (int i = 0; i < STEPS; i++) {
//        cpuHeatDissipation(inPtr, outPtr);
//        
//        std::swap(inPtr, outPtr);
//    }
//
//    auto end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> elapsed = end - start;
//
//    std::cout << "Simulation finished." << std::endl;
//    std::cout << "Execution time (CPU): " << elapsed.count() << " seconds." << std::endl;
//    
//    return 0;
//}