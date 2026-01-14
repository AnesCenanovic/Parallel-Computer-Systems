#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#include "heat_kernel.cuh"
#include "visualization.h"

int main() {
    const int width = 512;
    const int height = 512;
    const float alpha = 0.1f;
    const int steps = 500;

    size_t size = width * height * sizeof(float);

    // CPU memory
    std::vector<float> h_current(width * height, 0.0f);
    std::vector<float> h_next(width * height, 0.0f);

    // Initial hot spot
    h_current[(height/2) * width + (width/2)] = 100.0f;

    // GPU memory
    float *d_current, *d_next;
    cudaMalloc(&d_current, size);
    cudaMalloc(&d_next, size);

    cudaMemcpy(d_current, h_current.data(), size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    for (int i = 0; i < steps; i++) {
        heatStep<<<grid, block>>>(d_current, d_next, width, height, alpha);
        std::swap(d_current, d_next);
    }

    cudaMemcpy(h_current.data(), d_current, size, cudaMemcpyDeviceToHost);

    saveHeatMap(h_current.data(), width, height, "heat.ppm");

    cudaFree(d_current);
    cudaFree(d_next);

    std::cout << "Simulation complete. Image saved.\n";
    return 0;
}
