#include "heat_kernel.cuh"

__global__
void heatStep(const float* current, float* next, int width, int height, float alpha)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
        return;

    int idx = y * width + x;

    float center = current[idx];
    float up     = current[(y - 1) * width + x];
    float down   = current[(y + 1) * width + x];
    float left   = current[y * width + (x - 1)];
    float right  = current[y * width + (x + 1)];

    next[idx] = center + alpha * (up + down + left + right - 4.0f * center);
}
