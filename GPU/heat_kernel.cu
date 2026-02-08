// heat_kernel.cu - Optimized version with shared memory
#include "heat_kernel.cuh"

// Define block size for shared memory version
#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE + 2)  // 18
#define PADDED_TILE (TILE_SIZE + 1)

// ============================================================================
// ORIGINAL KERNEL (for comparison)
// ============================================================================
__global__
void heatStep(const float* current, float* next, int width, int height, float alpha)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
        return;

    int idx = y * width + x;

    float center = current[idx];
    float up = current[(y - 1) * width + x];
    float down = current[(y + 1) * width + x];
    float left = current[y * width + (x - 1)];
    float right = current[y * width + (x + 1)];

    next[idx] = center + alpha * (up + down + left + right - 4.0f * center);
}

// ============================================================================
// OPTIMIZED KERNEL - SHARED MEMORY VERSION
// ============================================================================
__global__
void heatStepShared(const float* current, float* next, int width, int height, float alpha)
{
    // Shared memory tile with halo region
    __shared__ float tile[TILE_SIZE][PADDED_TILE];

    // Global thread indices
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Local thread indices within tile (offset by 1 for halo)
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // Load center of tile from global memory
    if (x < width && y < height) {
        tile[ty][tx] = current[y * width + x];
    }
    else {
        tile[ty][tx] = 0.0f;  // Boundary padding
    }

    // Load halo regions (edges of tile)
    // Top halo
    if (threadIdx.y == 0 && y > 0) {
        tile[0][tx] = current[(y - 1) * width + x];
    }
    // Bottom halo
    if (threadIdx.y == BLOCK_SIZE - 1 && y < height - 1) {
        tile[ty + 1][tx] = current[(y + 1) * width + x];
    }
    // Left halo
    if (threadIdx.x == 0 && x > 0) {
        tile[ty][0] = current[y * width + (x - 1)];
    }
    // Right halo
    if (threadIdx.x == BLOCK_SIZE - 1 && x < width - 1) {
        tile[ty][tx + 1] = current[y * width + (x + 1)];
    }

    // Synchronize to ensure all data is loaded into shared memory
    __syncthreads();

    // Compute only for interior points (avoid boundaries)
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Access neighbors from shared memory (fast!)
        float center = tile[ty][tx];
        float up = tile[ty - 1][tx];
        float down = tile[ty + 1][tx];
        float left = tile[ty][tx - 1];
        float right = tile[ty][tx + 1];

        // Same heat equation as before
        next[y * width + x] = center + alpha * (up + down + left + right - 4.0f * center);
    }
}

// ============================================================================
// VISUALIZATION KERNELS (unchanged)
// ============================================================================

__device__
uchar4 temperatureToColor(float t)
{
    t = fmaxf(0.0f, fminf(1.0f, t));

    unsigned char r, g, b;

    if (t < 0.25f) {
        float factor = t / 0.25f;
        r = 0;
        g = (unsigned char)(255 * factor);
        b = 255;
    }
    else if (t < 0.5f) {
        float factor = (t - 0.25f) / 0.25f;
        r = 0;
        g = 255;
        b = (unsigned char)(255 * (1.0f - factor));
    }
    else if (t < 0.75f) {
        float factor = (t - 0.5f) / 0.25f;
        r = (unsigned char)(255 * factor);
        g = 255;
        b = 0;
    }
    else {
        float factor = (t - 0.75f) / 0.25f;
        r = 255;
        g = (unsigned char)(255 * (1.0f - factor * 0.5f));
        b = (unsigned char)(255 * factor);
    }

    return make_uchar4(r, g, b, 255);
}

__global__
void visualizeHeat(const float* temperature, uchar4* pixels, int width, int height, float minTemp, float maxTemp)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float temp = temperature[idx];
    float normalized = (temp - minTemp) / (maxTemp - minTemp);

    pixels[idx] = temperatureToColor(normalized);
}
