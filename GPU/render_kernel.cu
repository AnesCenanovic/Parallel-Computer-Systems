#include "render_kernel.cuh"

__global__
void renderKernel(uchar4* pixels, int width, int height, float time)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    unsigned char r = (unsigned char)(127 + 127 * sinf(x * 0.02f + time));
    unsigned char g = (unsigned char)(127 + 127 * sinf(y * 0.02f + time));
    unsigned char b = 128;

    pixels[idx] = make_uchar4(r, g, b, 255);
}
