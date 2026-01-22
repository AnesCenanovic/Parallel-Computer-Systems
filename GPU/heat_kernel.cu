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

// Helper function to create a smooth color gradient
__device__
uchar4 temperatureToColor(float t)
{
    // Clamp temperature to [0, 1]
    t = fmaxf(0.0f, fminf(1.0f, t));

    unsigned char r, g, b;

    if (t < 0.25f) {
        // Blue to Cyan (0.0 - 0.25)
        float factor = t / 0.25f;
        r = 0;
        g = (unsigned char)(255 * factor);
        b = 255;
    }
    else if (t < 0.5f) {
        // Cyan to Green (0.25 - 0.5)
        float factor = (t - 0.25f) / 0.25f;
        r = 0;
        g = 255;
        b = (unsigned char)(255 * (1.0f - factor));
    }
    else if (t < 0.75f) {
        // Green to Yellow (0.5 - 0.75)
        float factor = (t - 0.5f) / 0.25f;
        r = (unsigned char)(255 * factor);
        g = 255;
        b = 0;
    }
    else {
        // Yellow to Red to White (0.75 - 1.0)
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

    // Normalize temperature to [0, 1]
    float temp = temperature[idx];
    float normalized = (temp - minTemp) / (maxTemp - minTemp);

    pixels[idx] = temperatureToColor(normalized);
}
