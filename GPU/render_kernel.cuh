#pragma once
#include <cuda_runtime.h>

__global__
void renderKernel(uchar4* pixels, int width, int height, float time);
