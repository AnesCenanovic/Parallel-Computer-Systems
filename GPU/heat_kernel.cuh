#pragma once
#include <cuda_runtime.h>

// Heat simulation kernel
__global__
void heatStep(const float* current, float* next, int width, int height, float alpha);

// Visualization kernel - converts temperature to color
__global__
void visualizeHeat(const float* temperature, uchar4* pixels, int width, int height, float minTemp, float maxTemp);

__global__ void heatStepShared(const float* current, float* next, int width, int height, float alpha);