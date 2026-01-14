#pragma once

__global__
void heatStep(const float* current, float* next, int width, int height, float alpha);