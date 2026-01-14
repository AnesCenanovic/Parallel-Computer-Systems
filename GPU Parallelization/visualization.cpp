#include "visualization.h"
#include <fstream>
#include <algorithm>

void saveHeatMap(const float* data, int width, int height,
    const char* filename)
{
    float minT = *std::min_element(data, data + width * height);
    float maxT = *std::max_element(data, data + width * height);

    std::ofstream file(filename);
    file << "P3\n" << width << " " << height << "\n255\n";

    for (int i = 0; i < width * height; i++) {
        float normalized = (data[i] - minT) / (maxT - minT);
        int value = static_cast<int>(normalized * 255);

        // grayscale
        file << value << " " << value << " " << value << "\n";
    }
}
