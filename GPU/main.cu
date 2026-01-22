#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "heat_kernel.cuh"

const int WIDTH = 800;
const int HEIGHT = 600;

// Simulation parameters
float ALPHA = 0.2f;                 // Diffusion coefficient (adjustable)
const float MIN_TEMP = 0.0f;
const float MAX_TEMP = 100.0f;
const float HEAT_SOURCE_TEMP = 100.0f;
int HEAT_SOURCE_RADIUS = 15;

// OpenGL resources
GLuint pbo = 0;
GLuint texture = 0;
cudaGraphicsResource* cuda_pbo = nullptr;

// CUDA temperature buffers
float* d_temp_current = nullptr;
float* d_temp_next = nullptr;

// Simulation state
enum SimulationMode {
    MODE_INTERACTIVE,
    MODE_CORNER_RADIATOR,
    MODE_CENTRAL_HEATING,
    MODE_HOT_WALLS,
    MODE_MULTIPLE_SOURCES
};

SimulationMode currentMode = MODE_INTERACTIVE;
bool isPaused = false;
bool mousePressed = false;
double mouseX = 0.0, mouseY = 0.0;

// Performance metrics
int frameCount = 0;
double lastFpsTime = 0.0;
double currentFps = 0.0;
float minTemp = 0.0f, maxTemp = 0.0f, avgTemp = 0.0f;

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) 
                  << " at " << file << ":" << line << std::endl;
    }
}

void printControls()
{
    std::cout << "\n+========================================================+\n";
    std::cout << "|          HEAT DIFFUSION - KEYBOARD CONTROLS           |\n";
    std::cout << "+========================================================+\n";
    std::cout << "| SPACE      : Pause/Resume simulation                  |\n";
    std::cout << "| R          : Reset temperature field                  |\n";
    std::cout << "| 1          : Interactive Mode (click to add heat)     |\n";
    std::cout << "| 2          : Corner Radiator Mode                     |\n";
    std::cout << "| 3          : Central Heating Mode                     |\n";
    std::cout << "| 4          : Hot Walls Mode                           |\n";
    std::cout << "| 5          : Multiple Heat Sources Mode               |\n";
    std::cout << "| +/=        : Increase diffusion rate                  |\n";
    std::cout << "| -/_        : Decrease diffusion rate                  |\n";
    std::cout << "| [          : Decrease heat source size                |\n";
    std::cout << "| ]          : Increase heat source size                |\n";
    std::cout << "| ESC        : Exit                                     |\n";
    std::cout << "+========================================================+\n\n";
}

void printStats()
{
    std::cout << "\r[FPS: " << std::fixed << std::setprecision(1) << currentFps 
              << " | Alpha: " << std::setprecision(3) << ALPHA
              << " | Temp: Min=" << std::setprecision(1) << minTemp 
              << "C Max=" << maxTemp << "C Avg=" << avgTemp << "C"
              << " | Mode: ";
    
    switch(currentMode) {
        case MODE_INTERACTIVE: std::cout << "Interactive"; break;
        case MODE_CORNER_RADIATOR: std::cout << "Corner Radiator"; break;
        case MODE_CENTRAL_HEATING: std::cout << "Central Heating"; break;
        case MODE_HOT_WALLS: std::cout << "Hot Walls"; break;
        case MODE_MULTIPLE_SOURCES: std::cout << "Multiple Sources"; break;
    }
    
    if (isPaused) std::cout << " | PAUSED";
    std::cout << "]" << std::flush;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS) {
        switch(key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_SPACE:
                isPaused = !isPaused;
                std::cout << "\n[Simulation " << (isPaused ? "PAUSED" : "RESUMED") << "]\n";
                break;
            case GLFW_KEY_R:
                cudaMemset(d_temp_current, 0, WIDTH * HEIGHT * sizeof(float));
                cudaMemset(d_temp_next, 0, WIDTH * HEIGHT * sizeof(float));
                std::cout << "\n[Temperature field RESET]\n";
                break;
            case GLFW_KEY_1:
                currentMode = MODE_INTERACTIVE;
                std::cout << "\n[Mode: INTERACTIVE - Click to add heat]\n";
                break;
            case GLFW_KEY_2:
                currentMode = MODE_CORNER_RADIATOR;
                std::cout << "\n[Mode: CORNER RADIATOR]\n";
                break;
            case GLFW_KEY_3:
                currentMode = MODE_CENTRAL_HEATING;
                std::cout << "\n[Mode: CENTRAL HEATING]\n";
                break;
            case GLFW_KEY_4:
                currentMode = MODE_HOT_WALLS;
                std::cout << "\n[Mode: HOT WALLS]\n";
                break;
            case GLFW_KEY_5:
                currentMode = MODE_MULTIPLE_SOURCES;
                std::cout << "\n[Mode: MULTIPLE HEAT SOURCES]\n";
                break;
            case GLFW_KEY_EQUAL:
            case GLFW_KEY_KP_ADD:
                ALPHA = fminf(ALPHA + 0.05f, 0.25f);  // Cap at 0.25 for stability
                std::cout << "\n[Diffusion rate increased: " << ALPHA << "]\n";
                break;
            case GLFW_KEY_MINUS:
            case GLFW_KEY_KP_SUBTRACT:
                ALPHA = fmaxf(ALPHA - 0.05f, 0.01f);
                std::cout << "\n[Diffusion rate decreased: " << ALPHA << "]\n";
                break;
            case GLFW_KEY_LEFT_BRACKET:
                HEAT_SOURCE_RADIUS = fmaxf(HEAT_SOURCE_RADIUS - 5, 5);
                std::cout << "\n[Heat source radius: " << HEAT_SOURCE_RADIUS << "]\n";
                break;
            case GLFW_KEY_RIGHT_BRACKET:
                HEAT_SOURCE_RADIUS = fminf(HEAT_SOURCE_RADIUS + 5, 50);
                std::cout << "\n[Heat source radius: " << HEAT_SOURCE_RADIUS << "]\n";
                break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mousePressed = (action == GLFW_PRESS);
    }
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    mouseX = xpos;
    mouseY = ypos;
}

__global__
void addHeatSource(float* temperature, int width, int height, 
                   int centerX, int centerY, int radius, float temp)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int dx = x - centerX;
    int dy = y - centerY;
    float dist = sqrtf((float)(dx * dx + dy * dy));

    if (dist < radius) {
        int idx = y * width + x;
        float factor = 1.0f - (dist / radius);
        temperature[idx] = fmaxf(temperature[idx], temp * factor);
    }
}

__global__
void addRectHeatSource(float* temperature, int width, int height,
                       int x1, int y1, int x2, int y2, float temp)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        temperature[y * width + x] = temp;
    }
}

int selectCudaDevice()
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA devices found\n";
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    cudaSetDevice(0);
    return 0;
}

void initSimulation()
{
    size_t size = WIDTH * HEIGHT * sizeof(float);
    cudaCheckError(cudaMalloc(&d_temp_current, size));
    cudaCheckError(cudaMalloc(&d_temp_next, size));
    cudaCheckError(cudaMemset(d_temp_current, 0, size));
    cudaCheckError(cudaMemset(d_temp_next, 0, size));
}

void createPBO()
{
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(uchar4), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaCheckError(cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard));
}

void drawColorLegend()
{
    const int legendWidth = 200, legendHeight = 150;
    const int legendX = WIDTH - legendWidth - 20, legendY = HEIGHT - legendHeight - 20;
    
    glDisable(GL_TEXTURE_2D);
    glColor4f(0.0f, 0.0f, 0.0f, 0.7f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glBegin(GL_QUADS);
    glVertex2f(legendX - 10, legendY - 10);
    glVertex2f(legendX + legendWidth + 10, legendY - 10);
    glVertex2f(legendX + legendWidth + 10, legendY + legendHeight + 10);
    glVertex2f(legendX - 10, legendY + legendHeight + 10);
    glEnd();
    
    float colors[6][3] = {
        {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}
    };
    
    float bandHeight = legendHeight / 6.0f;
    for (int i = 0; i < 6; i++) {
        glBegin(GL_QUADS);
        glColor3fv(colors[i]);
        glVertex2f(legendX, legendY + i * bandHeight);
        glVertex2f(legendX + legendWidth, legendY + i * bandHeight);
        if (i < 5) glColor3fv(colors[i + 1]);
        glVertex2f(legendX + legendWidth, legendY + (i + 1) * bandHeight);
        glVertex2f(legendX, legendY + (i + 1) * bandHeight);
        glEnd();
    }
    
    glDisable(GL_BLEND);
}

void drawPBO()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glEnable(GL_TEXTURE_2D);
    glColor3f(1.0f, 1.0f, 1.0f);
    
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f((float)WIDTH, 0.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f((float)WIDTH, (float)HEIGHT);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, (float)HEIGHT);
    glEnd();
    
    glBindTexture(GL_TEXTURE_2D, 0);
}

void applyScenarioHeatSources(dim3 grid, dim3 block)
{
    switch(currentMode) {
        case MODE_INTERACTIVE:
            if (mousePressed) {
                // Don't flip Y - texture coordinates are already flipped in drawPBO
                addHeatSource<<<grid, block>>>(d_temp_current, WIDTH, HEIGHT, 
                    (int)mouseX, (int)mouseY, HEAT_SOURCE_RADIUS, HEAT_SOURCE_TEMP);
            }
            break;
        case MODE_CORNER_RADIATOR:
            addRectHeatSource<<<grid, block>>>(d_temp_current, WIDTH, HEIGHT, 10, 10, 50, 100, HEAT_SOURCE_TEMP);
            break;
        case MODE_CENTRAL_HEATING:
            addHeatSource<<<grid, block>>>(d_temp_current, WIDTH, HEIGHT, WIDTH/2, HEIGHT/2, 40, HEAT_SOURCE_TEMP);
            break;
        case MODE_HOT_WALLS:
            addRectHeatSource<<<grid, block>>>(d_temp_current, WIDTH, HEIGHT, 0, 0, WIDTH-1, 5, HEAT_SOURCE_TEMP * 0.8f);
            addRectHeatSource<<<grid, block>>>(d_temp_current, WIDTH, HEIGHT, 0, HEIGHT-6, WIDTH-1, HEIGHT-1, HEAT_SOURCE_TEMP * 0.8f);
            addRectHeatSource<<<grid, block>>>(d_temp_current, WIDTH, HEIGHT, 0, 0, 5, HEIGHT-1, HEAT_SOURCE_TEMP * 0.8f);
            addRectHeatSource<<<grid, block>>>(d_temp_current, WIDTH, HEIGHT, WIDTH-6, 0, WIDTH-1, HEIGHT-1, HEAT_SOURCE_TEMP * 0.8f);
            break;
        case MODE_MULTIPLE_SOURCES:
            addHeatSource<<<grid, block>>>(d_temp_current, WIDTH, HEIGHT, WIDTH/4, HEIGHT/4, 25, HEAT_SOURCE_TEMP);
            addHeatSource<<<grid, block>>>(d_temp_current, WIDTH, HEIGHT, 3*WIDTH/4, HEIGHT/4, 25, HEAT_SOURCE_TEMP);
            addHeatSource<<<grid, block>>>(d_temp_current, WIDTH, HEIGHT, WIDTH/4, 3*HEIGHT/4, 25, HEAT_SOURCE_TEMP);
            addHeatSource<<<grid, block>>>(d_temp_current, WIDTH, HEIGHT, 3*WIDTH/4, 3*HEIGHT/4, 25, HEAT_SOURCE_TEMP);
            break;
    }
}

void simulateAndVisualize()
{
    dim3 block(16, 16);
    dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    if (!isPaused) {
        applyScenarioHeatSources(grid, block);
        heatStep<<<grid, block>>>(d_temp_current, d_temp_next, WIDTH, HEIGHT, ALPHA);
        
        float* temp = d_temp_current;
        d_temp_current = d_temp_next;
        d_temp_next = temp;
    }

    uchar4* d_pixels;
    size_t num_bytes;
    cudaCheckError(cudaGraphicsMapResources(1, &cuda_pbo, 0));
    cudaCheckError(cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &num_bytes, cuda_pbo));
    visualizeHeat<<<grid, block>>>(d_temp_current, d_pixels, WIDTH, HEIGHT, MIN_TEMP, MAX_TEMP);
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaGraphicsUnmapResources(1, &cuda_pbo, 0));
}

void updatePerformanceMetrics(double currentTime)
{
    frameCount++;
    if (currentTime - lastFpsTime >= 0.5) {
        currentFps = frameCount / (currentTime - lastFpsTime);
        frameCount = 0;
        lastFpsTime = currentTime;
        
        // Sample temperatures across the entire grid (not just top-left)
        const int sampleSize = 1000;
        float h_temps[1000];
        int step = (WIDTH * HEIGHT) / sampleSize;
        
        // Copy sampled temperatures
        for (int i = 0; i < sampleSize; i++) {
            cudaMemcpy(&h_temps[i], d_temp_current + i * step, sizeof(float), cudaMemcpyDeviceToHost);
        }
        
        minTemp = maxTemp = h_temps[0];
        avgTemp = 0.0f;
        for (int i = 0; i < sampleSize; i++) {
            minTemp = fminf(minTemp, h_temps[i]);
            maxTemp = fmaxf(maxTemp, h_temps[i]);
            avgTemp += h_temps[i];
        }
        avgTemp /= (float)sampleSize;
        printStats();
    }
}

int main()
{
    std::cout << "============================================================\n";
    std::cout << "=     CUDA HEAT DIFFUSION VISUALIZATION v2.0               =\n";
    std::cout << "=     Professional Edition - GPU Accelerated               =\n";
    std::cout << "============================================================\n\n";

    if (selectCudaDevice() < 0) return -1;

    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Heat Diffusion - CUDA Professional", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return -1;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIDTH, 0, HEIGHT, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, WIDTH, HEIGHT);

    initSimulation();
    createPBO();
    printControls();

    lastFpsTime = glfwGetTime();

    while (!glfwWindowShouldClose(window))
    {
        simulateAndVisualize();
        updatePerformanceMetrics(glfwGetTime());

        glClear(GL_COLOR_BUFFER_BIT);
        drawPBO();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    if (cuda_pbo) cudaGraphicsUnregisterResource(cuda_pbo);
    if (pbo) glDeleteBuffers(1, &pbo);
    if (texture) glDeleteTextures(1, &texture);
    if (d_temp_current) cudaFree(d_temp_current);
    if (d_temp_next) cudaFree(d_temp_next);

    glfwTerminate();
    cudaDeviceReset();
    return 0;
}
