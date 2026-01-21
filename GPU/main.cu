#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "render_kernel.cuh"

const int WIDTH = 800;
const int HEIGHT = 600;

GLuint pbo = 0;
cudaGraphicsResource* cuda_pbo = nullptr;

void createPBO()
{
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
        WIDTH * HEIGHT * sizeof(uchar4),
        nullptr,
        GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(
        &cuda_pbo,
        pbo,
        cudaGraphicsMapFlagsWriteDiscard
    );
}

void drawPBO()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

int main()
{
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA Heat Room v2", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    glewInit();

    createPBO();

    dim3 block(16, 16);
    dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    float time = 0.0f;

    while (!glfwWindowShouldClose(window))
    {
        uchar4* d_pixels;
        size_t num_bytes;

        cudaGraphicsMapResources(1, &cuda_pbo, 0);
        cudaGraphicsResourceGetMappedPointer(
            (void**)&d_pixels,
            &num_bytes,
            cuda_pbo
        );

        renderKernel<<<grid, block>>>(d_pixels, WIDTH, HEIGHT, time);

        cudaGraphicsUnmapResources(1, &cuda_pbo, 0);

        glClear(GL_COLOR_BUFFER_BIT);
        drawPBO();

        glfwSwapBuffers(window);
        glfwPollEvents();

        time += 0.05f;
    }

    cudaGraphicsUnregisterResource(cuda_pbo);
    glDeleteBuffers(1, &pbo);

    glfwTerminate();
    return 0;
}
