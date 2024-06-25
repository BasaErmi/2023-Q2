
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include<iostream>
#include<vector>
#include<iostream>
#include <cudnn.h>
#include <chrono>

using namespace std;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int width, height, channels;//高、宽、通道数

float* read_img(const char* img_path) {
    unsigned char* data = stbi_load(img_path, &width, &height, &channels, 0);
    /*float* result = (float*)malloc(width * height * channels * sizeof(float));*/
    float* result = new float[width * height * channels];
    if (data) {
        std::cout << "width: " << width << ", height: " << height;
        // 处理图像数据，例如访问某个像素的颜色值
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixelIndex = (y * width + x) * channels;
                for (int c = 0; c < channels; c++) {
                    result[pixelIndex + c] = static_cast<int>(data[pixelIndex + c]);
                }
            }
        }
        // 释放图像数据
        stbi_image_free(data);
    }
    else {
        std::cerr << "Failed to load image" << std::endl;
    }
    return result;
}

float* pad_image(const float* img, int width, int height, int channels, int newWidth, int newHeight, float fillValue) {
    if (newWidth < width || newHeight < height) {
        // 新尺寸不能小于原图像尺寸
        return nullptr;
    }
    int padX = (newWidth - width) / 2;  // 水平填充的数量
    int padY = (newHeight - height) / 2; // 垂直填充的数量
    // 创建新的填充后的图像数组
    float* newImg = new float[newWidth * newHeight * channels];
    // 初始化新图像数组为填充值
    std::fill(newImg, newImg + newWidth * newHeight * channels, fillValue);
    // 将原图像数据复制到新图像数组中
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                int oldIndex = (y * width + x) * channels + c;
                int newIndex = ((y + padY) * newWidth + (x + padX)) * channels + c;
                newImg[newIndex] = img[oldIndex];
            }
        }
    }
    return newImg;
}

__global__ void convolutionKernel(const float* input, float* output, const float* kernel, 
                        int width, int height, int channels, int paddedWidth, int paddedHeight, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && c < channels) {
        float sum = 0.0f;
        for (int kc = 0; kc < channels; ++kc) {
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    int inX = x * stride + kx;
                    int inY = y * stride + ky;
                    if (inX >= 0 && inX < paddedWidth && inY >= 0 && inY < paddedHeight) {
                        sum += input[(inY * paddedWidth + inX) * channels + kc] * kernel[c * 3 * 3 * 3 + (kc * 3 + ky) * 3 + kx];
                    }
                }
            }
        }
        output[(y * width + x) * channels + c] = sum;
    }
}

void convolutionWithDifferentStrides(const float* input, float* output, const float* kernel, int width, int height, int channels,\
                                        int stride, int block_size) {
    int newHeight = (height - 1) * stride + 3;
    int newWidth = (width - 1) * stride + 3;

    float* d_input, * d_output, * d_kernel;
    cudaMalloc((void**)&d_input, newWidth * newHeight * channels * sizeof(float));
    cudaMalloc(&d_output, width * height * channels * sizeof(float));
    cudaMalloc(&d_kernel, 3 * 3 * 3 * 3 * sizeof(float));

    float* h_paddedInput = pad_image(input, width, height, channels, newWidth, newHeight, 0);

    cudaMemcpy(d_input, h_paddedInput, newWidth * newHeight * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, 3 * 3 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(block_size, block_size, 3);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);

    convolutionKernel << <gridSize, blockSize >> > (d_input, d_output, d_kernel, width, height, channels, newWidth, newHeight, stride);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, width * height * channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] h_paddedInput;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

__global__ void convolutionKernelShared(const float* input, float* output, const float* kernel,
    int width, int height, int channels, int paddedWidth, int paddedHeight, int stride) {
    extern __shared__ float sharedMem[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    int sharedWidth = blockDim.x + 2;
    int sharedHeight = blockDim.y + 2;

    // 计算共享内存中每个线程的索引
    int sharedIdx = (threadIdx.y + 1) * sharedWidth + (threadIdx.x + 1);

    if (x < width && y < height && c < channels) {
        // 加载输入数据到共享内存
        sharedMem[sharedIdx] = input[(y * stride * paddedWidth + x * stride) * channels + c];

        // 加载边缘数据到共享内存
        if (threadIdx.x == 0) {
            sharedMem[sharedIdx - 1] = input[(y * stride * paddedWidth + (x * stride - 1)) * channels + c];
        }
        if (threadIdx.x == blockDim.x - 1) {
            sharedMem[sharedIdx + 1] = input[(y * stride * paddedWidth + (x * stride + 1)) * channels + c];
        }
        if (threadIdx.y == 0) {
            sharedMem[sharedIdx - sharedWidth] = input[((y * stride - 1) * paddedWidth + x * stride) * channels + c];
        }
        if (threadIdx.y == blockDim.y - 1) {
            sharedMem[sharedIdx + sharedWidth] = input[((y * stride + 1) * paddedWidth + x * stride) * channels + c];
        }

        __syncthreads();

        float sum = 0.0f;
        for (int kc = 0; kc < channels; ++kc) {
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    int inX = threadIdx.x + kx;
                    int inY = threadIdx.y + ky;
                    float inputVal = sharedMem[(inY * sharedWidth + inX)];
                    float kernelVal = kernel[c * 3 * 3 * 3 + (kc * 3 + ky) * 3 + kx];
                    sum += inputVal * kernelVal;
                }
            }
        }
        output[(y * width + x) * channels + c] = sum;
    }
}

void convolutionWithDifferentStridesShared(const float* input, float* output, const float* kernel, int width, int height, int channels, \
    int stride, int block_size) {
    int newHeight = (height - 1) * stride + 3;
    int newWidth = (width - 1) * stride + 3;

    float* d_input, * d_output, * d_kernel;
    cudaMalloc((void**)&d_input, newWidth * newHeight * channels * sizeof(float));
    cudaMalloc(&d_output, width * height * channels * sizeof(float));
    cudaMalloc(&d_kernel, 3 * 3 * 3 * 3 * sizeof(float));

    float* h_paddedInput = pad_image(input, width, height, channels, newWidth, newHeight, 0);

    cudaMemcpy(d_input, h_paddedInput, newWidth * newHeight * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, 3 * 3 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(block_size, block_size, 3);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);

    convolutionKernelShared << <gridSize, blockSize >> > (d_input, d_output, d_kernel, width, height, channels, newWidth, newHeight, stride);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, width * height * channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] h_paddedInput;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

__global__ void img2col(float* input, float* output, int h, int w, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = 3;  // 卷积核大小

    int n = (h - k) / stride + 1;  // 输出矩阵的行数
    int m = (w - k) / stride + 1;  // 输出矩阵的列数
    int total_size = n * m * k * k * 3;

    if (idx < total_size) {
        int channel = idx % 3;
        int kx = (idx / 3) % k;
        int ky = (idx / 3 / k) % k;
        int w_out_idx = (idx / 3 / k / k) % m;
        int h_out_idx = (idx / 3 / k / k / m) % n;

        int w_in_idx = w_out_idx * stride + kx;
        int h_in_idx = h_out_idx * stride + ky;

        output[idx] = input[(h_in_idx * w + w_in_idx) * 3 + channel];
    }
}

__global__ void matrixMulKernel(float* A, float* B, float* C, int size_1, int size_2, int size_3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size_1 && col < size_3) {
        float value = 0.0f;
        for (int k = 0; k < size_2; k++) {
            value += A[row * size_2 + k] * B[k * size_3 + col];
        }
        C[row * size_3 + col] = value;
    }
}

void convolutionImg2Col(const float* input, float* output, const float* kernel, int width, int height, int channels, \
    int stride, int block_size) {
    int newHeight = (height - 1) * stride + 3;
    int newWidth = (width - 1) * stride + 3;

    float* d_input, * d_output, * d_kernel,* out_img2col;
    cudaMalloc(&d_input, newWidth * newHeight * channels * sizeof(float));
    cudaMalloc(&d_output, width * height * channels * sizeof(float));
    cudaMalloc(&d_kernel, 3 * 3 * 3 * 3 * sizeof(float));
    cudaMalloc(&out_img2col, height* width * 3 * 3 * 3 * sizeof(float));
    float* h_paddedInput = pad_image(input, width, height, channels, newWidth, newHeight, 0);

    cudaMemcpy(d_input, h_paddedInput, newWidth * newHeight * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, 3 * 3 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (height * width * 3 * 3 * 3 + block_size - 1) / block_size;
    img2col << <numBlocks, block_size >> > (d_input, out_img2col, height, width, stride);
    cudaDeviceSynchronize();

    dim3 block(block_size, block_size);
    dim3 grid((3 + block_size - 1) / block_size, (height * width + block_size - 1) / block_size);
    matrixMulKernel << <grid, block >> > (d_kernel, out_img2col, d_output, 3, 27, height*width);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, width * height * channels * sizeof(float), cudaMemcpyDeviceToHost);
    // Clean up
    delete[] h_paddedInput;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

int main()
{   
    char* filenames[] = { "image1.jfif", "image2.jfif","image3.jfif","image4.jfif" };
    int blocksizes[] = { 1, 2, 4, 8, 16, 32 };
    int stride = 1;
    
    //滑动窗口实现
    /*std::cout << "滑动窗口实现" << std::endl;
    for (int b = 0; b < 6; b++) {
        int BLOCK_SIZE = blocksizes[b];
        for (int i = 0; i < 4; i++) {
            float* h_img = read_img(filenames[i]);
            float* h_result = new float[width * height * channels];
            float* kernel = new float[3 * 3 * 3 * 3] {32.32};

            auto start = chrono::high_resolution_clock::now();

            convolutionWithDifferentStrides(h_img, h_result, kernel, width, height, channels, stride, BLOCK_SIZE);

            auto end = chrono::high_resolution_clock::now();
            chrono::duration<float> duration = end - start;
            std::cout << ", BLOCK_SIZE=" << BLOCK_SIZE << ", Time: " << duration.count() * 1000 << " ms " << std::endl;
            delete[] h_img;
            delete[] h_result;
            delete[] kernel;
        }
    }*/

    //共享内存滑动窗口实现
    std::cout << "共享内存滑动窗口实现" << std::endl;
    for (int b = 0; b < 6; b++) {
        int BLOCK_SIZE = blocksizes[b];
        for (int i = 0; i < 4; i++) {
            float* h_img = read_img(filenames[i]);
            float* h_result = new float[width * height * channels];
            float* kernel = new float[3 * 3 * 3 * 3] {0.32};
            auto start = chrono::high_resolution_clock::now();
            convolutionWithDifferentStridesShared(h_img, h_result, kernel, width, height, channels, stride, BLOCK_SIZE);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<float> duration = end - start;

            std::cout << ", BLOCK_SIZE=" << BLOCK_SIZE << ", Time: " << duration.count() * 1000 << " ms " << std::endl;
            delete[] h_img;
            delete[] h_result;
            delete[] kernel;
        }
    }
    //img2col实现
    std::cout << "img2col实现" << std::endl;
    for (int b = 0; b < 6; b++) {
        int BLOCK_SIZE = blocksizes[b];
        for (int i = 0; i < 4; i++) {
            float* h_img = read_img(filenames[i]);
            float* h_result = new float[width * height * channels];
            float* h_check = new float[width * height * channels];
            float* kernel = new float[3 * 3 * 3 * 3] {32.32};

            auto start = chrono::high_resolution_clock::now();

            convolutionImg2Col(h_img, h_result, kernel, width, height, channels, stride, BLOCK_SIZE);

            auto end = chrono::high_resolution_clock::now();
            chrono::duration<float> duration = end - start;
            std::cout << ", BLOCK_SIZE=" << BLOCK_SIZE << ", Time: " << duration.count() * 1000 << " ms " << std::endl;
            delete[] h_img;
            delete[] h_result;
            delete[] kernel;
        }
    }
    return 0;
    
}

