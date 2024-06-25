#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <cudnn.h>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

int width, height, channels;

// 读取图像
float* read_img(const char* img_path) {
    unsigned char* data = stbi_load(img_path, &width, &height, &channels, 0);
    float* result = new float[width * height * channels];
    if (data) {
        cout << "width: " << width << ", height: " << height << endl;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixelIndex = (y * width + x) * channels;
                for (int c = 0; c < channels; c++) {
                    result[pixelIndex + c] = static_cast<float>(data[pixelIndex + c]);
                }
            }
        }
        stbi_image_free(data);
    }
    else {
        cerr << "Failed to load image" << endl;
    }
    return result;
}

int main() {
    // 初始化 cuDNN
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // 读取图像
    float* host_image = read_img("image.jfif");
    int image_size = width * height * channels * sizeof(float);

    // 卷积核
    int kernel_size = 3 * 3 * 3 * 3; // 3 个 3x3x3 卷积核
    float* host_kernel = new float[kernel_size];
    // 初始化卷积核 (例如, 设置为随机值)
    for (int i = 0; i < kernel_size; i++) {
        host_kernel[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 开始计时
    auto start = chrono::high_resolution_clock::now();
    // 分配 GPU 内存
    float* device_image;
    cudaMalloc(&device_image, image_size);
    cudaMemcpy(device_image, host_image, image_size, cudaMemcpyHostToDevice);

    float* device_kernel;
    cudaMalloc(&device_kernel, kernel_size * sizeof(float));
    cudaMemcpy(device_kernel, host_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // 创建卷积描述符
    cudnnFilterDescriptor_t filter_desc;
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, 3, 3, 3);

    // 创建输入和输出张量描述符
    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channels, height, width);

    int output_height = height - 2; // 3x3 卷积核，填充 0，步幅 1
    int output_width = width - 2;
    int output_channels = 3;

    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_channels, output_height, output_width);

    // 创建卷积描述符
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    // 获取输出尺寸和分配输出内存
    int n, c, h, w;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &n, &c, &h, &w);

    float* device_output;
    cudaMalloc(&device_output, n * c * h * w * sizeof(float));

    // 获取前向卷积算法
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t algo;
    cudnnFindConvolutionForwardAlgorithm(cudnn, input_desc, filter_desc, conv_desc, output_desc, 1, &returnedAlgoCount, &algo);

    // 获取前向卷积需要的工作空间大小
    size_t workspace_bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, 
                                            conv_desc, output_desc, algo.algo, &workspace_bytes);

    void* d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspace_bytes);
    // 执行卷积
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, input_desc, device_image, filter_desc, device_kernel, 
                            conv_desc, algo.algo, d_workspace, workspace_bytes, &beta, output_desc, device_output);

    // 结束计时
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<float> duration = end - start;

    cout << "Time for convolution: " << duration.count()* 1000 << " ms" << endl;

    // 清理资源
    cudaFree(device_image);
    cudaFree(device_kernel);
    cudaFree(device_output);
    cudaFree(d_workspace);
    delete[] host_image;
    delete[] host_kernel;
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}