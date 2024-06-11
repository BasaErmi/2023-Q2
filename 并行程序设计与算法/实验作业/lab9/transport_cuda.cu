#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

void generateRandomMatrix(float* matrix, int n) {
    srand(time(0));
    for (int i = 0; i < n * n; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

__global__ void transposeGlobalMemory(float* A, float* A_T, int n) {
    // 计算当前线程在全局矩阵中的位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查索引是否在矩阵范围内
    if (x < n && y < n) {
        // 将A中的元素转置后存储到A_T中
        A_T[x * n + y] = A[y * n + x];
    }
}


__global__ void transposeSharedMemory(float* A, float* A_T, int n) {
    const int block_size = 16;
    // 定义共享内存
    __shared__ float tile[block_size][block_size+1]; 

    // 计算全局索引
    int x = blockIdx.x * block_size + threadIdx.x;
    int y = blockIdx.y * block_size + threadIdx.y;

    // 检查索引是否在矩阵范围内
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = A[y * n + x];
    }

    // 同步线程，确保共享内存填充完毕
    __syncthreads();

    // 计算新的全局索引
    x = blockIdx.y * block_size + threadIdx.x;
    y = blockIdx.x * block_size + threadIdx.y;

    // 检查索引是否在矩阵范围内
    if (x < n && y < n) {
        A_T[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }
}


void transposeCPU(float* A, float* A_T, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_T[j * n + i] = A[i * n + j];
        }
    }
}

int main() {
    int n = 2048; // 矩阵规模
    int threads = 16;
    size_t bytes = n * n * sizeof(float);

    std::cout << "Matrix size: " << n << "x" << n << std::endl;
    std::cout << "Threads per block: " << threads << "x" << threads << std::endl;

    // 分配内存
    float* h_A = (float*)malloc(bytes);
    float* h_A_T = (float*)malloc(bytes);
    float* h_A_T_shared = (float*)malloc(bytes);

    generateRandomMatrix(h_A, n);

    float* d_A, *d_A_T, *d_A_T_shared;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_A_T, bytes);
    cudaMalloc(&d_A_T_shared, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    // 配置线程块和网格维度
    dim3 threadsPerBlock(threads, threads);
    dim3 blocksPerGrid((n + 1) / threads, (n + 1) / threads);

    // 启动CUDA内核
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 全局内存版本
    cudaEventRecord(start, 0);
    transposeGlobalMemory<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_A_T, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time for global memory transpose: " << elapsedTime << " ms" << std::endl;

    // 共享内存版本
    cudaEventRecord(start, 0);
    transposeSharedMemory<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_A_T_shared, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time for shared memory transpose: " << elapsedTime << " ms" << std::endl;

    // 将结果从设备复制回主机
    cudaMemcpy(h_A_T, d_A_T, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A_T_shared, d_A_T_shared, bytes, cudaMemcpyDeviceToHost);

    // CPU版本转置
    float* h_A_T_CPU = (float*)malloc(bytes);
    time_t startCPU = clock();
    transposeCPU(h_A, h_A_T_CPU, n);
    time_t endCPU = clock();
    std::cout << "Time for CPU transpose: " << (endCPU - startCPU) * 1000 / CLOCKS_PER_SEC << " ms" << std::endl;

    // 检查结果是否正确
    bool correct = true;
    for (int i = 0; i < n * n; i++) {
        if (fabs(h_A_T[i] - h_A_T_CPU[i]) > 1e-5 || fabs(h_A_T_shared[i] - h_A_T_CPU[i]) > 1e-5) {
            correct = false;
            break;
        }
    }


    if (correct) {
        std::cout << "Transpose successful!" << std::endl;
    } else {
        std::cout << "Transpose failed!" << std::endl;
    }

    // 释放内存
    free(h_A);
    free(h_A_T);
    free(h_A_T_CPU);
    cudaFree(d_A);
    cudaFree(d_A_T);

    return 0;
}
