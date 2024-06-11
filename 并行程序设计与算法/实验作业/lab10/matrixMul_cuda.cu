#include <cuda_runtime.h>
#include <iostream>

// 矩阵维度
#define M 2048
#define N 2048
#define K 2048
#define BLOCK_SIZE 16

// CUDA计算矩阵乘法
__global__ void matrixMulCUDA(double *C, double *A, double *B, int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row <= m && col <= k) {
        double value = 0.0;
        for (int i = 0; i < n; ++i) {
            value += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = value;
    }
}

// 共享内存计算矩阵乘法
__global__ void matrixMulSharedCUDA(double *C, double *A, double *B, int m, int n, int k) {
    __shared__ double shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double shared_B[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0;

    for (int i = 0; i < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        if (row < m && i * BLOCK_SIZE + threadIdx.x < n) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * n + i * BLOCK_SIZE + threadIdx.x]; // A[row][i * BLOCK_SIZE + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (col < k && i * BLOCK_SIZE + threadIdx.y < n) {
            shared_B[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * k + col]; // B[i * BLOCK_SIZE + threadIdx.y][col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            value += shared_A[threadIdx.y][i] * shared_B[i][threadIdx.x];
        } 

        __syncthreads();
    }

    if (row < m && col < k) {
        C[row * k + col] = value;
    }
}

// 串行计算矩阵乘法
void matrixMulCPU(double *C, double *A, double *B, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            double value = 0.0;
            for (int l = 0; l < n; l++) {
                value += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = value;
        }
    }
}


int main() {
    // 分配主机内存
    double *A = new double[M * N];
    double *B = new double[N * K];
    double *C = new double[M * K];
    double *C_CPU = new double[M * K];
    double *C_shared = new double[M * K];
    
    // 初始化矩阵, A和B为随机矩阵，C为0矩阵
    for (int i = 0; i < M * N; i++) {
        A[i] = rand() / (double)RAND_MAX;
    }
    for (int i = 0; i < N * K; i++) {
        B[i] = rand() / (double)RAND_MAX;
    }
    for (int i = 0; i < M * K; i++) {
        C[i] = 0.0;
        C_CPU[i] = 0.0;
        C_shared[i] = 0.0;
    }

    double *d_A, *d_B, *d_C, *d_C_shared;

    // 分配设备内存
    cudaMalloc((void **)&d_A, M * N * sizeof(double));
    cudaMalloc((void **)&d_B, N * K * sizeof(double));
    cudaMalloc((void **)&d_C, M * K * sizeof(double));
    cudaMalloc((void **)&d_C_shared, M * K * sizeof(double));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, A, M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * K * sizeof(double), cudaMemcpyHostToDevice);

    // 定义CUDA网格和块结构
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 计时
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    // 调用CUDA内核函数
    matrixMulCUDA<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, M, N, K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time for matrixMulCUDA: " << elapsedTime << "ms" << std::endl;
    // 将结果从设备复制回主机
    cudaMemcpy(C, d_C, M * K * sizeof(double), cudaMemcpyDeviceToHost);

    // 计算共享内存版本
    cudaEventRecord(start, 0);
    matrixMulSharedCUDA<<<dimGrid, dimBlock>>>(d_C_shared, d_A, d_B, M, N, K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time for matrixMulSharedCUDA: " << elapsedTime << "ms" << std::endl;
    // 将结果从设备复制回主机
    cudaMemcpy(C_shared, d_C_shared, M * K * sizeof(double), cudaMemcpyDeviceToHost);

    // // 串行计算矩阵乘法
    // cudaEventRecord(start, 0);
    // matrixMulCPU(C_CPU, A, B, M, N, K);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // std::cout << "Time for matrixMulCPU: " << elapsedTime << "ms" << std::endl;

    // bool correct = true;
    // // 检查结果
    // for (int i = 0; i < M * K; i++) {
    //     if (abs(C[i] - C_CPU[i]) > 1e-5 || abs(C[i] - C_shared[i]) > 1e-5){
    //         correct = false;
    //         std::cout << "Error at position " << i << ": " << C[i] << " != " << C_CPU[i] << std::endl;
    //         break;
    //     }
    // }

    // if (correct) {
    //     std::cout << "The result is correct!" << std::endl;
    // } else {
    //     std::cout << "The result is wrong!" << std::endl;
    // }

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // 释放主机内存
    free(A);
    free(B);
    free(C);
    free(C_CPU);

    return 0;
}
