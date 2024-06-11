#include <cuda_runtime.h>
#include <iostream>

// CUDA 内核函数
__global__ void helloWorldKernel() {
    int blockId = blockIdx.x;
    int threadIdX = threadIdx.x;
    int threadIdY = threadIdx.y;
    
    printf("Hello World from Thread (%d, %d) in Block %d!\n", threadIdX, threadIdY, blockId);
}

int main() {
    int n = 5; // 线程块数量
    int x = 3; // 线程块维度 - x
    int y = 3; // 线程块维度 - y

    dim3 threadsPerBlock(x, y); // 每个线程块内线程的二维维度
    dim3 blocksPerGrid(n);      // 线程块数量

    // 启动CUDA内核
    helloWorldKernel<<<blocksPerGrid, threadsPerBlock>>>();

    // 主线程输出
    std::cout << "Hello World from the host!" << std::endl;

    // 等待CUDA设备完成所有任务
    cudaDeviceSynchronize();

    return 0;
}
