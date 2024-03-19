#define min(x,y) (((x) < (y)) ? (x) : (y))
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace std;
int main() {
    double *A, *B, *C;
    int m, n, k, i, j;

    m = 2000, k = 200, n = 1000;
    printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
           " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);

    A = (double *) malloc(m * k * sizeof(double));
    B = (double *) malloc(k * n * sizeof(double));
    C = (double *) malloc(m * n * sizeof(double));

    // 使用随机设备生成随机种子
    random_device rd;
    // 使用 Mersenne Twister 引擎生成随机数
    mt19937 gen(rd());
    // 定义均匀分布，范围在512到2048之间
    uniform_real_distribution<double> dis(512.0, 2048.0);

    // 生成随机数
    double random_number = dis(gen);
    //初始化A、B、C矩阵中的数值，每个值在[512,2048]之间的随机数
    for (i = 0; i < m; i++)
        for (j = 0; j < k; j++)
            A[i * k + j] = (double) (dis(gen));
    for (i = 0; i < k; i++)
        for (j = 0; j < n; j++)
            B[i * n + j] = (double) (dis(gen));
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            C[i * n + j] = 0.0;

    // 计时开始
    clock_t start, end;
    start = clock();
    // 矩阵乘法
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            for (int p = 0; p < k; p++)
                C[i * n + j] += A[i * k + p] * B[j + p * n];

    //调整循环顺序
//    for (j = 0; j < n; j++)
//        for (i = 0; i < m; i++)
//            for (int p = 0; p < k; p++)
//                C[i * n + j] += A[i * k + p] * B[j + p * n];

    // 计时结束
    end = clock();
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            printf("%lf ", C[i * n + j]);
        puts("");
    }

    printf(" 矩阵乘法时间 %f seconds\n", (double) (end - start) / CLOCKS_PER_SEC);
}