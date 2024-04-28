#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <ctime>

using namespace std;

int M, N, K;  // 定义矩阵的维度
double **A, **B, **C;  // 定义指向矩阵的指针
int thread_num;  // 线程数量

// 矩阵乘法的线程函数
void *matrix_multiply(void *arg) {
    long pid = (long)arg;
    int row = M / thread_num;  // 每个线程计算的行数
    int rem = M % thread_num;  // 不能均分的额外行数
    int start_row = pid < rem ? pid * (row + 1) : pid * row + rem;
    int end_row = start_row + (pid < rem ? row + 1 : row);

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < K; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return NULL;
}

// 初始化矩阵
void initialize_matrices() {
    srand(243);  // 设置随机数种子
    A = new double*[M];
    B = new double*[N];
    C = new double*[M];
    
    for (int i = 0; i < M; ++i) {
        A[i] = new double[N];
        for (int j = 0; j < N; ++j) {
            A[i][j] = rand() % 1000 / 10.0;
        }
    }

    for (int i = 0; i < N; ++i) {
        B[i] = new double[K];
        for (int j = 0; j < K; ++j) {
            B[i][j] = rand() % 1000 / 10.0;
        }
    }

    for (int i = 0; i < M; ++i) {
        C[i] = new double[K]();
    }
}

// 打印矩阵
void print_matrix(double **matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << matrix[i][j] << "\t";
        }
        cout << endl;
    }
}

// 释放内存
void free_memory() {
    for (int i = 0; i < M; ++i) delete[] A[i];
    for (int i = 0; i < N; ++i) delete[] B[i];
    for (int i = 0; i < M; ++i) delete[] C[i];
    delete[] A;
    delete[] B;
    delete[] C;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        cout << "Usage: " << argv[0] << " M N K num_threads" << endl;
        return 1;
    }

    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    thread_num = atoi(argv[4]);

    initialize_matrices();

    pthread_t *threads = new pthread_t[thread_num];
    clock_t start_time = clock();
    
    for (int i = 0; i < thread_num; ++i) {
        pthread_create(&threads[i], NULL, matrix_multiply, (void *)i);
    }
    
    for (int i = 0; i < thread_num; ++i) {
        pthread_join(threads[i], NULL);
    }

    clock_t end_time = clock();
    double time_elapsed = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    cout << "Using time: " << time_elapsed << " s" << endl;  // 打印使用时间

    delete[] threads;
    free_memory();  // 释放内存

    return 0;
}
