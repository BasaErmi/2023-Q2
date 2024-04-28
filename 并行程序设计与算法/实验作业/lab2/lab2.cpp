#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <mpi.h>

using namespace std;

// 初始化矩阵
void initialize_matrices(int m, int n, int k, vector<double>& A, vector<double>& B) {
    srand(time(NULL));
    for (int i = 0; i < m * n; ++i) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < n * k; ++i) {
        B[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

// 计算矩阵乘法
void matrix_multiplication(int m, int n, int k, const vector<double>& A, const vector<double>& B, vector<double>& C) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            C[i * k + j] = 0.0;
            for (int p = 0; p < n; ++p) {
                C[i * k + j] += A[i * n + p] * B[p * k + j];
            }
        }
    }
}



int main(int argc, char * argv[] ){
    // 读取参数
    int m=atoi(argv[1]);
	int n=atoi(argv[2]);
	int k=atoi(argv[3]);
    MPI_Init(&argc, &argv); //初始化mpi环境

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // 获取当前进程的rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // 获取总的进程size
    
    // master进程
    if (rank == 0) {
        // 创建三个矩阵
        vector<double> A(m * n);
        vector<double> B(n * k);
        vector<double> C(m * k);

        initialize_matrices(m, n, k, A, B); // 初始化矩阵

        double start_time = MPI_Wtime();

        MPI_Bcast(B.data(), n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);  // 广播B到所有进程

        int rows_per_process = m / size;    // 计算每个进程负责的行数
        int remaining_rows = m % size;      // 剩余行数
        
        // 定义发送计数和位移数组
        vector<int> sendcounts(size);
        vector<int> displs(size);

        // 计算每个进程应该接收的行数和起始位置
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = (i < remaining_rows) ? (rows_per_process + 1) : rows_per_process;
            displs[i] = (i == 0) ? 0 : (displs[i - 1] + sendcounts[i - 1]);
        }

        
        MPI_Datatype row_type;
        MPI_Type_contiguous(n, MPI_DOUBLE, &row_type);
        MPI_Type_commit(&row_type);

        // 通过Scatterv操作将A矩阵分发到各个进程
        MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), row_type, MPI_IN_PLACE, sendcounts[rank] * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        matrix_multiplication(sendcounts[rank], n, k, A, B, C); // 计算矩阵乘法
        // 通过Gatherv操作将各个进程的结果收集回来
        MPI_Gatherv(MPI_IN_PLACE, sendcounts[rank] * k, MPI_DOUBLE, C.data(), sendcounts.data(), displs.data(), row_type, 0, MPI_COMM_WORLD);

        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;

        cout << "Time taken for matrix multiplication: " << elapsed_time << " seconds" << endl;

        MPI_Type_free(&row_type);
    } else {
        vector<double> B(n * k);

        MPI_Bcast(B.data(), n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        int rows_per_process = m / size;
        int remaining_rows = m % size;
        int my_rows = (rank < remaining_rows) ? (rows_per_process + 1) : rows_per_process;

        MPI_Datatype row_type;
        MPI_Type_contiguous(n, MPI_DOUBLE, &row_type);
        MPI_Type_commit(&row_type);

        // 通过Scatterv操作接收A矩阵的一部分
        vector<double> A(my_rows * n);

        MPI_Scatterv(nullptr, nullptr, nullptr, row_type, A.data(), my_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        vector<double> C(my_rows * k);  // 创建矩阵C

        matrix_multiplication(my_rows, n, k, A, B, C);  // 计算矩阵乘法
        // 通过Gatherv操作将结果发送回master进程
        MPI_Gatherv(C.data(), my_rows * k, MPI_DOUBLE, nullptr, nullptr, nullptr, row_type, 0, MPI_COMM_WORLD);

        MPI_Type_free(&row_type);
    }

    MPI_Finalize();

    return 0;
}