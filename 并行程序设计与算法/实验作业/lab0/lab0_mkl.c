#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <ctime>

int main()
{
    double *A, *B, *C;
    int m, n, k, i, j;
    double alpha, beta;

    m = 2000, k = 200, n = 2000;
    printf(" 初始化矩阵维度 \n"
           " 矩阵A(%ix%i) 矩阵B(%ix%i)\n运算时间结果\n", m, k, k, n);
    alpha = 1.0; beta = 0.0;

    A = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
    B = (double *)mkl_malloc( k*n*sizeof( double ), 64 );
    C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }

    // 初始化矩阵
    for (i = 0; i < (m*k); i++) {
        A[i] = (double)(i+1);
    }

    for (i = 0; i < (k*n); i++) {
        B[i] = (double)(-i-1);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }

    time_t start, end;
    start = clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);
    end = clock();
    printf ("\n 使用MKL进行矩阵运算的时间: %f 秒\n", (double)(end-start)/CLOCKS_PER_SEC);

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}