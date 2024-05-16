#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "parallel_for.h"

typedef struct {
    float **A, **B, **C;
    int n;
} MatrixArgs;

struct timeval startwtime, endwtime;
double seq_time;

void* matrix_multiply(int idx, void* args){
    MatrixArgs *args_data = (MatrixArgs*) args;
    int i = idx / args_data->n;
    int j = idx % args_data->n;
    args_data->C[i][j] = 0;
    for(int k = 0; k < args_data->n; k++)
        args_data->C[i][j] += args_data->A[i][k] * args_data->B[k][j];
    return NULL;
}

int main(int argc, char* argv[]) {
    if(argc != 5) {
        printf("Usage: %s m n k p\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int p = atoi(argv[4]);

    float **A = (float**)malloc(m * sizeof(float*));
    float **B = (float**)malloc(n * sizeof(float*));
    for(int i = 0; i < m; i++)
        A[i] = (float*)malloc(n * sizeof(float));
    for(int i = 0; i < n; i++)
        B[i] = (float*)malloc(k * sizeof(float));

    srand(time(NULL));
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            A[i][j] = rand() / (float)RAND_MAX;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < k; j++)
            B[i][j] = rand() / (float)RAND_MAX;

    float **C = (float**)malloc(m * sizeof(float*));
    for(int i = 0; i < m; i++)
        C[i] = (float*)malloc(k * sizeof(float));

    MatrixArgs args = {A, B, C, n};

    gettimeofday(&startwtime, NULL);

    parallel_for(0, m*k, 1, matrix_multiply, (void*)&args, p);

    gettimeofday(&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

    printf("Time: %f seconds\n", seq_time);
    for(int i = 0; i < m; i++)
        free(A[i]);
    for(int i = 0; i < n; i++)
        free(B[i]);
    for(int i = 0; i < m; i++)
        free(C[i]);
    free(A);
    free(B);
    free(C);
    return 0;
}