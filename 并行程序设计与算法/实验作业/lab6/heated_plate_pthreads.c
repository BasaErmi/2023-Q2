#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "parallel_for.h"
#include <pthread.h>

typedef struct {
    double **u;
    double **w;
    int M;
    int N;
    double epsilon;
    double diff;
    pthread_mutex_t mutex;
} IterationData;



void* compute_iteration(int idx, void* arg) {
    IterationData *data = (IterationData*)arg;
    int i = idx;
    int j;
    int N = data->N;
    double **u = data->u;
    double **w = data->w;

    for ( j = 1; j < N - 1; j++ )
    {
        w[i][j] = ( u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] ) / 4.0;
    }
    return NULL;
}

void* update_values(int idx, void* arg) {
    IterationData *data = (IterationData*)arg;
    int i = idx;
    int j;
    int N = data->N;
    double **u = data->u;
    double **w = data->w;

    for ( j = 0; j < N; j++ )
    {
        u[i][j] = w[i][j];
    }

    return NULL;
}

void* compute_diff(int idx, void* arg) {
    IterationData *data = (IterationData*)arg;
    int i = idx;
    int j;
    int M = data->M;
    int N = data->N;
    double **u = data->u;
    double **w = data->w;
    double diff_local = 0.0;

    for (j = 1; j < N - 1; j++) {
        diff_local = fmax(diff_local, fabs(u[i][j] - w[i][j]));
    }

    pthread_mutex_lock(&data->mutex);
    if (data->diff < diff_local) {
        data->diff = diff_local;
    }
    pthread_mutex_unlock(&data->mutex);

    return NULL;
}

int main(int argc, char* argv[]) {
    if(argc != 4) {
        printf("Usage: %s M N p\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int p = atoi(argv[3]);

    double epsilon = 0.001;
    double diff = epsilon;
    int iterations = 0;
    int iterations_print = 1;
    struct timeval startwtime, endwtime;

    double **u = (double**)malloc(M * sizeof(double*));
    double **w = (double**)malloc(M * sizeof(double*));
    for(int i = 0; i < M; i++) {
        u[i] = (double*)malloc(N * sizeof(double));
        w[i] = (double*)malloc(N * sizeof(double));
    }

    double mean = 0.0;
    for (int i = 1; i < M - 1; i++) {
        w[i][0] = 100.0;
    }
    for (int i = 1; i < M - 1; i++) {
        w[i][N - 1] = 100.0;
    }
    for (int j = 0; j < N; j++) {
        w[M - 1][j] = 100.0;
    }
    for (int j = 0; j < N; j++) {
        w[0][j] = 0.0;
    }

    for (int i = 1; i < M - 1; i++ ) {
        mean = mean + w[i][0] + w[i][N - 1];
    }
    for (int j = 0; j < N; j++) {
        mean = mean + w[M - 1][j] + w[0][j];
    }

    mean = mean / (double)(2 * M + 2 * N - 4);

    printf("\n");
    printf("  MEAN = %f\n", mean);

    for (int i = 1; i < M - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            w[i][j] = mean;
        }
    }

    IterationData data = {u, w, M, N, epsilon, diff};
    // 锁初始化
    pthread_mutex_init(&data.mutex, NULL);

    printf(" Iteration  Change\n");
    printf("\n");

    gettimeofday(&startwtime, NULL);

    while (diff >= epsilon) {

        data.diff = 0.0;
        parallel_for(0, M, 1, update_values, (void*)&data, p);

        parallel_for(1, M - 1, 1, compute_iteration, (void*)&data, p);
        
        parallel_for(1, M - 1, 1, compute_diff, (void*)&data, p);

        diff = data.diff;

        iterations++;
        if (iterations == iterations_print) {
            printf(" %8d  %f\n", iterations, diff);
            iterations_print = 2 * iterations_print;
        }


        }

    gettimeofday(&endwtime, NULL);

    printf("\n");
    printf(" %10d  %f\n", iterations, diff);
    printf("\n");
    printf(" Error tolerance achieved.\n");
    printf(" Wallclock time = %f\n", (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec));

    for(int i = 0; i < M; i++) {
        free(u[i]);
        free(w[i]);
    }
    free(u);
    free(w);
    return 0;
}