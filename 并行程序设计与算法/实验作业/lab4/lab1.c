#include <pthread.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

clock_t start_time, end_time1, end_time2, end_time3, end_time;
double t1,t2,t3;

typedef struct {
    double a, b, c;      // 方程参数
    double discriminant; // 判别式
    double root1, root2; // 两个根
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int ready;           // 用于条件变量的标志
} SharedData;

void* calculate_discriminant(void *param) {
    SharedData *data = (SharedData *)param;
    pthread_mutex_lock(&data->mutex);
    data->discriminant = data->b * data->b - 4 * data->a * data->c;
    data->ready = 1;
    pthread_cond_broadcast(&data->cond); // 使用broadcast，因为多个线程等待这个条件
    pthread_mutex_unlock(&data->mutex);
    end_time1 = clock();
    return NULL;
}

void* calculate_root1(void *param) {
    SharedData *data = (SharedData *)param;
    pthread_mutex_lock(&data->mutex);
    while (!data->ready) {
        pthread_cond_wait(&data->cond, &data->mutex);
    }
    pthread_mutex_unlock(&data->mutex);

    if (data->discriminant >= 0) {
        data->root1 = (-data->b + sqrt(data->discriminant)) / (2 * data->a);
    } else {
        data->root1 = NAN; // 复数根不计算
    }
    end_time2 = clock();
    return NULL;
}

void* calculate_root2(void *param) {
    SharedData *data = (SharedData *)param;
    pthread_mutex_lock(&data->mutex);
    while (!data->ready) {
        pthread_cond_wait(&data->cond, &data->mutex);
    }
    pthread_mutex_unlock(&data->mutex);

    if (data->discriminant >= 0) {
        data->root2 = (-data->b - sqrt(data->discriminant)) / (2 * data->a);
    } else {
        data->root2 = NAN; // 复数根不计算
    }
    end_time3 = clock();
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s a b c\n", argv[0]);
        return 1;
    }

    SharedData data;
    data.a = atof(argv[1]);
    data.b = atof(argv[2]);
    data.c = atof(argv[3]);
    data.ready = 0;
    pthread_mutex_init(&data.mutex, NULL);
    pthread_cond_init(&data.cond, NULL);

    struct timeval start, end;

    start_time = clock();

    pthread_t tid1, tid2, tid3;
    pthread_create(&tid1, NULL, calculate_discriminant, &data);
    pthread_create(&tid2, NULL, calculate_root1, &data);
    pthread_create(&tid3, NULL, calculate_root2, &data);

    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);
    pthread_join(tid3, NULL);

    end_time = clock();       
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("Roots: x1 = %lf, x2 = %lf\n", data.root1, data.root2);
    printf("总执行时间: %lf secs\n", total_time);
    printf("计算判别式时间: %lf secs\n", (double)(end_time1 - start_time) / CLOCKS_PER_SEC);
    printf("计算第一个根时间: %lf secs\n", (double)(end_time2 - start_time) / CLOCKS_PER_SEC);
    printf("计算第二个根时间: %lf secs\n", (double)(end_time3 - start_time) / CLOCKS_PER_SEC);

    pthread_mutex_destroy(&data.mutex);
    pthread_cond_destroy(&data.cond);

    return 0;
}
