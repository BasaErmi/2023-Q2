#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

int thread_count;
long long int m, n;
clock_t start_time, end_time;
pthread_mutex_t mutex;


void* compute_pi(void *rank)
{
    long my_rank = (long)rank;
    long long int local_n = n / thread_count;
    long long int my_first_i = my_rank * (n / thread_count);
    long long int my_last_i = my_first_i + (n / thread_count);
    long long int local_m = 0;

    for (long long int i = my_first_i; i < my_last_i; i++)
    {
        double x =(double)rand() / RAND_MAX;
        double y =(double)rand() / RAND_MAX;
        if (x * x + y * y <= 1)
        {
            local_m++;
        }
    }
    pthread_mutex_lock(&mutex);
    m += local_m ;
    pthread_mutex_unlock(&mutex);
    return NULL;
}


int main(int argc, char* argv[])
{
    srand(time(NULL));
    long thread;
    pthread_t* thread_handles;

    n = atoi(argv[1]);
    thread_count = strtol(argv[2], NULL, 10);
    thread_handles = malloc(thread_count * sizeof(pthread_t));

    pthread_mutex_init(&mutex, NULL);

    start_time = clock();
    // 创建线程
    for (thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, compute_pi, (void*) thread);
    
    for (thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    
    end_time = clock();

    double pi = 4 * (double)m / (double)n;
    printf("n = %lld, m = %lld, pi= %lf\n", n, m, 3.141593 - pi);
    
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("总运行时间: %lf secs\n", total_time);

    pthread_mutex_destroy(&mutex);

    return 0;

}


