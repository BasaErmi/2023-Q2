#include <pthread.h>
#include <stdlib.h>

typedef struct {
    int start;
    int end;
    void *(*functor)(int, void*);
    void *arg;
    int thread_id;
} ThreadArg;

void* thread_func(void* arg) {
    ThreadArg* thread_arg = (ThreadArg*)arg;
    for (int i = thread_arg->start; i < thread_arg->end; i++) {
        thread_arg->functor(i, thread_arg->arg);
    }
    return NULL;
}

void parallel_for(int start, int end, int inc, void *(*functor)(int, void*), void *arg, int num_threads) {
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    ThreadArg* thread_args = malloc(num_threads * sizeof(ThreadArg));

    int range = end - start;
    int chunk_size = range / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_args[i].start = start + i * chunk_size;
        thread_args[i].end = i == num_threads - 1 ? end : thread_args[i].start + chunk_size;
        thread_args[i].functor = functor;
        thread_args[i].arg = arg;
        thread_args[i].thread_id = i;
        pthread_create(&threads[i], NULL, thread_func, &thread_args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(thread_args);
}