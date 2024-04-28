#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <ctime>

using namespace std;

int N;
int *A;
long long sum = 0;
int thread_num;

// 保存每个线程的开始和结束索引
struct SumThread {
    int start_idx;
    int end_idx;
};

// 线程函数，用于计算数组部分的和
void* add(void* args) {
    SumThread* t = static_cast<SumThread*>(args);
    int partial_sum = 0;

    for (int i = t->start_idx; i < t->end_idx; ++i) {
        partial_sum += A[i];
    }
    __sync_fetch_and_add(&sum, partial_sum);  // 原子加操作，用于更新全局和
    return NULL;
}

// 随机值初始化数组
void init(int N) {
    srand(243);
    A = new int[N];
    for (int i = 0; i < N; ++i) {
        A[i] = rand() % 1000 / 10.0;
    }
}


void print_array(int N, int* A) {
    for (int i = 0; i < N; ++i) {
        cout << A[i] << "\t";
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " N thread_num" << endl;
        return 1;
    }

    N = atoi(argv[1]);
    thread_num = atoi(argv[2]);

    init(N);

    pthread_t *thread_handles = new pthread_t[thread_num];
    SumThread *startend = new SumThread[thread_num];

    int p = N / thread_num;
    int q = N % thread_num;
    int start_idx = 0;

    clock_t start_time = clock();
    for (int t = 0; t < thread_num; ++t) {
        startend[t].start_idx = start_idx;
        startend[t].end_idx = start_idx + p + (t < q ? 1 : 0);
        pthread_create(&thread_handles[t], NULL, add, &startend[t]);
        start_idx = startend[t].end_idx;
    }

    for (int t = 0; t < thread_num; ++t) {
        pthread_join(thread_handles[t], NULL);
    }
    clock_t end_time = clock();

    double using_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    cout << "Result: " << sum << endl;
    cout << "Using time: " << using_time << " s" << endl;

    delete[] thread_handles;
    delete[] startend;
    delete[] A;

    return 0;
}
