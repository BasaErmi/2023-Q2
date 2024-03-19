import random

m = 2000
k = 200
n = 1000

# 对结果计时
import time
# 用列表生成m x n的矩阵A,每个元素为[512,2048]之间的随机数
A = [[random.randint(512, 2048) for j in range(n)] for i in range(m)]
B = [[random.randint(512, 2048) for j in range(k)] for i in range(n)]

start_time = time.time()
# 对A和B进行乘法运算
C = [[0 for j in range(k)] for i in range(m)]
for i in range(m):
    for j in range(k):
        for l in range(n):
            C[i][j] += A[i][l] * B[l][j]

end_time = time.time()
use_time = end_time - start_time
print("time: ", use_time, "s")