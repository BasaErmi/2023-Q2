## Lab0-环境设置与串行矩阵乘法

### 环境设置

本次实验选择在自己的主机上安装WSL2来进行实验，如下图所示，已经安装好WSL2，能在windows11主机上运行Ubuntu 22.04子系统：



按照实验文档执行如下代码分别按照OpenMPI和MKL：

```shell
sudo apt-get update
sudo apt-get install libopenmpi-dev –y
sudo apt-get install vim -y
```

```shell
// 下载intel 公钥
sudo wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

// Add the APT Repository
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update

// Install 
sudo apt-get install intel-mkl-64bit-2020.2
source /opt/intel/compilers_and_libraries_2020/linux/mkl/bin/mklvars.sh intel64 ilp64

```

安装完毕后环境配置完成。



### python实现

由于numpy等科学计算库中对矩阵乘法等计算在底层或者硬件层面有很多加速操作，为了体现出python原生的计算速度，这里只引用了random库和time库，分别用来随机初始化矩阵以及计时。

使用嵌套列表生成矩阵A和矩阵B和C，最后再用三层循环对A和B通过通用矩阵乘法计算得到C的值并计时，核心代码如下：

```python
# 用列表生成m x n的矩阵A,每个元素为[512,2048]之间的随机数
A = [[random.uniform(512, 2048) for j in range(n)] for i in range(m)]
B = [[random.uniform(512, 2048) for j in range(k)] for i in range(n)]

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
```

运行文件，结果如下：





### C/C++实现

为了初始化方便，这里使用了C++实现。首先为A、B、C矩阵分配空间，在C++中使用数组来模拟矩阵：

```C++
    A = (double *) malloc(m * k * sizeof(double));
    B = (double *) malloc(k * n * sizeof(double));
    C = (double *) malloc(m * n * sizeof(double));
```

然后定义随机数生成器，保证生成的数值为512-2048之间的随机浮点数

```C++
    // 定义随机数生成器
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(512.0, 2048.0);
```

对三个矩阵进行随机初始化之后进行矩阵乘法并计时，矩阵乘法代码与python一致，都是通过三层循环用通用矩阵乘法计算得到：

```C++
    // 计时开始
    clock_t start, end;
    start = clock();

    // 矩阵乘法
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            for (int p = 0; p < k; p++)
                C[i * n + j] += A[i * k + p] * B[j + p * n];

    // 计时结束
    end = clock();
```



### 调整循环顺序

其他代码不变，将主循环代码第一二层调换：

```C++
//调整循环顺序   
for (j = 0; j < n; j++)
       for (i = 0; i < m; i++)
           for (int p = 0; p < k; p++)
               C[i * n + j] += A[i * k + p] * B[j + p * n];
```



### 循环展开

进行展开数为x的循环展开，具体思路是对最后一层循环遍历的维度进行展开（这里为k），由于k不一定是x的倍数，所以先对`k-k%x`，先对能整除的部分进行循环展开，最后再对余数单独进行一次for循环读取，代码如下：

```C++
    // 计时开始
    start = clock();
    int x = 4;
    // 矩阵乘法
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            for (int p = 0; p < (k - k % x); p += 4) {
                C[i * n + j] += A[i * k + p] * B[j + p * n];
                C[i * n + j] += A[i * k + (p + 1)] * B[j + (p + 1) * n];
                C[i * n + j] += A[i * k + (p + 2)] * B[j + (p + 2) * n];
                C[i * n + j] += A[i * k + (p + 3)] * B[j + (p + 3) * n];
            }
            // 处理剩余的元素
            for (int p = k - k % x; p < k; p++) {
                C[i * n + j] += A[i * k + p] * B[j + p * n];
            }
        }
    }
```

经过多次试验后发现x=4的情况下进行循环展开提升的速度最多，这里选择x=4.



使用如下命令行进行无优化编译：

```shell
g++ -O0 lab0.cpp -o lab0
```

为了确定代码在不进行编译优化的情况下的运行情况，这里编译优化级别设置成0，编译优化标志不开启（经测试，-ffast-math开启与否无显著区别；-funroll-loops开启后运行速度甚至会变慢；-fomit-frame-pointer由于代码未用到函数，所以开启也没有意义）

运行得到如下结果：

![image-20240401200511902](/Users/liuguanlin/Github/Communication-theory/并行程序设计与算法/实验作业/lab0/assets/image-20240401200511902.png)

多次取平均后原始矩阵乘法大概在3.2s左右，调整循环顺序后大概在3.05左右，而循环展开后的乘法时间大概在2.85s。



### 编译优化

这里开启最快的优化级别`-Ofast`，