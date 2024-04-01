## Lab0-环境设置与串行矩阵乘法

### 环境设置

在之前许多课程（操作系统、计算机网络）中已经安装了虚拟机，虚拟机软件为Virtualbox，虚拟机版本为Ubuntu18.04，如下：

<img src="C:\Users\10174\AppData\Roaming\Typora\typora-user-images\image-20240319223528095.png" alt="image-20240319223528095" style="zoom:67%;" />

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

![image-20240319224813541](C:\Users\10174\AppData\Roaming\Typora\typora-user-images\image-20240319224813541.png)



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

使用如下命令行编译：

```shell
g++ -O0 lab0.cpp -o lab0_1
```

为了确定代码在不进行编译优化的情况下的运行情况，这里编译优化级别设置成0，编译优化标志不开启（经测试，-ffast-math开启与否无显著区别；-funroll-loops开启后运行速度甚至会变慢；-fomit-frame-pointer由于代码未用到函数，所以开启也没有意义）

运行结果如下：

![image-20240319230203939](C:\Users\10174\AppData\Roaming\Typora\typora-user-images\image-20240319230203939.png)



### 调整循环顺序

其他代码不变，将主循环代码第一二层调换：

```C++
//调整循环顺序   
for (j = 0; j < n; j++)
       for (i = 0; i < m; i++)
           for (int p = 0; p < k; p++)
               C[i * n + j] += A[i * k + p] * B[j + p * n];
```

编译代码：

```shell
g++ -O0 lab0.cpp -o lab0_2
```

运行结果如下：

![image-20240319230451238](C:\Users\10174\AppData\Roaming\Typora\typora-user-images\image-20240319230451238.png)



### 