#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
using namespace std;


// 初始化矩阵
void init_Mat(int row, int col,double* mat){
    for (int m=0;m<row;m++){
        for (int n=0;n<col;n++){
            mat[m*col+n]=(double)(rand()%1000/10.0);
        }
    }
}

int main(int argc, char * argv[] ){
    // 解析命令行参数得到的矩阵维度
    int M=atoi(argv[1]);
    int N=atoi(argv[2]);
    int K=atoi(argv[3]);
    // 动态分配存储矩阵和结果矩阵的空间
    double *b= new double [ N* K ];
    double *result = new double [ M * K ];
    double *a=NULL,*c=NULL;
    int pid, process_num, line;

    // 初始化MPI环境
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD,&pid);    // 获取当前进程的ID
    MPI_Comm_size(MPI_COMM_WORLD,&process_num);    // 获取进程总数

    line = M/process_num;
    srand(508);

    // 主进程
    if(pid==0){
        // 动态分配存储矩阵A的空间
        a=new double[M*N];
        c=new double[M*K];

        init_Mat(M,N,a);
        init_Mat(N,K,b);

        // 记录开始时间
        double start_time;
        double end_time;
        start_time=MPI_Wtime();
        // 发送矩阵A和 B到子进程
        for (int i=1;i<process_num;i++){
            MPI_Send(b,N*K,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
        }
        for (int i=1;i<process_num;i++){
            MPI_Send(a+(i-1)*line*N,N*line,MPI_DOUBLE,i,1,MPI_COMM_WORLD);
        }

        // 接收子进程的计算结果
        for (int i=1;i<process_num;i++){
            MPI_Recv(result,line*K,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            for(int l=0;l<line;l++){
                for(int k=0;k<K;k++){
                    c[((i-1)*line+l)*K+k]=result[l*K+k];
                }
            }
        }

        // 计算矩阵C
        for (int i=(process_num-1)*line;i<M;i++){
            for (int j=0;j<K;j++){
                double tmp=0;
                for (int k=0;k<N;k++)
                    tmp += a[i*N+k]*b[k*K+j];
                c[i*K+j] = tmp;
            }
        }

        // 计算总耗时
        end_time=MPI_Wtime();
        double using_time=end_time-start_time;

        cout<<"using time:"<<using_time<<endl;
    }

    // 子进程
    else{
        // 动态分配存储接收矩阵B和A的空间
        double* temp = new double [ N * line ];

        // 接收矩阵B和A的行
        MPI_Recv(b,N*K,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(temp,N*line,MPI_DOUBLE,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		
        // 计算矩阵乘法
        for(int i=0;i<line;i++){
            for(int j=0;j<N;j++){
                double tmp=0;
                for(int k=0;k<N;k++)
                    tmp += temp[i*N+k]*b[k*K+j];
                result[i*K+j] = tmp;
            }
        }
        // 发送计算结果到主进程
        MPI_Send(result, line*K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}