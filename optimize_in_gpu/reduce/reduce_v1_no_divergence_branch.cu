#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

// 每个线程块中的线程数
#define THREAD_PER_BLOCK 256

/**
 * reduce1: 消除分支发散的版本
 * 优化：
 * 1. 消除了分支发散：使用 index 计算代替模运算，所有线程都执行相同路径
 * 2. 通过条件判断 (index < blockDim.x) 避免越界访问
 * 
 * 问题：
 * 1. 仍然存在 bank conflict：访问模式 sdata[index] 和 sdata[index + s] 可能导致 bank 冲突
 * 2. 线程利用率仍然不高：每次迭代只有部分线程参与计算
 */
__global__ void reduce1(float *d_in,float *d_out){
    // 共享内存数组，用于存储每个线程块内的部分和
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程从全局内存加载一个元素到共享内存
    unsigned int tid = threadIdx.x;  // 线程在块内的索引
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  // 线程的全局索引
    sdata[tid] = d_in[i];
    __syncthreads();  // 同步所有线程，确保所有数据加载完成

    // 在共享内存中进行归约操作
    // 优化：使用连续索引计算，消除分支发散
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        // 计算参与归约的索引位置
        // 所有线程都执行这个计算，避免了分支发散
        int index = 2 * s * tid;
        // 边界检查：只有索引在有效范围内的线程才执行加法
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();  // 每次迭代后同步
    }

    // 将当前线程块的归约结果写入全局内存
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

/**
 * 检查函数：验证 GPU 计算结果是否正确
 * @param out GPU 计算结果数组
 * @param res CPU 计算的参考结果数组
 * @param n 数组长度
 * @return true 如果结果匹配，false 否则
 */
bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
            return false;
    }
    return true;
}

int main(){
    // 数据大小：32MB 的浮点数数组
    const int N=32*1024*1024;
    
    // 分配主机内存
    float *a=(float *)malloc(N*sizeof(float));
    
    // 分配设备内存
    float *d_a;
    cudaMalloc((void **)&d_a,N*sizeof(float));

    // 计算线程块数量
    int block_num=N/THREAD_PER_BLOCK;
    
    // 分配输出数组（每个线程块产生一个结果）
    float *out=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out,(N/THREAD_PER_BLOCK)*sizeof(float));
    
    // CPU 计算的参考结果，用于验证
    float *res=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));

    // 初始化输入数据为 1
    for(int i=0;i<N;i++){
        a[i]=1;
    }

    // CPU 端计算参考结果（每个线程块对应的数据段的和）
    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<THREAD_PER_BLOCK;j++){
            cur+=a[i*THREAD_PER_BLOCK+j];
        }
        res[i]=cur;
    }

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);

    // 配置网格和线程块维度
    dim3 Grid( N/THREAD_PER_BLOCK,1);  // 网格维度
    dim3 Block( THREAD_PER_BLOCK,1);    // 线程块维度

    // 启动 GPU 内核
    reduce1<<<Grid,Block>>>(d_a,d_out);

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    // 验证结果
    if(check(out,res,block_num))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    // 释放内存
    cudaFree(d_a);
    cudaFree(d_out);
}
