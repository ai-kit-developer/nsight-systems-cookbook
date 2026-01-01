#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

// 每个线程块中的线程数
#define THREAD_PER_BLOCK 256

/**
 * reduce2: 消除 bank conflict 的版本
 * 优化：
 * 1. 使用反向循环（从 blockDim.x/2 开始，每次除以 2）
 * 2. 访问模式 sdata[tid] 和 sdata[tid + s] 避免了 bank conflict
 * 3. 所有活跃线程连续访问共享内存，提高内存带宽利用率
 * 
 * 问题：
 * 1. 仍有分支发散：if (tid < s) 导致部分线程不执行
 * 2. 线程利用率：每次迭代后，一半的线程变为空闲
 */
__global__ void reduce2(float *d_in,float *d_out){
    // 共享内存数组，用于存储每个线程块内的部分和
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程从全局内存加载一个元素到共享内存
    unsigned int tid = threadIdx.x;  // 线程在块内的索引
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  // 线程的全局索引
    sdata[tid] = d_in[i];
    __syncthreads();  // 同步所有线程，确保所有数据加载完成

    // 在共享内存中进行归约操作
    // 优化：使用反向循环，从中间开始，逐步缩小范围
    // 这种访问模式避免了 bank conflict，因为相邻线程访问相邻的内存位置
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        // 只有前 s 个线程参与归约
        // 每个线程将 sdata[tid] 和 sdata[tid + s] 相加
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
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
    reduce2<<<Grid,Block>>>(d_a,d_out);

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
