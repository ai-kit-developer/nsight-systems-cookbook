#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

// 每个线程块中的线程数
#define THREAD_PER_BLOCK 256

/**
 * reduce3: 在加载时进行加法的版本
 * 优化：
 * 1. 每个线程加载两个元素并在加载时立即相加
 * 2. 减少了全局内存访问次数（从 N 次减少到 N/2 次）
 * 3. 提高了内存带宽利用率
 * 4. 需要调整网格大小：Grid = N/(2*THREAD_PER_BLOCK)
 * 
 * 注意：这种优化减少了线程块数量，从而减少了最终的归约步骤
 */
__global__ void reduce3(float *d_in,float *d_out){
    // 共享内存数组，用于存储每个线程块内的部分和
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 优化：每个线程加载两个元素并在加载时立即相加
    // 这减少了全局内存访问次数，提高了带宽利用率
    unsigned int tid = threadIdx.x;  // 线程在块内的索引
    // 计算全局索引：每个线程块处理 2*blockDim.x 个元素
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // 加载两个元素并立即相加，减少后续的归约步骤
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();  // 同步所有线程，确保所有数据加载完成

    // 在共享内存中进行归约操作
    // 使用反向循环，避免 bank conflict
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
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

    // 每个线程块处理 2*THREAD_PER_BLOCK 个元素（因为每个线程加载两个元素）
    int NUM_PER_BLOCK = 2*THREAD_PER_BLOCK;
    int block_num = N / NUM_PER_BLOCK;
    
    // 分配输出数组（每个线程块产生一个结果）
    float *out=(float *)malloc(block_num*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out,block_num*sizeof(float));
    
    // CPU 计算的参考结果，用于验证
    float *res=(float *)malloc(block_num*sizeof(float));

    // 初始化输入数据为 1
    for(int i=0;i<N;i++){
        a[i]=1;
    }

    // CPU 端计算参考结果（每个线程块对应的数据段的和）
    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<NUM_PER_BLOCK;j++){
            cur+=a[i * NUM_PER_BLOCK + j];
        }
        res[i]=cur;
    }

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);

    // 配置网格和线程块维度
    dim3 Grid( block_num, 1);  // 网格维度（线程块数量减少了一半）
    dim3 Block( THREAD_PER_BLOCK, 1);  // 线程块维度

    // 启动 GPU 内核
    reduce3<<<Grid,Block>>>(d_a,d_out);

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
