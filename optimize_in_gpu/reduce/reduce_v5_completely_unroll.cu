#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

// 每个线程块中的线程数
#define THREAD_PER_BLOCK 256

/**
 * warpReduce: 模板化的 Warp 内归约函数
 * 优化：
 * 1. 使用模板参数 blockSize 在编译时确定展开哪些步骤
 * 2. 条件编译：只展开需要的归约步骤
 * 3. 使用 volatile 防止编译器优化
 * 4. 完全展开，消除循环开销
 */
template <unsigned int blockSize>
__device__ void warpReduce(volatile float* cache, unsigned int tid){
    // 根据 blockSize 在编译时决定展开哪些步骤
    // 这样可以避免不必要的计算，同时保持代码的通用性
    if (blockSize >= 64)cache[tid]+=cache[tid+32];  // 如果 blockSize >= 64，需要这一步
    if (blockSize >= 32)cache[tid]+=cache[tid+16];  // 如果 blockSize >= 32，需要这一步
    if (blockSize >= 16)cache[tid]+=cache[tid+8];   // 如果 blockSize >= 16，需要这一步
    if (blockSize >= 8)cache[tid]+=cache[tid+4];    // 如果 blockSize >= 8，需要这一步
    if (blockSize >= 4)cache[tid]+=cache[tid+2];    // 如果 blockSize >= 4，需要这一步
    if (blockSize >= 2)cache[tid]+=cache[tid+1];    // 如果 blockSize >= 2，需要这一步
}

/**
 * reduce5: 完全展开归约循环的版本
 * 优化：
 * 1. 在加载时进行加法（reduce3 的优化）
 * 2. 完全展开归约循环，消除循环开销
 * 3. 使用模板参数在编译时优化代码
 * 4. 展开最后一个 warp 的归约（reduce4 的优化）
 * 5. 条件编译：根据 blockSize 只生成需要的代码
 * 
 * 这是目前最高效的共享内存归约实现之一
 */
template <unsigned int blockSize>
__global__ void reduce5(float *d_in,float *d_out){
    // 共享内存数组，用于存储每个线程块内的部分和
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 每个线程加载两个元素并在加载时立即相加
    unsigned int tid = threadIdx.x;  // 线程在块内的索引
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();  // 同步所有线程，确保所有数据加载完成

    // 优化：完全展开归约循环
    // 使用条件编译，根据 blockSize 在编译时决定执行哪些步骤
    // 这消除了循环开销和分支预测失败的开销
    if (blockSize >= 512) {
        if (tid < 256) { 
            sdata[tid] += sdata[tid + 256]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) {
        if (tid < 128) { 
            sdata[tid] += sdata[tid + 128]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) {
        if (tid < 64) { 
            sdata[tid] += sdata[tid + 64]; 
        } 
        __syncthreads(); 
    }
    // 最后 32 个元素使用展开的 warp 归约
    if (tid < 32) warpReduce<blockSize>(sdata, tid);

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

    // 启动 GPU 内核（使用模板参数指定线程块大小）
    reduce5<THREAD_PER_BLOCK><<<Grid,Block>>>(d_a,d_out);

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
