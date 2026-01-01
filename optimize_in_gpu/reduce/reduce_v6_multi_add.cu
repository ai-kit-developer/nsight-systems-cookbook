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
 * 与 reduce5 中的实现相同，用于 warp 内的最后归约步骤
 */
template <unsigned int blockSize>
__device__ void warpReduce(volatile float* cache, unsigned int tid){
    if (blockSize >= 64)cache[tid]+=cache[tid+32];
    if (blockSize >= 32)cache[tid]+=cache[tid+16];
    if (blockSize >= 16)cache[tid]+=cache[tid+8];
    if (blockSize >= 8)cache[tid]+=cache[tid+4];
    if (blockSize >= 4)cache[tid]+=cache[tid+2];
    if (blockSize >= 2)cache[tid]+=cache[tid+1];
}

/**
 * reduce6: 每个线程处理多个元素的版本
 * 优化：
 * 1. 每个线程处理 NUM_PER_THREAD 个元素，减少线程块数量
 * 2. 使用 #pragma unroll 展开循环，提高性能
 * 3. 减少了全局内存访问的延迟影响
 * 4. 提高了 GPU 的占用率（occupancy）
 * 5. 结合了 reduce5 的所有优化（完全展开、warp 归约等）
 * 
 * 这种优化特别适合处理大量数据的情况
 */
template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce6(float *d_in,float *d_out, unsigned int n){
    // 共享内存数组，大小等于线程块大小
    __shared__ float sdata[blockSize];

    // 每个线程处理 NUM_PER_THREAD 个元素
    unsigned int tid = threadIdx.x;  // 线程在块内的索引
    // 计算起始全局索引：每个线程块处理 blockSize * NUM_PER_THREAD 个元素
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    // 初始化共享内存中的累加器
    sdata[tid] = 0;

    // 优化：使用 #pragma unroll 展开循环
    // 每个线程加载并累加 NUM_PER_THREAD 个元素
    // 这减少了线程块数量，提高了内存带宽利用率
    #pragma unroll
    for(int iter=0; iter<NUM_PER_THREAD; iter++){
        // 使用跨步访问模式：相邻线程访问间隔 blockSize 的元素
        // 这有助于合并内存访问，提高带宽利用率
        sdata[tid] += d_in[i+iter*blockSize];
    }
    
    __syncthreads();  // 同步所有线程，确保所有数据加载完成

    // 在共享内存中进行归约操作（与 reduce5 相同）
    // 完全展开归约循环
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

    // 固定线程块数量为 1024
    const int block_num = 1024;
    // 计算每个线程块需要处理的元素数量
    const int NUM_PER_BLOCK = N / block_num;
    // 计算每个线程需要处理的元素数量
    const int NUM_PER_THREAD = NUM_PER_BLOCK/THREAD_PER_BLOCK;
    
    // 分配输出数组（每个线程块产生一个结果）
    float *out=(float *)malloc(block_num*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out,block_num*sizeof(float));
    
    // CPU 计算的参考结果，用于验证
    float *res=(float *)malloc(block_num*sizeof(float));

    // 初始化输入数据（使用模运算生成测试数据）
    for(int i=0;i<N;i++){
        a[i]=i%456;
    }

    // CPU 端计算参考结果（每个线程块对应的数据段的和）
    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<NUM_PER_BLOCK;j++){
            if(i * NUM_PER_BLOCK + j < N){
                cur+=a[i * NUM_PER_BLOCK + j];
            }
        }
        res[i]=cur;
    }

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);

    // 配置网格和线程块维度
    dim3 Grid( block_num, 1);  // 网格维度
    dim3 Block( THREAD_PER_BLOCK, 1);  // 线程块维度
    
    // 启动 GPU 内核（使用模板参数指定线程块大小和每个线程处理的元素数）
    reduce6<THREAD_PER_BLOCK, NUM_PER_THREAD><<<Grid,Block>>>(d_a, d_out, N);

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
