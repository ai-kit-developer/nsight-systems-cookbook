#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

// 每个线程块中的线程数
#define THREAD_PER_BLOCK 256

// 向量化加载宏：使用向量类型提高内存带宽利用率
// FETCH_FLOAT2: 一次加载 2 个 float（8 字节）
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
// FETCH_FLOAT4: 一次加载 4 个 float（16 字节）
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

/**
 * add: 基础版本的逐元素加法内核
 * 计算 c = a + b，其中 a、b、c 是相同大小的数组
 * 
 * 实现：
 * - 每个线程处理一个元素
 * - 简单的逐元素加法
 */
__global__ void add(float* a, float* b, float* c)
{
    // 计算全局线程索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 执行逐元素加法
    c[idx] = a[idx] + b[idx];
}

/**
 * vec2_add: 使用 float2 向量化的逐元素加法内核
 * 计算 c = a + b，其中 a、b、c 是相同大小的数组
 * 
 * 优化：
 * 1. 使用 float2 向量化加载，一次处理 2 个元素
 * 2. 减少内存访问次数（从 2N 次减少到 N 次）
 * 3. 提高内存带宽利用率
 * 4. 每个线程处理 2 个元素，减少线程数量需求
 */
__global__ void vec2_add(float* a, float* b, float* c)
{
    // 计算全局索引（每个线程处理 2 个元素）
    int idx = (threadIdx.x + blockIdx.x * blockDim.x)*2;
    // 使用 float2 向量化加载
    float2 reg_a = FETCH_FLOAT2(a[idx]);
    float2 reg_b = FETCH_FLOAT2(b[idx]);
    float2 reg_c;
    // 执行向量化加法
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    // 使用 float2 向量化存储
    FETCH_FLOAT2(c[idx]) = reg_c;
}

/**
 * vec4_add: 使用 float4 向量化的逐元素加法内核
 * 计算 c = a + b，其中 a、b、c 是相同大小的数组
 * 
 * 优化：
 * 1. 使用 float4 向量化加载，一次处理 4 个元素
 * 2. 进一步减少内存访问次数（从 2N 次减少到 N/2 次）
 * 3. 最大化内存带宽利用率
 * 4. 每个线程处理 4 个元素，进一步减少线程数量需求
 * 
 * 这是最高效的版本，适合大多数现代 GPU
 */
__global__ void vec4_add(float* a, float* b, float* c)
{
    // 计算全局索引（每个线程处理 4 个元素）
    int idx = (threadIdx.x + blockIdx.x * blockDim.x)*4;
    // 使用 float4 向量化加载
    float4 reg_a = FETCH_FLOAT4(a[idx]);
    float4 reg_b = FETCH_FLOAT4(b[idx]);
    float4 reg_c;
    // 执行向量化加法
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    // 使用 float4 向量化存储
    FETCH_FLOAT4(c[idx]) = reg_c;
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

/**
 * 主函数：测试逐元素加法实现的正确性和性能
 */
int main(){
    // 数据大小：32MB 的浮点数数组
    const int N=32*1024*1024;
    
    // 分配主机内存
    float *a=(float *)malloc(N*sizeof(float));
    float *b=(float *)malloc(N*sizeof(float));
    float *out=(float *)malloc(N*sizeof(float));
    
    // 分配设备内存
    float *d_a;
    float *d_b;
    float *d_out;
    cudaMalloc((void **)&d_a,N*sizeof(float));
    cudaMalloc((void **)&d_b,N*sizeof(float));
    cudaMalloc((void **)&d_out,N*sizeof(float));
    
    // CPU 计算的参考结果，用于验证
    float *res=(float *)malloc(N*sizeof(float));

    // 初始化输入数据
    for(int i=0;i<N;i++){
        a[i]=1;      // 数组 a 全为 1
        b[i]=i;      // 数组 b 为索引值
        res[i]=a[i]+b[i];  // CPU 端计算参考结果
    }

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,N*sizeof(float),cudaMemcpyHostToDevice);

    // 配置网格和线程块维度
    // 注意：由于使用 vec4_add，每个线程处理 4 个元素
    // 所以网格大小是 N/(THREAD_PER_BLOCK*4)
    dim3 Grid( N/THREAD_PER_BLOCK/4, 1);
    dim3 Block( THREAD_PER_BLOCK, 1);

    // 多次迭代执行内核，用于性能测试
    int iter = 2000;
    for(int i=0; i<iter; i++){
        vec4_add<<<Grid,Block>>>(d_a, d_b, d_out);
    }

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(out,d_out,N*sizeof(float),cudaMemcpyDeviceToHost);

    // 验证结果
    if(check(out,res,N))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<N;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    // 释放内存
    cudaFree(d_a);
    cudaFree(d_out);
}
