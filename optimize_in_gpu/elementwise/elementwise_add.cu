#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

// 每个线程块中的线程数
#define THREADS_PER_BLOCK 256

// 向量化加载宏：使用向量类型提高内存带宽利用率
// FETCH_FLOAT2: 一次加载 2 个 float（8 字节）
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
// FETCH_FLOAT4: 一次加载 4 个 float（16 字节）
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

/**
 * add: 基础版本的逐元素加法内核
 * 计算 output = input_a + input_b，其中 input_a、input_b、output 是相同大小的数组
 * 
 * 实现：
 * - 每个线程处理一个元素
 * - 简单的逐元素加法
 */
__global__ void add(float* input_a, float* input_b, float* output)
{
    // 计算全局线程索引
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 执行逐元素加法
    output[global_idx] = input_a[global_idx] + input_b[global_idx];
}

/**
 * vec2_add: 使用 float2 向量化的逐元素加法内核
 * 计算 output = input_a + input_b，其中 input_a、input_b、output 是相同大小的数组
 * 
 * 优化：
 * 1. 使用 float2 向量化加载，一次处理 2 个元素
 * 2. 减少内存访问次数（从 2N 次减少到 N 次）
 * 3. 提高内存带宽利用率
 * 4. 每个线程处理 2 个元素，减少线程数量需求
 */
__global__ void vec2_add(float* input_a, float* input_b, float* output)
{
    // 计算全局索引（每个线程处理 2 个元素）
    int global_idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    // 使用 float2 向量化加载
    float2 reg_a = FETCH_FLOAT2(input_a[global_idx]);
    float2 reg_b = FETCH_FLOAT2(input_b[global_idx]);
    float2 reg_c;
    // 执行向量化加法
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    // 使用 float2 向量化存储
    FETCH_FLOAT2(output[global_idx]) = reg_c;
}

/**
 * vec4_add: 使用 float4 向量化的逐元素加法内核
 * 计算 output = input_a + input_b，其中 input_a、input_b、output 是相同大小的数组
 * 
 * 优化：
 * 1. 使用 float4 向量化加载，一次处理 4 个元素
 * 2. 进一步减少内存访问次数（从 2N 次减少到 N/2 次）
 * 3. 最大化内存带宽利用率
 * 4. 每个线程处理 4 个元素，进一步减少线程数量需求
 * 
 * 这是最高效的版本，适合大多数现代 GPU
 */
__global__ void vec4_add(float* input_a, float* input_b, float* output)
{
    // 计算全局索引（每个线程处理 4 个元素）
    int global_idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    // 使用 float4 向量化加载
    float4 reg_a = FETCH_FLOAT4(input_a[global_idx]);
    float4 reg_b = FETCH_FLOAT4(input_b[global_idx]);
    float4 reg_c;
    // 执行向量化加法
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    // 使用 float4 向量化存储
    FETCH_FLOAT4(output[global_idx]) = reg_c;
}

/**
 * 检查函数：验证 GPU 计算结果是否正确
 * @param host_output_data GPU 计算结果数组
 * @param reference_result CPU 计算的参考结果数组
 * @param num_elements 数组长度
 * @return true 如果结果匹配，false 否则
 */
bool verify_result(float *host_output_data, float *reference_result, int num_elements){
    for(int element_idx = 0; element_idx < num_elements; element_idx++){
        if(host_output_data[element_idx] != reference_result[element_idx])
            return false;
    }
    return true;
}

/**
 * 主函数：测试逐元素加法实现的正确性和性能
 */
int main(){
    // 数据大小：32MB 的浮点数数组
    const int num_elements = 32 * 1024 * 1024;
    
    // 分配主机内存
    float *host_input_a = (float *)malloc(num_elements * sizeof(float));
    float *host_input_b = (float *)malloc(num_elements * sizeof(float));
    float *host_output_data = (float *)malloc(num_elements * sizeof(float));
    
    // 分配设备内存
    float *device_input_a;
    float *device_input_b;
    float *device_output_data;
    cudaMalloc((void **)&device_input_a, num_elements * sizeof(float));
    cudaMalloc((void **)&device_input_b, num_elements * sizeof(float));
    cudaMalloc((void **)&device_output_data, num_elements * sizeof(float));
    
    // CPU 计算的参考结果，用于验证
    float *reference_result = (float *)malloc(num_elements * sizeof(float));

    // 初始化输入数据
    for(int element_idx = 0; element_idx < num_elements; element_idx++){
        host_input_a[element_idx] = 1;      // 数组 input_a 全为 1
        host_input_b[element_idx] = element_idx;      // 数组 input_b 为索引值
        reference_result[element_idx] = host_input_a[element_idx] + host_input_b[element_idx];  // CPU 端计算参考结果
    }

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(device_input_a, host_input_a, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_input_b, host_input_b, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // 配置网格和线程块维度
    // 注意：由于使用 vec4_add，每个线程处理 4 个元素
    // 所以网格大小是 num_elements/(THREADS_PER_BLOCK*4)
    dim3 grid_dim(num_elements / THREADS_PER_BLOCK / 4, 1);
    dim3 block_dim(THREADS_PER_BLOCK, 1);

    // 多次迭代执行内核，用于性能测试
    int num_iterations = 2000;
    for(int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++){
        vec4_add<<<grid_dim, block_dim>>>(device_input_a, device_input_b, device_output_data);
    }

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(host_output_data, device_output_data, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    if(verify_result(host_output_data, reference_result, num_elements))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int element_idx = 0; element_idx < num_elements; element_idx++){
            printf("%lf ", host_output_data[element_idx]);
        }
        printf("\n");
    }

    // 释放内存
    cudaFree(device_input_a);
    cudaFree(device_output_data);
}
