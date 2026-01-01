#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 每个线程块中的线程数
#define THREADS_PER_BLOCK 256

/**
 * Reduce操作版本0：使用全局内存
 * 这是最基础的reduce实现，直接在全局内存上进行归约操作
 * 缺点：全局内存访问延迟高，性能较差
 * 
 * @param device_input 输入数组的全局内存指针
 * @param device_output 输出数组的全局内存指针，每个block输出一个结果
 */
__global__ void reduce(float *device_input, float *device_output)
{
    // 计算当前block负责的输入数据起始位置
    float *input_begin = device_input + blockDim.x * blockIdx.x;
    
    // 使用二分归约算法：每次迭代将数据量减半
    // 第1次迭代：thread 0,2,4,6... 将 thread 1,3,5,7... 的值加到自己的位置
    // 第2次迭代：thread 0,4,8,12... 将 thread 2,6,10,14... 的值加到自己的位置
    // 以此类推，直到只剩下thread 0有最终结果
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // 只有满足条件的线程参与归约，避免数据竞争
        // threadIdx.x % (stride * 2) == 0 确保每个线程只参与一次归约
        if (threadIdx.x % (stride * 2) == 0)
            input_begin[threadIdx.x] += input_begin[threadIdx.x + stride];
        // 同步所有线程，确保所有加法操作完成后再进行下一轮
        __syncthreads();
    }
    // 注释说明：上述循环的展开形式示例
    // if (threadIdx.x == 0 or 2 or 4 or 6)
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 1];
    // if (threadIdx.x == 0 or 4)
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 2];
    // if (threadIdx.x == 0)
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 4];
    
    // 只有thread 0将结果写入输出数组
    if (threadIdx.x == 0)
        device_output[blockIdx.x] = input_begin[0];
}

/**
 * 验证GPU计算结果与CPU计算结果是否一致
 * 
 * @param host_output_data GPU计算的结果数组
 * @param reference_result CPU计算的结果数组（参考值）
 * @param num_elements 数组长度
 * @return true 如果所有元素的误差都在允许范围内
 */
bool verify_result(float *host_output_data, float *reference_result, int num_elements)
{
    for (int element_idx = 0; element_idx < num_elements; element_idx++)
    {
        // 允许的误差范围：0.005
        if (abs(host_output_data[element_idx] - reference_result[element_idx]) > 0.005)
            return false;
    }
    return true;
}

int main()
{
    // 数据规模：32M个浮点数
    const int num_elements = 32 * 1024 * 1024;
    
    // 在主机端分配输入数组内存
    float *host_input_data = (float *)malloc(num_elements * sizeof(float));
    
    // 在设备端分配输入数组内存
    float *device_input_data;
    cudaMalloc((void **)&device_input_data, num_elements * sizeof(float));

    // 计算需要的block数量：每个block处理THREADS_PER_BLOCK个元素
    int num_blocks = num_elements / THREADS_PER_BLOCK;
    
    // 在主机端分配输出数组内存（每个block输出一个结果）
    float *host_output_data = (float *)malloc((num_elements / THREADS_PER_BLOCK) * sizeof(float));
    
    // 在设备端分配输出数组内存
    float *device_output_data;
    cudaMalloc((void **)&device_output_data, (num_elements / THREADS_PER_BLOCK) * sizeof(float));
    
    // CPU计算结果（用于验证）
    float *reference_result = (float *)malloc((num_elements / THREADS_PER_BLOCK) * sizeof(float));
    
    // 初始化输入数据：生成[-1, 1]范围内的随机数
    for (int element_idx = 0; element_idx < num_elements; element_idx++)
    {
        host_input_data[element_idx] = 2.0 * (float)drand48() - 1.0;
    }
    
    // CPU端计算：为每个block计算reduce结果（用于验证GPU结果）
    for (int block_idx = 0; block_idx < num_blocks; block_idx++)
    {
        float partial_sum = 0;
        for (int thread_idx = 0; thread_idx < THREADS_PER_BLOCK; thread_idx++)
        {
            partial_sum += host_input_data[block_idx * THREADS_PER_BLOCK + thread_idx];
        }
        reference_result[block_idx] = partial_sum;
    }

    // 将输入数据从主机内存拷贝到设备内存
    cudaMemcpy(device_input_data, host_input_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // 配置kernel启动参数
    dim3 grid_dim(num_elements / THREADS_PER_BLOCK, 1);  // Grid维度：每个block处理THREADS_PER_BLOCK个元素
    dim3 block_dim(THREADS_PER_BLOCK, 1);      // Block维度：每个block有THREADS_PER_BLOCK个线程
    
    // 执行kernel多次（用于性能测试）
    for (int iteration_idx = 0; iteration_idx < 50; iteration_idx++)
        reduce<<<grid_dim, block_dim>>>(device_input_data, device_output_data);
    
    // 将结果从设备内存拷贝回主机内存
    cudaMemcpy(host_output_data, device_output_data, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证GPU计算结果是否正确
    if (verify_result(host_output_data, reference_result, num_blocks))
        printf("the ans is right\n");
    else
    {
        printf("the ans is wrong\n");
        // 打印错误的输出值用于调试
        for (int block_idx = 0; block_idx < num_blocks; block_idx++)
        {
            printf("%lf ", host_output_data[block_idx]);
        }
        printf("\n");
    }

    // 释放设备内存
    cudaFree(device_input_data);
    cudaFree(device_output_data);
    return 0;
}
// "command" :
// "/usr/local/cuda-12.2/bin/nvcc
// -forward-unknown-to-host-compiler
// -isystem=/usr/local/cuda-12.2/include
// -g
// --generate-code=arch=compute_52,code=[compute_52,sm_52]
// -G
// -x cu
// -dc /home/hongkailin/universe_best_cuda_practice/1_cuda_reduce_study/my_reduce_v0_global_memory.cu
// -o CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.o",