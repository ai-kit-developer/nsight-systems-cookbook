#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

/**
 * Reduce操作版本4（方案A）：加载时进行第一次归约
 * 相比v3版本，每个线程在加载数据时同时处理2个元素，减少内存访问次数
 * 优点：提高内存访问效率，减少全局内存访问次数
 * 
 * @param device_input 输入数组的全局内存指针
 * @param device_output 输出数组的全局内存指针，每个block输出一个结果
 */
__global__ void reduce(float *device_input, float *device_output)
{
    // 声明共享内存数组
    __shared__ float shared_data[THREADS_PER_BLOCK];
    
    // 每个block处理2倍的数据：blockDim.x * 2 个元素
    // 计算当前block负责的输入数据起始位置
    float *input_begin = device_input + blockDim.x * blockIdx.x * 2;
    
    // 优化：每个线程加载2个元素并立即相加，减少后续归约轮数
    // thread 0 处理 input_begin[0] 和 input_begin[256]
    // thread 1 处理 input_begin[1] 和 input_begin[257]
    // 以此类推
    shared_data[threadIdx.x] = input_begin[threadIdx.x] + input_begin[threadIdx.x + blockDim.x];
    __syncthreads();

    // 在共享内存上进行二分归约（从blockDim.x/2开始）
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        __syncthreads();
    }
    // if (threadIdx.x == 0 or 2 or 4 or 6)
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 1];
    // if (threadIdx.x == 0 or 4)
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 2];
    // if (threadIdx.x == 0)
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 4];
    if (threadIdx.x == 0)
        device_output[blockIdx.x] = shared_data[0];
}

bool verify_result(float *host_output_data, float *reference_result, int num_elements)
{
    for (int element_idx = 0; element_idx < num_elements; element_idx++)
    {
        if (abs(host_output_data[element_idx] - reference_result[element_idx]) > 0.005)
            return false;
    }
    return true;
}

int main()
{
    // printf("hello reduce\n");
    const int num_elements = 32 * 1024 * 1024;
    float *host_input_data = (float *)malloc(num_elements * sizeof(float));
    float *device_input_data;
    cudaMalloc((void **)&device_input_data, num_elements * sizeof(float));

    int num_blocks = num_elements / THREADS_PER_BLOCK / 2;
    float *host_output_data = (float *)malloc(num_blocks * sizeof(float));
    float *device_output_data;
    cudaMalloc((void **)&device_output_data, num_blocks * sizeof(float));
    float *reference_result = (float *)malloc(num_blocks * sizeof(float));
    for (int element_idx = 0; element_idx < num_elements; element_idx++)
    {
        host_input_data[element_idx] = 2.0 * (float)drand48() - 1.0;
    }
    // cpu calc
    for (int block_idx = 0; block_idx < num_blocks; block_idx++)
    {
        float partial_sum = 0;
        for (int element_idx = 0; element_idx < 2 * THREADS_PER_BLOCK; element_idx++)
        {
            partial_sum += host_input_data[block_idx * 2 * THREADS_PER_BLOCK + element_idx];
        }
        reference_result[block_idx] = partial_sum;
    }

    cudaMemcpy(device_input_data, host_input_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(THREADS_PER_BLOCK, 1);
    for (int iteration_idx = 0; iteration_idx < 10; iteration_idx++)
        reduce<<<grid_dim, block_dim>>>(device_input_data, device_output_data);
    cudaMemcpy(host_output_data, device_output_data, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    if (verify_result(host_output_data, reference_result, num_blocks))
        printf("the ans is right\n");
    else
    {
        printf("the ans is wrong\n");
        for (int block_idx = 0; block_idx < num_blocks; block_idx++)
        {
            printf("%lf ", host_output_data[block_idx]);
        }
        printf("\n");
    }

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