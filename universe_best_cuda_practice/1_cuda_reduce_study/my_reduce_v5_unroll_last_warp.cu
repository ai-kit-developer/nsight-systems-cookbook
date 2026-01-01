#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

/**
 * Warp级别的归约函数
 * 在warp内（32个线程）进行完全展开的归约操作
 * 使用volatile关键字确保编译器不会优化掉这些内存访问
 * 注意：warp内不需要__syncthreads()，因为warp内线程是同步执行的
 * 
 * @param cache 共享内存数组指针
 * @param thread_idx 线程ID
 */
__device__ void warp_reduce(volatile float *cache, unsigned int thread_idx)
{
    // Warp内归约：完全展开，避免循环开销
    // 第1步：thread 0-31 读取 thread 32-63 的值（如果存在）
    cache[thread_idx] += cache[thread_idx + 32];
    //__syncthreads();  // warp内不需要同步
    // 第2步：thread 0-15 读取 thread 16-31 的值
    cache[thread_idx] += cache[thread_idx + 16];
    //__syncthreads();
    // 第3步：thread 0-7 读取 thread 8-15 的值
    cache[thread_idx] += cache[thread_idx + 8];
    //__syncthreads();
    // 第4步：thread 0-3 读取 thread 4-7 的值
    cache[thread_idx] += cache[thread_idx + 4];
    //__syncthreads();
    // 第5步：thread 0-1 读取 thread 2-3 的值
    cache[thread_idx] += cache[thread_idx + 2];
    //__syncthreads();
    // 第6步：thread 0 读取 thread 1 的值
    cache[thread_idx] += cache[thread_idx + 1];
    //__syncthreads();
}

/**
 * Reduce操作版本5：展开最后一个warp的归约
 * 相比v4版本，当剩余元素少于等于32时，使用专门的warp归约函数
 * 优点：减少同步开销，提高最后几轮归约的效率
 * 
 * @param device_input 输入数组的全局内存指针
 * @param device_output 输出数组的全局内存指针，每个block输出一个结果
 */
__global__ void reduce(float *device_input, float *device_output)
{
    int thread_idx = threadIdx.x;
    __shared__ float shared_data[THREADS_PER_BLOCK];
    
    // 每个线程加载2个元素并相加
    float *input_begin = device_input + blockDim.x * blockIdx.x * 2;
    shared_data[thread_idx] = input_begin[thread_idx] + input_begin[thread_idx + blockDim.x];
    __syncthreads();

    // 归约循环：直到剩余元素数大于32
    for (int stride = blockDim.x / 2; stride > 32; stride /= 2)
    {
        if (thread_idx < stride)
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        __syncthreads();
    }
    
    // 最后32个元素使用warp归约（不需要__syncthreads）
    if (thread_idx < 32)
    {
        warp_reduce(shared_data, thread_idx);
    }
    if (thread_idx == 0)
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