#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__device__ void warp_reduce(volatile float *cache, unsigned int thread_idx)
{
    cache[thread_idx] += cache[thread_idx + 32];
    //__syncthreads();
    cache[thread_idx] += cache[thread_idx + 16];
    //__syncthreads();
    cache[thread_idx] += cache[thread_idx + 8];
    //__syncthreads();
    cache[thread_idx] += cache[thread_idx + 4];
    //__syncthreads();
    cache[thread_idx] += cache[thread_idx + 2];
    //__syncthreads();
    cache[thread_idx] += cache[thread_idx + 1];
    //__syncthreads();
}
template <unsigned int elements_per_block, unsigned int elements_per_thread>
__global__ void reduce(float *device_input, float *device_output)
{
    int thread_idx = threadIdx.x;
    __shared__ float shared_data[THREADS_PER_BLOCK];

    float *input_begin = device_input + elements_per_block * blockIdx.x;
    shared_data[thread_idx] = 0;
    for (int iteration_idx = 0; iteration_idx < elements_per_thread; iteration_idx++)
        shared_data[thread_idx] += input_begin[thread_idx + iteration_idx * THREADS_PER_BLOCK];
    __syncthreads();

    if (THREADS_PER_BLOCK >= 512)
    {
        if (thread_idx < 256)
            shared_data[thread_idx] += shared_data[thread_idx + 256];
        __syncthreads();
    }
    if (THREADS_PER_BLOCK >= 256)
    {
        if (thread_idx < 128)
            shared_data[thread_idx] += shared_data[thread_idx + 128];
        __syncthreads();
    }
    if (THREADS_PER_BLOCK >= 64)
    {
        if (thread_idx < 64)
            shared_data[thread_idx] += shared_data[thread_idx + 64];
        __syncthreads();
    }

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

    constexpr int num_blocks = 1024;
    constexpr int elements_per_block = num_elements / num_blocks;
    constexpr int elements_per_thread = elements_per_block / THREADS_PER_BLOCK;
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
        for (int element_idx = 0; element_idx < elements_per_block; element_idx++)
        {
            partial_sum += host_input_data[block_idx * elements_per_block + element_idx];
        }
        reference_result[block_idx] = partial_sum;
    }

    cudaMemcpy(device_input_data, host_input_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_dim(num_blocks, 1);
    dim3 block_dim(THREADS_PER_BLOCK, 1);
    for (int iteration_idx = 0; iteration_idx < 10; iteration_idx++)
        reduce<elements_per_block, elements_per_thread><<<grid_dim, block_dim>>>(device_input_data, device_output_data);
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