#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

/**
 * Reduce操作版本8：完全使用shuffle指令（与v7相同，但命名不同）
 * 这是reduce优化的最终版本，完全使用warp shuffle指令进行归约
 * 优点：最大化利用warp shuffle的性能优势，减少共享内存使用
 * 
 * @tparam elements_per_block 每个block处理的元素数量
 * @tparam elements_per_thread 每个线程处理的元素数量
 * @param device_input 输入数组的全局内存指针
 * @param device_output 输出数组的全局内存指针，每个block输出一个结果
 */
template <unsigned int elements_per_block, unsigned int elements_per_thread>
__global__ void reduce(float *device_input, float *device_output)
{
    int thread_idx = threadIdx.x;
    // 使用寄存器存储每个线程的局部和
    float sum = 0.f;
    
    // 计算当前block负责的输入数据起始位置
    float *input_begin = device_input + elements_per_block * blockIdx.x;

    // 每个线程处理多个元素，累加到寄存器中
    for (int iteration_idx = 0; iteration_idx < elements_per_thread; iteration_idx++)
        sum += input_begin[thread_idx + iteration_idx * THREADS_PER_BLOCK];

    // 第一级：warp内归约，使用shuffle指令
    // __shfl_down_sync: 从lane_id + delta的线程获取值并累加
    // 0xffffffff: 所有32个lane都参与（掩码）
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // lane 0-15 从 lane 16-31 获取值
    sum += __shfl_down_sync(0xffffffff, sum, 8);   // lane 0-7 从 lane 8-15 获取值
    sum += __shfl_down_sync(0xffffffff, sum, 4);   // lane 0-3 从 lane 4-7 获取值
    sum += __shfl_down_sync(0xffffffff, sum, 2);   // lane 0-1 从 lane 2-3 获取值
    sum += __shfl_down_sync(0xffffffff, sum, 1);   // lane 0 从 lane 1 获取值
    // 此时每个warp的lane 0包含该warp的归约结果

    // 第二级：跨warp归约
    // 使用共享内存存储每个warp的归约结果
    __shared__ float warp_level_sums[32];
    const int lane_id = thread_idx % 32;  // warp内的lane ID (0-31)
    const int warp_id = thread_idx / 32;  // warp ID
    
    // 每个warp的lane 0将其结果写入共享内存
    if (lane_id == 0)
        warp_level_sums[warp_id] = sum;

    __syncthreads();

    // 第三级：对warp级别的结果再次使用shuffle归约
    if (warp_id == 0)
    {
        // 从共享内存读取warp级别的结果（如果存在）
        sum = (lane_id < blockDim.x / 32) ? warp_level_sums[lane_id] : 0.f;
        // 再次使用shuffle进行归约
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
        // 此时只有thread 0有最终结果
    }

    // 只有thread 0将结果写入全局内存
    if (thread_idx == 0)
        device_output[blockIdx.x] = sum;
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
