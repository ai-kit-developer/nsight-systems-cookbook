#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

/**
 * Reduce操作版本3：消除共享内存bank冲突
 * 相比v2版本，改变了归约的方向和索引方式
 * 从 stride = 1 递增改为 stride = blockDim.x/2 递减，避免共享内存bank冲突
 * 优点：减少共享内存访问冲突，提高内存带宽利用率
 * 
 * @param device_input 输入数组的全局内存指针
 * @param device_output 输出数组的全局内存指针，每个block输出一个结果
 */
__global__ void reduce(float *device_input, float *device_output)
{
    // 声明共享内存数组
    __shared__ float shared_data[THREADS_PER_BLOCK];
    
    // 计算当前block负责的输入数据起始位置
    float *input_begin = device_input + blockDim.x * blockIdx.x;
    
    // 将全局内存中的数据加载到共享内存
    shared_data[threadIdx.x] = input_begin[threadIdx.x];
    __syncthreads();

    // 改进的归约循环：从大到小递减，避免bank冲突
    // 第1轮：thread 0-127 读取 thread 128-255 的值
    // 第2轮：thread 0-63 读取 thread 64-127 的值
    // 以此类推，直到只剩下thread 0
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        // 只有前stride个线程参与归约，索引连续，避免bank冲突
        if (threadIdx.x < stride)
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        // 同步所有线程
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

    int num_blocks = num_elements / THREADS_PER_BLOCK;
    float *host_output_data = (float *)malloc((num_elements / THREADS_PER_BLOCK) * sizeof(float));
    float *device_output_data;
    cudaMalloc((void **)&device_output_data, (num_elements / THREADS_PER_BLOCK) * sizeof(float));
    float *reference_result = (float *)malloc((num_elements / THREADS_PER_BLOCK) * sizeof(float));
    for (int element_idx = 0; element_idx < num_elements; element_idx++)
    {
        host_input_data[element_idx] = 2.0 * (float)drand48() - 1.0;
    }
    // cpu calc
    for (int block_idx = 0; block_idx < num_blocks; block_idx++)
    {
        float partial_sum = 0;
        for (int thread_idx = 0; thread_idx < THREADS_PER_BLOCK; thread_idx++)
        {
            partial_sum += host_input_data[block_idx * THREADS_PER_BLOCK + thread_idx];
        }
        reference_result[block_idx] = partial_sum;
    }

    cudaMemcpy(device_input_data, host_input_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_dim(num_elements / THREADS_PER_BLOCK, 1);
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