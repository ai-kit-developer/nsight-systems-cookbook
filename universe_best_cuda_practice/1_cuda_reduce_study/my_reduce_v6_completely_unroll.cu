#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

/**
 * Warp级别的归约函数
 * 在warp内（32个线程）进行完全展开的归约操作
 * 使用volatile关键字确保编译器不会优化掉这些内存访问
 * 
 * @param cache 共享内存数组指针
 * @param tid 线程ID
 */
__device__ void warpReduce(volatile float *cache, unsigned int tid)
{
    cache[tid] += cache[tid + 32];
    //__syncthreads();  // warp内不需要同步
    cache[tid] += cache[tid + 16];
    //__syncthreads();
    cache[tid] += cache[tid + 8];
    //__syncthreads();
    cache[tid] += cache[tid + 4];
    //__syncthreads();
    cache[tid] += cache[tid + 2];
    //__syncthreads();
    cache[tid] += cache[tid + 1];
    //__syncthreads();
}

/**
 * Reduce操作版本6：完全展开归约循环
 * 相比v5版本，使用条件编译完全展开归约循环，消除循环开销
 * 优点：编译器可以更好地优化代码，提高执行效率
 * 
 * @param d_input 输入数组的全局内存指针
 * @param d_output 输出数组的全局内存指针，每个block输出一个结果
 */
__global__ void reduce(float *d_input, float *d_output)
{
    int tid = threadIdx.x;
    __shared__ float shared[THREAD_PER_BLOCK];
    
    // 每个block处理2倍的数据
    float *input_begin = d_input + blockDim.x * blockIdx.x * 2;
    
    // 每个线程加载2个元素并立即相加
    shared[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
    __syncthreads();
    
    // 完全展开的归约循环：使用条件编译消除循环
    // 注释掉的代码是原来的循环版本：
    // #pragma unroll
    //     for (int i = blockDim.x / 2; i > 32; i /= 2)
    //     {
    //         if (tid < i)
    //             shared[tid] += shared[tid + i];
    //         __syncthreads();
    //     }
    
    // 根据THREAD_PER_BLOCK的值，展开归约步骤
    if (THREAD_PER_BLOCK >= 512)
    {
        if (tid < 256)
            shared[tid] += shared[tid + 256];
        __syncthreads();
    }
    if (THREAD_PER_BLOCK >= 256)
    {
        if (tid < 128)
            shared[tid] += shared[tid + 128];
        __syncthreads();
    }
    if (THREAD_PER_BLOCK >= 64)
    {
        if (tid < 64)
            shared[tid] += shared[tid + 64];
        __syncthreads();
    }

    // 最后32个元素使用warp归约（不需要__syncthreads）
    if (tid < 32)
    {
        warpReduce(shared, tid);
    }
    
    // 只有thread 0将结果写入全局内存
    if (tid == 0)
        d_output[blockIdx.x] = shared[0];
}

bool check(float *out, float *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (abs(out[i] - res[i]) > 0.005)
            return false;
    }
    return true;
}

int main()
{
    // printf("hello reduce\n");
    const int N = 32 * 1024 * 1024;
    float *input = (float *)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));

    int block_num = N / THREAD_PER_BLOCK / 2;
    float *output = (float *)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));
    float *result = (float *)malloc(block_num * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }
    // cpu calc
    for (int i = 0; i < block_num; i++)
    {
        float cur = 0;
        for (int j = 0; j < 2 * THREAD_PER_BLOCK; j++)
        {
            cur += input[i * 2 * THREAD_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    for (int i = 0; i < 10; i++)
        reduce<<<Grid, Block>>>(d_input, d_output);
    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, result, block_num))
        printf("the ans is right\n");
    else
    {
        printf("the ans is wrong\n");
        for (int i = 0; i < block_num; i++)
        {
            printf("%lf ", output[i]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
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