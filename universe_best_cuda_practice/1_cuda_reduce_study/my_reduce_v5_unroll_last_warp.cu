#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

/**
 * Warp级别的归约函数
 * 在warp内（32个线程）进行完全展开的归约操作
 * 使用volatile关键字确保编译器不会优化掉这些内存访问
 * 注意：warp内不需要__syncthreads()，因为warp内线程是同步执行的
 * 
 * @param cache 共享内存数组指针
 * @param tid 线程ID
 */
__device__ void warpReduce(volatile float *cache, unsigned int tid)
{
    // Warp内归约：完全展开，避免循环开销
    // 第1步：thread 0-31 读取 thread 32-63 的值（如果存在）
    cache[tid] += cache[tid + 32];
    //__syncthreads();  // warp内不需要同步
    // 第2步：thread 0-15 读取 thread 16-31 的值
    cache[tid] += cache[tid + 16];
    //__syncthreads();
    // 第3步：thread 0-7 读取 thread 8-15 的值
    cache[tid] += cache[tid + 8];
    //__syncthreads();
    // 第4步：thread 0-3 读取 thread 4-7 的值
    cache[tid] += cache[tid + 4];
    //__syncthreads();
    // 第5步：thread 0-1 读取 thread 2-3 的值
    cache[tid] += cache[tid + 2];
    //__syncthreads();
    // 第6步：thread 0 读取 thread 1 的值
    cache[tid] += cache[tid + 1];
    //__syncthreads();
}

/**
 * Reduce操作版本5：展开最后一个warp的归约
 * 相比v4版本，当剩余元素少于等于32时，使用专门的warp归约函数
 * 优点：减少同步开销，提高最后几轮归约的效率
 * 
 * @param d_input 输入数组的全局内存指针
 * @param d_output 输出数组的全局内存指针，每个block输出一个结果
 */
__global__ void reduce(float *d_input, float *d_output)
{
    int tid = threadIdx.x;
    __shared__ float shared[THREAD_PER_BLOCK];
    
    // 每个线程加载2个元素并相加
    float *input_begin = d_input + blockDim.x * blockIdx.x * 2;
    shared[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
    __syncthreads();

    // 归约循环：直到剩余元素数大于32
    for (int i = blockDim.x / 2; i > 32; i /= 2)
    {
        if (tid < i)
            shared[tid] += shared[tid + i];
        __syncthreads();
    }
    
    // 最后32个元素使用warp归约（不需要__syncthreads）
    if (tid < 32)
    {
        warpReduce(shared, tid);
    }
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