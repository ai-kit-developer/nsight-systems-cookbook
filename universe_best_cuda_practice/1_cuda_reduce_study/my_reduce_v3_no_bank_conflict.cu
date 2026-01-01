#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

/**
 * Reduce操作版本3：消除共享内存bank冲突
 * 相比v2版本，改变了归约的方向和索引方式
 * 从 i = 1 递增改为 i = blockDim.x/2 递减，避免共享内存bank冲突
 * 优点：减少共享内存访问冲突，提高内存带宽利用率
 * 
 * @param d_input 输入数组的全局内存指针
 * @param d_output 输出数组的全局内存指针，每个block输出一个结果
 */
__global__ void reduce(float *d_input, float *d_output)
{
    // 声明共享内存数组
    __shared__ float shared[THREAD_PER_BLOCK];
    
    // 计算当前block负责的输入数据起始位置
    float *input_begin = d_input + blockDim.x * blockIdx.x;
    
    // 将全局内存中的数据加载到共享内存
    shared[threadIdx.x] = input_begin[threadIdx.x];
    __syncthreads();

    // 改进的归约循环：从大到小递减，避免bank冲突
    // 第1轮：thread 0-127 读取 thread 128-255 的值
    // 第2轮：thread 0-63 读取 thread 64-127 的值
    // 以此类推，直到只剩下thread 0
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        // 只有前i个线程参与归约，索引连续，避免bank冲突
        if (threadIdx.x < i)
            shared[threadIdx.x] += shared[threadIdx.x + i];
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

    int block_num = N / THREAD_PER_BLOCK;
    float *output = (float *)malloc((N / THREAD_PER_BLOCK) * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, (N / THREAD_PER_BLOCK) * sizeof(float));
    float *result = (float *)malloc((N / THREAD_PER_BLOCK) * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }
    // cpu calc
    for (int i = 0; i < block_num; i++)
    {
        float cur = 0;
        for (int j = 0; j < THREAD_PER_BLOCK; j++)
        {
            cur += input[i * THREAD_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(N / THREAD_PER_BLOCK, 1);
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