#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

/**
 * Reduce操作版本8：完全使用shuffle指令（与v7相同，但命名不同）
 * 这是reduce优化的最终版本，完全使用warp shuffle指令进行归约
 * 优点：最大化利用warp shuffle的性能优势，减少共享内存使用
 * 
 * @tparam NUM_PER_BLOCK 每个block处理的元素数量
 * @tparam NUM_PER_THREAD 每个线程处理的元素数量
 * @param d_input 输入数组的全局内存指针
 * @param d_output 输出数组的全局内存指针，每个block输出一个结果
 */
template <unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce(float *d_input, float *d_output)
{
    int tid = threadIdx.x;
    // 使用寄存器存储每个线程的局部和
    float sum = 0.f;
    
    // 计算当前block负责的输入数据起始位置
    float *input_begin = d_input + NUM_PER_BLOCK * blockIdx.x;

    // 每个线程处理多个元素，累加到寄存器中
    for (int i = 0; i < NUM_PER_THREAD; i++)
        sum += input_begin[tid + i * THREAD_PER_BLOCK];

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
    __shared__ float warpLevelSums[32];
    const int laneId = tid % 32;  // warp内的lane ID (0-31)
    const int warpId = tid / 32;  // warp ID
    
    // 每个warp的lane 0将其结果写入共享内存
    if (laneId == 0)
        warpLevelSums[warpId] = sum;

    __syncthreads();

    // 第三级：对warp级别的结果再次使用shuffle归约
    if (warpId == 0)
    {
        // 从共享内存读取warp级别的结果（如果存在）
        sum = (laneId < blockDim.x / 32) ? warpLevelSums[laneId] : 0.f;
        // 再次使用shuffle进行归约
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
        // 此时只有thread 0有最终结果
    }

    // 只有thread 0将结果写入全局内存
    if (tid == 0)
        d_output[blockIdx.x] = sum;
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

    constexpr int block_num = 1024;
    constexpr int num_per_block = N / block_num;
    constexpr int num_per_thread = num_per_block / THREAD_PER_BLOCK;
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
        for (int j = 0; j < num_per_block; j++)
        {
            cur += input[i * num_per_block + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    for (int i = 0; i < 10; i++)
        reduce<num_per_block, num_per_thread><<<Grid, Block>>>(d_input, d_output);
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
