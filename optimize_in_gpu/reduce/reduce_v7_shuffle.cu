#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

// 每个线程块中的线程数
#define THREAD_PER_BLOCK 256
// Warp 大小：CUDA 中一个 warp 包含 32 个线程
#define WARP_SIZE 32

/**
 * warpReduceSum: 使用 Shuffle 指令的 Warp 内归约函数
 * 优化：
 * 1. 使用 __shfl_down_sync 指令进行 warp 内数据交换
 * 2. 不需要共享内存，直接在寄存器间交换数据
 * 3. 延迟更低，带宽更高（寄存器访问比共享内存快）
 * 4. 使用 __forceinline__ 强制内联，减少函数调用开销
 * 5. 0xffffffff 是掩码，表示所有 32 个线程都参与
 * 
 * Shuffle 指令说明：
 * __shfl_down_sync(mask, var, delta) 从索引为 (laneId + delta) 的线程获取 var 的值
 * 例如：laneId=0 的线程获取 laneId=16 的线程的值（当 delta=16 时）
 */
template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    // 使用 Shuffle 指令进行 warp 内归约
    // 每个步骤将步长减半，直到所有值归约到 laneId=0 的线程
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);   // 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);   // 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);   // 0-1, 2-3, 4-5, etc.
    return sum;
}

/**
 * reduce7: 使用 Shuffle 指令的最终优化版本
 * 优化：
 * 1. 每个线程处理多个元素（reduce6 的优化）
 * 2. 使用 Shuffle 指令进行 warp 内归约，避免共享内存访问
 * 3. 两阶段归约：
 *    a. 每个 warp 内部使用 Shuffle 指令归约
 *    b. 使用第一个 warp 归约所有 warp 的结果
 * 4. 减少了共享内存的使用和 bank conflict
 * 5. 这是目前最高效的归约实现之一
 * 
 * 性能优势：
 * - Shuffle 指令延迟低（~1 cycle）
 * - 不需要共享内存同步
 * - 寄存器访问比共享内存快
 */
template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce7(float *d_in,float *d_out, unsigned int n){
    // 使用寄存器存储每个线程的部分和
    float sum = 0;

    // 每个线程加载并累加 NUM_PER_THREAD 个元素
    unsigned int tid = threadIdx.x;  // 线程在块内的索引
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    // 展开循环，每个线程处理多个元素
    #pragma unroll
    for(int iter=0; iter<NUM_PER_THREAD; iter++){
        sum += d_in[i+iter*blockSize];
    }
    
    // 共享内存：存储每个 warp 的部分和（每个线程块最多有 WARP_SIZE 个 warp）
    static __shared__ float warpLevelSums[WARP_SIZE]; 
    const int laneId = threadIdx.x % WARP_SIZE;  // 线程在 warp 内的索引（0-31）
    const int warpId = threadIdx.x / WARP_SIZE;   // warp 在块内的索引

    // 第一阶段：使用 Shuffle 指令在每个 warp 内进行归约
    // 结果存储在每个 warp 的 laneId=0 的线程中
    sum = warpReduceSum<blockSize>(sum);

    // 将每个 warp 的归约结果写入共享内存
    if(laneId == 0 )warpLevelSums[warpId] = sum;
    __syncthreads();  // 同步所有 warp，确保所有 warp 的结果都已写入
    
    // 第二阶段：使用第一个 warp 归约所有 warp 的结果
    // 只有前 blockDim.x/WARP_SIZE 个线程参与（即第一个 warp）
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    
    // 在第一个 warp 内进行最终归约
    if (warpId == 0) sum = warpReduceSum<blockSize/WARP_SIZE>(sum); 
    
    // 将当前线程块的归约结果写入全局内存
    if (tid == 0) d_out[blockIdx.x] = sum;
}


/**
 * 检查函数：验证 GPU 计算结果是否正确
 * @param out GPU 计算结果数组
 * @param res CPU 计算的参考结果数组
 * @param n 数组长度
 * @return true 如果结果匹配，false 否则
 */
bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
            return false;
    }
    return true;
}

int main(){
    // 数据大小：32MB 的浮点数数组
    const int N=32*1024*1024;
    
    // 分配主机内存
    float *a=(float *)malloc(N*sizeof(float));
    
    // 分配设备内存
    float *d_a;
    cudaMalloc((void **)&d_a,N*sizeof(float));

    // 固定线程块数量为 1024
    const int block_num = 1024;
    // 计算每个线程块需要处理的元素数量
    const int NUM_PER_BLOCK = N / block_num;
    // 计算每个线程需要处理的元素数量
    const int NUM_PER_THREAD = NUM_PER_BLOCK/THREAD_PER_BLOCK;
    
    // 分配输出数组（每个线程块产生一个结果）
    float *out=(float *)malloc(block_num*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out,block_num*sizeof(float));
    
    // CPU 计算的参考结果，用于验证
    float *res=(float *)malloc(block_num*sizeof(float));

    // 初始化输入数据（使用模运算生成测试数据）
    for(int i=0;i<N;i++){
        a[i]=i%456;
    }

    // CPU 端计算参考结果（每个线程块对应的数据段的和）
    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<NUM_PER_BLOCK;j++){
            if(i * NUM_PER_BLOCK + j < N){
                cur+=a[i * NUM_PER_BLOCK + j];
            }
        }
        res[i]=cur;
    }

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);

    // 配置网格和线程块维度
    dim3 Grid( block_num, 1);  // 网格维度
    dim3 Block( THREAD_PER_BLOCK, 1);  // 线程块维度

    // 多次迭代执行内核，用于性能测试
    int iter = 2000;
    for(int i=0; i<iter; i++){
        reduce7<THREAD_PER_BLOCK, NUM_PER_THREAD><<<Grid,Block>>>(d_a, d_out, N);
    }

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    // 验证结果
    if(check(out,res,block_num))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    // 释放内存
    cudaFree(d_a);
    cudaFree(d_out);
}
