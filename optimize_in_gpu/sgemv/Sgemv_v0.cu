#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 

// 计算行主序矩阵中从行列索引和行宽（leading dimension）计算偏移量
// 在行主序矩阵中，ld 是矩阵的宽度
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// 使用 float4 向量化加载，一次加载 4 个 float，提高内存带宽利用率
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// CUDA 错误检查宏：检查 CUDA 函数调用是否成功
#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

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
 */
template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    // 使用 Shuffle 指令进行 warp 内归约
    // 每个步骤将步长减半，直到所有值归约到 laneId=0 的线程
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);   // 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);   // 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);   // 0-1, 2-3, 4-5, etc.
    return sum;
}

/**
 * Sgemv_v0: 基础版本的 SGEMV（单精度矩阵向量乘法）实现
 * 计算 y = A * x，其中 A 是 M×N 矩阵，x 是 N 维向量，y 是 M 维向量
 * 
 * 实现策略：
 * 1. 每个线程块处理多行（blockDim.y 行）
 * 2. 每个 warp（32 个线程）处理一行
 * 3. 每个线程计算该行与向量 x 的部分内积
 * 4. 使用 warp 内归约得到最终结果
 * 
 * 适用场景：N == 32 或 N 是 32 的倍数
 * 
 * 优化点：
 * - 使用 warp 内归约减少共享内存使用
 * - 使用 __restrict__ 关键字帮助编译器优化
 */
__global__ void Sgemv_v0( 
    float * __restrict__ A,  // 输入矩阵 A (M×N)
    float * __restrict__ x,  // 输入向量 x (N×1)
    float * __restrict__ y,  // 输出向量 y (M×1)
    const int M,             // 矩阵 A 的行数
    const int N) {           // 矩阵 A 的列数（向量 x 的长度）
    // 线程块索引
    int bx = blockIdx.x;

    // 线程在块内的索引
    int tx = threadIdx.x;  // x 方向线程索引
    int ty = threadIdx.y;  // y 方向线程索引

    const int warp_size=32;  // Warp 大小
    int laneId= tx % warp_size;  // 线程在 warp 内的索引（0-31）
    // 计算当前线程处理的行索引
    int current_row = blockDim.y * bx + ty;

    // 边界检查：确保行索引在有效范围内
    if(current_row < M){
        float res=0;  // 累加器，存储部分内积结果
        // 计算每个线程需要处理的元素数量
        int kIteration = N/warp_size;
        if(kIteration==0) kIteration=1;  // 确保至少处理一个元素
        
        // 展开循环，每个线程处理 kIteration 个元素
        #pragma unroll
        for(int i=0; i< kIteration; i++){
            // 计算当前线程处理的列索引
            int current_col = i*warp_size + laneId;
            // 累加：A[current_row][current_col] * x[current_col]
            res += A[current_row*N + current_col] * x[current_col];
        }
        
        // 使用 warp 内归约将 32 个线程的部分结果归约到 laneId=0 的线程
        res = warpReduceSum<warp_size>(res);
        
        // 将最终结果写入输出向量
        if(laneId==0) y[current_row]=res;
    }
}

/**
 * 主函数：测试 SGEMV 实现的正确性和性能
 * 用法：./main [M] [N]
 * - M: 矩阵 A 的行数
 * - N: 矩阵 A 的列数（向量 x 的长度）
 */
int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc != 3) {
        printf("usage: ./main [M] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);  // 矩阵行数
    size_t N = atoi(argv[2]);  // 矩阵列数

    // 计算所需内存大小
    size_t bytes_A = sizeof(float) * M * N;  // 矩阵 A 的内存大小
    size_t bytes_x = sizeof(float) * N;      // 向量 x 的内存大小
    size_t bytes_y = sizeof(float) * M;     // 向量 y 的内存大小
    
    // 分配主机内存
    float* h_A = (float*)malloc(bytes_A);    // 主机端矩阵 A
    float* h_x = (float*)malloc(bytes_x);   // 主机端向量 x
    float* h_y = (float*)malloc(bytes_y);   // 主机端向量 y（自定义实现的结果）
    float* h_y1 = (float*)malloc(bytes_y);  // 主机端向量 y（cuBLAS 的结果，用于验证）

    // 设备内存指针
    float* d_A;
    float* d_x;
    float* d_y;

    // 分配设备内存
    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_x, bytes_x));
    checkCudaErrors(cudaMalloc(&d_y, bytes_y));

    // 初始化矩阵 A 的数据
    for( int i = 0; i < M * N; i++ ) {
        h_A[i] = (float)i/N;
    }

    // 初始化向量 x 的数据（全为 1）
    for( int i = 0; i < N; i++ ) {
        h_x[i] = 1;
    }
    // 初始化输出向量为 0
    memset(h_y,0,M*sizeof(float));
    memset(h_y1,0,M*sizeof(float));

    // 迭代次数，用于性能测试
    int nIter = 1000;
    
    // 将数据从主机内存复制到设备内存
    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    
    // 执行自定义的 SGEMV 内核
    for (int run = 0 ; run < nIter; run ++ ) {
        // 配置网格和线程块维度
        // 每个线程块处理 4 行，每行由 32 个线程（一个 warp）处理
        dim3 dimGrid(M/4);      // 网格维度：需要 (M/4) 个线程块
        dim3 dimBlock(32,4);    // 线程块维度：32×4 = 128 个线程
        Sgemv_v0<<< dimGrid, dimBlock >>>(d_A, d_x, d_y, M, N);
    }
    // 将结果从设备内存复制回主机内存
    checkCudaErrors(cudaMemcpy( h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    // 使用 cuBLAS 计算参考结果
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;  // 标量乘数
    float beta = 0;     // 标量乘数（不使用累加）
    checkCudaErrors(cudaMemcpy( d_y, h_y1, bytes_y, cudaMemcpyHostToDevice));
    // 执行 cuBLAS SGEMV（注意：使用转置操作 CUBLAS_OP_T）
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemv (blas_handle, CUBLAS_OP_T, 
            N, M, &alpha, 
            d_A, N, d_x, 1, &beta, d_y, 1
        );
    }
    checkCudaErrors(cudaMemcpy( h_y1, d_y, bytes_y, cudaMemcpyDeviceToHost));
    cublasDestroy(blas_handle); 
    
    // 验证结果：比较自定义实现和 cuBLAS 的结果
    double eps = 1.e-6;  // 机器精度阈值
    bool correct = true;
    for (int i = 0; i < M; i++) {
        double abs_err = fabs(h_y[i] - h_y1[i]);  // 绝对误差
        double dot_length = M;
        double abs_val = fabs(h_y[i]);
        double rel_err = abs_err / abs_val / dot_length;  // 相对误差
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_y[i], h_y1[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    
    // 释放内存
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y1);
}
