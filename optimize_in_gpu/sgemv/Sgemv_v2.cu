#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

/**
 * Sgemv_v2: 针对小 N 值优化的 SGEMV 实现
 * 计算 y = A * x，其中 A 是 M×N 矩阵，x 是 N 维向量，y 是 M 维向量
 * 
 * 优化策略：
 * 1. 每个 warp 处理多行（ROW_PER_WARP 行）
 * 2. 将 32 个线程分成多个组，每组处理一行
 * 3. 每个线程只处理一个元素，适合小 N 值的情况
 * 4. 使用模板参数在编译时确定每行使用的线程数
 * 
 * 适用场景：N <= 16
 * 
 * 性能优势：
 * - 提高线程利用率（当 N 很小时，避免线程浪费）
 * - 减少线程块数量
 * - 更好的负载均衡
 * 
 * @tparam ROW_PER_WARP 每个 warp 处理的行数
 */
template <
    const int ROW_PER_WARP  // 每个 warp 处理的行数
    > 
__global__ void Sgemv_v2( 
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
    
    // 计算当前 warp 处理的起始行
    int current_warp_row = (blockDim.y * bx + ty) * ROW_PER_WARP;
    
    // 计算处理一行所需的线程数（warp 大小除以每 warp 处理的行数）
    const int kWarp_size = warp_size / ROW_PER_WARP;
    // 计算线程在处理一行的线程组内的索引
    int kLaneId = laneId % kWarp_size;
    // 计算当前线程处理的行索引
    int current_thread_row = current_warp_row + laneId / kWarp_size;

    // 边界检查：确保行索引在有效范围内
    if(current_thread_row < M){
        float res=0;  // 累加器
        // 每个线程只处理一个元素
        int current_col = kLaneId;
        // 计算内积的一个元素
        res += A[current_thread_row * N + current_col] * x[current_col];
        
        // 使用部分 warp 进行归约（只使用处理当前行的线程）
        res = warpReduceSum<kWarp_size>(res);
        
        // 将最终结果写入输出向量（由每行的第一个线程写入）
        if(kLaneId==0) y[current_thread_row]=res;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("usage: ./main [M] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);

    size_t bytes_A = sizeof(float) * M * N;
    size_t bytes_x = sizeof(float) * N;
    size_t bytes_y = sizeof(float) * M;
    float* h_A = (float*)malloc(bytes_A);
    float* h_x = (float*)malloc(bytes_x);
    float* h_y = (float*)malloc(bytes_y);
    float* h_y1 = (float*)malloc(bytes_y);

    float* d_A;
    float* d_x;
    float* d_y;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_x, bytes_x));
    checkCudaErrors(cudaMalloc(&d_y, bytes_y));

    const int WARP_SIZE=32;
    const int ROW_PER_WARP=2;
    const int THREAD_PER_BLOCK=128;
    const int WARP_PER_BLOCK=THREAD_PER_BLOCK/WARP_SIZE;
    const int ROW_PER_BLOCK=WARP_PER_BLOCK * ROW_PER_WARP;

    // 生成A的数据
    for( int i = 0; i < M * N; i++ ) {
        h_A[i] = (float)i/N;
    }

    // 生成x的数据
    for( int i = 0; i < N; i++ ) {
        h_x[i] = 1;
    }
    memset(h_y,0,M*sizeof(float));
    memset(h_y1,0,M*sizeof(float));

    int nIter = 1000;
    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimGrid(M/ROW_PER_BLOCK);
        dim3 dimBlock(32,THREAD_PER_BLOCK/WARP_SIZE);
        Sgemv_v2<ROW_PER_WARP><<< dimGrid, dimBlock >>>(d_A, d_x, d_y, M, N);
    }
    checkCudaErrors(cudaMemcpy( h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy( d_y, h_y1, bytes_y, cudaMemcpyHostToDevice));
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemv (blas_handle, CUBLAS_OP_T, 
            N, M, &alpha, 
            d_A, N, d_x, 1, &beta, d_y, 1
        );
    }
    checkCudaErrors(cudaMemcpy( h_y1, d_y, bytes_y, cudaMemcpyDeviceToHost));
    cublasDestroy(blas_handle); 
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M; i++) {
        double abs_err = fabs(h_y[i] - h_y1[i]);
        double dot_length = M;
        double abs_val = fabs(h_y[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_y[i], h_y1[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y1);
}
