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
 * Sgemv_v1: 使用向量化加载优化的 SGEMV 实现
 * 计算 y = A * x，其中 A 是 M×N 矩阵，x 是 N 维向量，y 是 M 维向量
 * 
 * 优化策略：
 * 1. 使用 float4 向量化加载，一次加载 4 个 float
 * 2. 减少全局内存访问次数（从 N 次减少到 N/4 次）
 * 3. 提高内存带宽利用率
 * 4. 每个线程处理 4 个元素，减少线程数量需求
 * 
 * 适用场景：N >= 128 且 N 是 128 的倍数（32 * 4 = 128）
 * 
 * 性能优势：
 * - 向量化内存访问提高带宽利用率
 * - 减少循环迭代次数
 * - 更好的指令级并行性
 */
__global__ void Sgemv_v1( 
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
        // 计算每个线程需要处理的 float4 向量数量
        // 每个线程处理 4 个元素，所以迭代次数是 N/(warp_size*4)
        int kIteration = (N/warp_size)/4;
        if(kIteration==0) kIteration=1;  // 确保至少处理一个向量
        
        // 优化：将 A 矩阵的当前行指针提前计算，减少重复计算
        A = &A[current_row*N];
        
        // 展开循环，使用向量化加载
        #pragma unroll
        for(int i=0; i< kIteration; i++){
            // 计算当前 float4 向量的索引
            int current_col_vec = (i*warp_size + laneId);
            // 使用 float4 向量化加载：一次加载 4 个 float
            float4 current_val= reinterpret_cast<float4 *>(A)[current_col_vec];
            float4 current_x = reinterpret_cast<float4 *>(x)[current_col_vec];
            // 计算 4 个元素的内积并累加
            res += current_val.x*current_x.x;
            res += current_val.y*current_x.y;
            res += current_val.z*current_x.z;
            res += current_val.w*current_x.w;
        }
        
        // 使用 warp 内归约将 32 个线程的部分结果归约到 laneId=0 的线程
        res = warpReduceSum<warp_size>(res);
        
        // 将最终结果写入输出向量
        if(laneId==0) y[current_row]=res;
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
        dim3 dimGrid(M/4);
        dim3 dimBlock(32,4);
        Sgemv_v1<<< dimGrid, dimBlock >>>(d_A, d_x, d_y, M, N);
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
