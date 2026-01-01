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

template <unsigned int warp_size>
__device__ __forceinline__ float warp_reduce_sum(float sum) {
    if (warp_size >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (warp_size >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (warp_size >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (warp_size >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (warp_size >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
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
    float * __restrict__ matrix_a,  // 输入矩阵 A (M×N)
    float * __restrict__ vector_x,  // 输入向量 x (N×1)
    float * __restrict__ vector_y,  // 输出向量 y (M×1)
    const int num_rows,             // 矩阵 A 的行数
    const int num_cols) {           // 矩阵 A 的列数（向量 x 的长度）
    // 线程块索引
    int block_idx_x = blockIdx.x;

    // 线程在块内的索引
    int thread_idx_x = threadIdx.x;  // x 方向线程索引
    int thread_idx_y = threadIdx.y;  // y 方向线程索引

    const int warp_size = 32;  // Warp 大小
    int lane_id = thread_idx_x % warp_size;  // 线程在 warp 内的索引（0-31）
    // 计算当前线程处理的行索引
    int current_row = blockDim.y * block_idx_x + thread_idx_y;

    // 边界检查：确保行索引在有效范围内
    if(current_row < num_rows){
        float partial_sum = 0;  // 累加器，存储部分内积结果
        // 计算每个线程需要处理的 float4 向量数量
        // 每个线程处理 4 个元素，所以迭代次数是 num_cols/(warp_size*4)
        int num_iterations = (num_cols / warp_size) / 4;
        if(num_iterations == 0) num_iterations = 1;  // 确保至少处理一个向量
        
        // 优化：将 matrix_a 矩阵的当前行指针提前计算，减少重复计算
        matrix_a = &matrix_a[current_row * num_cols];
        
        // 展开循环，使用向量化加载
        #pragma unroll
        for(int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++){
            // 计算当前 float4 向量的索引
            int current_col_vec = (iteration_idx * warp_size + lane_id);
            // 使用 float4 向量化加载：一次加载 4 个 float
            float4 current_val = reinterpret_cast<float4 *>(matrix_a)[current_col_vec];
            float4 current_x = reinterpret_cast<float4 *>(vector_x)[current_col_vec];
            // 计算 4 个元素的内积并累加
            partial_sum += current_val.x * current_x.x;
            partial_sum += current_val.y * current_x.y;
            partial_sum += current_val.z * current_x.z;
            partial_sum += current_val.w * current_x.w;
        }
        
        // 使用 warp 内归约将 32 个线程的部分结果归约到 lane_id=0 的线程
        partial_sum = warp_reduce_sum<warp_size>(partial_sum);
        
        // 将最终结果写入输出向量
        if(lane_id == 0) vector_y[current_row] = partial_sum;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("usage: ./main [M] [N]\n");
        exit(0);
    }
    size_t num_rows = atoi(argv[1]);
    size_t num_cols = atoi(argv[2]);

    size_t bytes_matrix_a = sizeof(float) * num_rows * num_cols;
    size_t bytes_vector_x = sizeof(float) * num_cols;
    size_t bytes_vector_y = sizeof(float) * num_rows;
    float* host_matrix_a = (float*)malloc(bytes_matrix_a);
    float* host_vector_x = (float*)malloc(bytes_vector_x);
    float* host_vector_y = (float*)malloc(bytes_vector_y);
    float* host_vector_y_ref = (float*)malloc(bytes_vector_y);

    float* device_matrix_a;
    float* device_vector_x;
    float* device_vector_y;

    checkCudaErrors(cudaMalloc(&device_matrix_a, bytes_matrix_a));
    checkCudaErrors(cudaMalloc(&device_vector_x, bytes_vector_x));
    checkCudaErrors(cudaMalloc(&device_vector_y, bytes_vector_y));

    // 生成A的数据
    for( int element_idx = 0; element_idx < num_rows * num_cols; element_idx++ ) {
        host_matrix_a[element_idx] = (float)element_idx / num_cols;
    }

    // 生成x的数据
    for( int element_idx = 0; element_idx < num_cols; element_idx++ ) {
        host_vector_x[element_idx] = 1;
    }
    memset(host_vector_y, 0, num_rows * sizeof(float));
    memset(host_vector_y_ref, 0, num_rows * sizeof(float));

    int num_iterations = 1000;
    checkCudaErrors(cudaMemcpy(device_matrix_a, host_matrix_a, bytes_matrix_a, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_vector_x, host_vector_x, bytes_vector_x, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_vector_y, host_vector_y, bytes_vector_y, cudaMemcpyHostToDevice));
    for (int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++ ) {
        dim3 grid_dim(num_rows / 4);
        dim3 block_dim(32, 4);
        Sgemv_v1<<< grid_dim, block_dim >>>(device_matrix_a, device_vector_x, device_vector_y, num_rows, num_cols);
    }
    checkCudaErrors(cudaMemcpy(host_vector_y, device_vector_y, bytes_vector_y, cudaMemcpyDeviceToHost));

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy(device_vector_y, host_vector_y_ref, bytes_vector_y, cudaMemcpyHostToDevice));
    for (int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++ ) {
        cublasSgemv (blas_handle, CUBLAS_OP_T, 
            num_cols, num_rows, &alpha, 
            device_matrix_a, num_cols, device_vector_x, 1, &beta, device_vector_y, 1
        );
    }
    checkCudaErrors(cudaMemcpy(host_vector_y_ref, device_vector_y, bytes_vector_y, cudaMemcpyDeviceToHost));
    cublasDestroy(blas_handle); 
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int element_idx = 0; element_idx < num_rows; element_idx++) {
        double abs_err = fabs(host_vector_y[element_idx] - host_vector_y_ref[element_idx]);
        double dot_length = num_rows;
        double abs_val = fabs(host_vector_y[element_idx]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    element_idx, host_vector_y[element_idx], host_vector_y_ref[element_idx], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    
    // Free Memory
    cudaFree(device_matrix_a);
    cudaFree(device_vector_x);
    cudaFree(device_vector_y);
    
    free(host_matrix_a);
    free(host_vector_x);
    free(host_vector_y);
    free(host_vector_y_ref);
}
