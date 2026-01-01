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
    
    // 计算当前 warp 处理的起始行
    int current_warp_row = (blockDim.y * block_idx_x + thread_idx_y) * ROW_PER_WARP;
    
    // 计算处理一行所需的线程数（warp 大小除以每 warp 处理的行数）
    const int threads_per_row = warp_size / ROW_PER_WARP;
    // 计算线程在处理一行的线程组内的索引
    int lane_id_in_row = lane_id % threads_per_row;
    // 计算当前线程处理的行索引
    int current_thread_row = current_warp_row + lane_id / threads_per_row;

    // 边界检查：确保行索引在有效范围内
    if(current_thread_row < num_rows){
        float partial_sum = 0;  // 累加器
        // 每个线程只处理一个元素
        int current_col = lane_id_in_row;
        // 计算内积的一个元素
        partial_sum += matrix_a[current_thread_row * num_cols + current_col] * vector_x[current_col];
        
        // 使用部分 warp 进行归约（只使用处理当前行的线程）
        partial_sum = warp_reduce_sum<threads_per_row>(partial_sum);
        
        // 将最终结果写入输出向量（由每行的第一个线程写入）
        if(lane_id_in_row == 0) vector_y[current_thread_row] = partial_sum;
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

    const int WARP_SIZE = 32;
    const int ROW_PER_WARP = 2;
    const int THREADS_PER_BLOCK = 128;
    const int WARP_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
    const int ROW_PER_BLOCK = WARP_PER_BLOCK * ROW_PER_WARP;

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
        dim3 grid_dim(num_rows / ROW_PER_BLOCK);
        dim3 block_dim(32, THREADS_PER_BLOCK / WARP_SIZE);
        Sgemv_v2<ROW_PER_WARP><<< grid_dim, block_dim >>>(device_matrix_a, device_vector_x, device_vector_y, num_rows, num_cols);
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
