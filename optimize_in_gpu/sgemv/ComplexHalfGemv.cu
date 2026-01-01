#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 
#include <cuComplex.h>
#include <thrust/complex.h>
#include "cuHalfComplex.cuh"

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

/**
 * warp_reduce_sum: 使用 Shuffle 指令的 Warp 内归约函数（half 精度版本）
 * 用于对 half 类型的值进行 warp 内归约
 * 优化策略与 float 版本相同，但使用 half 精度以节省内存和带宽
 */
template <unsigned int warp_size>
__device__ __forceinline__ half warp_reduce_sum(half sum) {
    if (warp_size >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (warp_size >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if (warp_size >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);   // 0-4, 1-5, 2-6, etc.
    if (warp_size >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);   // 0-2, 1-3, 4-6, 5-7, etc.
    if (warp_size >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);   // 0-1, 2-3, 4-5, etc.
    return sum;
}

/**
 * ComplexHalfGemv: 半精度复数矩阵向量乘法内核
 * 计算 y = A * x，其中 A 是 M×N 复数矩阵，x 是 N 维复数向量，y 是 M 维复数向量
 * 
 * 优化策略：
 * 1. 使用 half 精度（FP16）减少内存占用和带宽需求
 * 2. 使用 float4 向量化加载，一次加载 4 个 half（相当于 2 个复数）
 * 3. 每个线程处理多个复数元素，提高计算密度
 * 4. 使用 warp 内归约减少共享内存使用
 * 
 * 适用场景：
 * - 需要高吞吐量的复数矩阵向量乘法
 * - 对精度要求不是特别严格的场景
 * - 内存带宽受限的场景
 * 
 * 注意：half 精度在 CPU 上不支持，所以验证时需要使用 float 精度
 */
__global__ void ComplexHalfGemv( 
    cuHalfComplex * __restrict__ matrix_a,  // 输入复数矩阵 A (M×N)
    cuHalfComplex * __restrict__ vector_x,   // 输入复数向量 x (N×1)
    cuHalfComplex * __restrict__ vector_y,   // 输出复数向量 y (M×1)
    const int num_rows,                      // 矩阵 A 的行数
    const int num_cols) {                    // 矩阵 A 的列数（向量 x 的长度）
    // Block index
    int block_idx_x = blockIdx.x;

    // Thread index
    int thread_idx_x = threadIdx.x;
    int thread_idx_y = threadIdx.y;

    const int warp_size = 32;
    int lane_id = thread_idx_x % warp_size;
    int current_row = blockDim.y * block_idx_x + thread_idx_y;

    if(current_row < num_rows){
        cuHalfComplex partial_sum = cuHalfComplex(0, 0);
        int num_iterations = (num_cols / warp_size) / 4;
        if(num_iterations == 0) num_iterations = 1;
        matrix_a = &matrix_a[current_row * num_cols];

        #pragma unroll
        for(int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++){
            int current_col_vec = (iteration_idx * warp_size + lane_id) / 4;
            float4 current_val = reinterpret_cast<float4 *>(matrix_a)[current_col_vec];
            float4 current_x = reinterpret_cast<float4 *>(vector_x)[current_col_vec];
            cuHalfComplex val0 = reinterpret_cast<cuHalfComplex *>(&current_val)[0];
            cuHalfComplex val1 = reinterpret_cast<cuHalfComplex *>(&current_val)[1];
            cuHalfComplex val2 = reinterpret_cast<cuHalfComplex *>(&current_val)[2];
            cuHalfComplex val3 = reinterpret_cast<cuHalfComplex *>(&current_val)[3];
            cuHalfComplex x0 = reinterpret_cast<cuHalfComplex *>(&current_x)[0];
            cuHalfComplex x1 = reinterpret_cast<cuHalfComplex *>(&current_x)[1];
            cuHalfComplex x2 = reinterpret_cast<cuHalfComplex *>(&current_x)[2];
            cuHalfComplex x3 = reinterpret_cast<cuHalfComplex *>(&current_x)[3];

            partial_sum = partial_sum + val0 * x0;
            partial_sum = partial_sum + val1 * x1;
            partial_sum = partial_sum + val2 * x2;
            partial_sum = partial_sum + val3 * x3;
        }
        half result_real = partial_sum.r;
        half result_imag = partial_sum.i;
        result_real = warp_reduce_sum<warp_size>(result_real);
        result_imag = warp_reduce_sum<warp_size>(result_imag);
        if(lane_id == 0) vector_y[current_row] = cuHalfComplex(result_real, result_imag);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("usage: ./main [M] [N]\n");
        exit(0);
    }
    size_t num_rows = atoi(argv[1]);
    size_t num_cols = atoi(argv[2]);

    size_t bytes_matrix_a = sizeof(cuHalfComplex) * num_rows * num_cols;
    size_t bytes_vector_x = sizeof(cuHalfComplex) * num_cols;
    size_t bytes_vector_y = sizeof(cuHalfComplex) * num_rows;
    size_t bytes_vector_y_ref = sizeof(float2) * num_rows;
    cuHalfComplex* host_matrix_a = (cuHalfComplex*)malloc(bytes_matrix_a);
    cuHalfComplex* host_vector_x = (cuHalfComplex*)malloc(bytes_vector_x);
    cuHalfComplex* host_vector_y = (cuHalfComplex*)malloc(bytes_vector_y);
    float2* host_vector_y_ref = (float2*)malloc(bytes_vector_y_ref);

    cuHalfComplex* device_matrix_a;
    cuHalfComplex* device_vector_x;
    cuHalfComplex* device_vector_y;

    checkCudaErrors(cudaMalloc((void**)&device_matrix_a, bytes_matrix_a));
    checkCudaErrors(cudaMalloc((void**)&device_vector_x, bytes_vector_x));
    checkCudaErrors(cudaMalloc((void**)&device_vector_y, bytes_vector_y));

    // 生成A的数据
    for( int element_idx = 0; element_idx < num_rows * num_cols; element_idx++ ) {
        half real_val = 1;
        half imag_val = 1;
        host_matrix_a[element_idx] = cuHalfComplex(real_val, imag_val);
    }

    // 生成x的数据
    for( int element_idx = 0; element_idx < num_cols; element_idx++ ) {
        half real_val = 1;
        half imag_val = 1;
        host_vector_x[element_idx] = cuHalfComplex(real_val, imag_val);
    }
    memset(host_vector_y, 0, num_rows * sizeof(cuHalfComplex));
    memset(host_vector_y_ref, 0, num_rows * sizeof(float2));

    int num_iterations = 1000;
    checkCudaErrors(cudaMemcpy(device_matrix_a, host_matrix_a, bytes_matrix_a, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_vector_x, host_vector_x, bytes_vector_x, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_vector_y, host_vector_y, bytes_vector_y, cudaMemcpyHostToDevice));
    for (int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++ ) {
        dim3 grid_dim(num_rows / 4);
        dim3 block_dim(32, 4);
        ComplexHalfGemv<<< grid_dim, block_dim >>>(device_matrix_a, device_vector_x, device_vector_y, num_rows, num_cols);
    }
    checkCudaErrors(cudaMemcpy(host_vector_y, device_vector_y, bytes_vector_y, cudaMemcpyDeviceToHost));

    // compute the result in cpu
    // fp16 is not support in CPU, so use float
    for(int row_idx = 0; row_idx < num_rows; row_idx++){
        float result_real = 0;
        float result_imag = 0;
        for(int col_idx = 0; col_idx < num_cols; col_idx++){
            float a_real = host_matrix_a[row_idx * num_cols + col_idx].r;
            float a_imag = host_matrix_a[row_idx * num_cols + col_idx].i;
            float b_real = host_vector_x[col_idx].r;
            float b_imag = host_vector_x[col_idx].i;
            float res_real = a_real * b_real - a_imag * b_imag;
            float res_imag = a_imag * b_real + a_real * b_imag;
            result_real += res_real;
            result_imag += res_imag;
        }
        float2 result;
        result.x = result_real;
        result.y = result_imag;
        host_vector_y_ref[row_idx] = result;
    }

    // simple check, not reasonable
    double eps = 1.e-3;
    bool correct = true;
    for (int element_idx = 0; element_idx < num_rows; element_idx++) {
        double abs_err = fabs((float)(host_vector_y[element_idx].r) - host_vector_y_ref[element_idx].x) + 
                        fabs((float)(host_vector_y[element_idx].i) - host_vector_y_ref[element_idx].y);
        if (abs_err > eps) {
            printf("Error! Matrix[%05d]=(%.8f,%.8f), ref=(%.8f,%.8f) error term is > %E\n",
                    element_idx, (float)(host_vector_y[element_idx].r), (float)(host_vector_y[element_idx].i), 
                    (host_vector_y_ref[element_idx].x), (host_vector_y_ref[element_idx].y), eps);
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
