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
 * warp_reduce_sum: 使用 Shuffle 指令的 Warp 内归约函数
 * 优化：
 * 1. 使用 __shfl_down_sync 指令进行 warp 内数据交换
 * 2. 不需要共享内存，直接在寄存器间交换数据
 * 3. 延迟更低，带宽更高（寄存器访问比共享内存快）
 * 4. 使用 __forceinline__ 强制内联，减少函数调用开销
 * 5. 0xffffffff 是掩码，表示所有 32 个线程都参与
 * 
 * Shuffle 指令说明：
 * __shfl_down_sync(mask, var, delta) 从索引为 (lane_id + delta) 的线程获取 var 的值
 */
template <unsigned int warp_size>
__device__ __forceinline__ float warp_reduce_sum(float sum) {
    // 使用 Shuffle 指令进行 warp 内归约
    // 每个步骤将步长减半，直到所有值归约到 lane_id=0 的线程
    if (warp_size >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (warp_size >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if (warp_size >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);   // 0-4, 1-5, 2-6, etc.
    if (warp_size >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);   // 0-2, 1-3, 4-6, 5-7, etc.
    if (warp_size >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);   // 0-1, 2-3, 4-5, etc.
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
        // 计算每个线程需要处理的元素数量
        int num_iterations = num_cols / warp_size;
        if(num_iterations == 0) num_iterations = 1;  // 确保至少处理一个元素
        
        // 展开循环，每个线程处理 num_iterations 个元素
        #pragma unroll
        for(int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++){
            // 计算当前线程处理的列索引
            int current_col = iteration_idx * warp_size + lane_id;
            // 累加：matrix_a[current_row][current_col] * vector_x[current_col]
            partial_sum += matrix_a[current_row * num_cols + current_col] * vector_x[current_col];
        }
        
        // 使用 warp 内归约将 32 个线程的部分结果归约到 lane_id=0 的线程
        partial_sum = warp_reduce_sum<warp_size>(partial_sum);
        
        // 将最终结果写入输出向量
        if(lane_id == 0) vector_y[current_row] = partial_sum;
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
    size_t num_rows = atoi(argv[1]);  // 矩阵行数
    size_t num_cols = atoi(argv[2]);  // 矩阵列数

    // 计算所需内存大小
    size_t bytes_matrix_a = sizeof(float) * num_rows * num_cols;  // 矩阵 A 的内存大小
    size_t bytes_vector_x = sizeof(float) * num_cols;      // 向量 x 的内存大小
    size_t bytes_vector_y = sizeof(float) * num_rows;     // 向量 y 的内存大小
    
    // 分配主机内存
    float* host_matrix_a = (float*)malloc(bytes_matrix_a);    // 主机端矩阵 A
    float* host_vector_x = (float*)malloc(bytes_vector_x);   // 主机端向量 x
    float* host_vector_y = (float*)malloc(bytes_vector_y);   // 主机端向量 y（自定义实现的结果）
    float* host_vector_y_ref = (float*)malloc(bytes_vector_y);  // 主机端向量 y（cuBLAS 的结果，用于验证）

    // 设备内存指针
    float* device_matrix_a;
    float* device_vector_x;
    float* device_vector_y;

    // 分配设备内存
    checkCudaErrors(cudaMalloc(&device_matrix_a, bytes_matrix_a));
    checkCudaErrors(cudaMalloc(&device_vector_x, bytes_vector_x));
    checkCudaErrors(cudaMalloc(&device_vector_y, bytes_vector_y));

    // 初始化矩阵 A 的数据
    for( int element_idx = 0; element_idx < num_rows * num_cols; element_idx++ ) {
        host_matrix_a[element_idx] = (float)element_idx / num_cols;
    }

    // 初始化向量 x 的数据（全为 1）
    for( int element_idx = 0; element_idx < num_cols; element_idx++ ) {
        host_vector_x[element_idx] = 1;
    }
    // 初始化输出向量为 0
    memset(host_vector_y, 0, num_rows * sizeof(float));
    memset(host_vector_y_ref, 0, num_rows * sizeof(float));

    // 迭代次数，用于性能测试
    int num_iterations = 1000;
    
    // 将数据从主机内存复制到设备内存
    checkCudaErrors(cudaMemcpy(device_matrix_a, host_matrix_a, bytes_matrix_a, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_vector_x, host_vector_x, bytes_vector_x, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_vector_y, host_vector_y, bytes_vector_y, cudaMemcpyHostToDevice));
    
    // 执行自定义的 SGEMV 内核
    for (int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++ ) {
        // 配置网格和线程块维度
        // 每个线程块处理 4 行，每行由 32 个线程（一个 warp）处理
        dim3 grid_dim(num_rows / 4);      // 网格维度：需要 (num_rows/4) 个线程块
        dim3 block_dim(32, 4);    // 线程块维度：32×4 = 128 个线程
        Sgemv_v0<<< grid_dim, block_dim >>>(device_matrix_a, device_vector_x, device_vector_y, num_rows, num_cols);
    }
    // 将结果从设备内存复制回主机内存
    checkCudaErrors(cudaMemcpy(host_vector_y, device_vector_y, bytes_vector_y, cudaMemcpyDeviceToHost));

    // 使用 cuBLAS 计算参考结果
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;  // 标量乘数
    float beta = 0;     // 标量乘数（不使用累加）
    checkCudaErrors(cudaMemcpy(device_vector_y, host_vector_y_ref, bytes_vector_y, cudaMemcpyHostToDevice));
    // 执行 cuBLAS SGEMV（注意：使用转置操作 CUBLAS_OP_T）
    for (int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++ ) {
        cublasSgemv (blas_handle, CUBLAS_OP_T, 
            num_cols, num_rows, &alpha, 
            device_matrix_a, num_cols, device_vector_x, 1, &beta, device_vector_y, 1
        );
    }
    checkCudaErrors(cudaMemcpy(host_vector_y_ref, device_vector_y, bytes_vector_y, cudaMemcpyDeviceToHost));
    cublasDestroy(blas_handle); 
    
    // 验证结果：比较自定义实现和 cuBLAS 的结果
    double eps = 1.e-6;  // 机器精度阈值
    bool correct = true;
    for (int element_idx = 0; element_idx < num_rows; element_idx++) {
        double abs_err = fabs(host_vector_y[element_idx] - host_vector_y_ref[element_idx]);  // 绝对误差
        double dot_length = num_rows;
        double abs_val = fabs(host_vector_y[element_idx]);
        double rel_err = abs_err / abs_val / dot_length;  // 相对误差
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    element_idx, host_vector_y[element_idx], host_vector_y_ref[element_idx], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    
    // 释放内存
    cudaFree(device_matrix_a);
    cudaFree(device_vector_x);
    cudaFree(device_vector_y);
    
    free(host_matrix_a);
    free(host_vector_x);
    free(host_vector_y);
    free(host_vector_y_ref);
}
