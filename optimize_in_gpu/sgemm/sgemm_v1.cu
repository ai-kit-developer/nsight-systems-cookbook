// optimize sgemm

#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

/**
 * Sgemm: 优化的单精度矩阵乘法（SGEMM）内核
 * 计算 C = A * B，其中 A 是 M×K 矩阵，B 是 K×N 矩阵，C 是 M×N 矩阵
 * 
 * 优化策略：
 * 1. 分块计算：将矩阵分成块，每个线程块计算 C 的一个子块
 * 2. 共享内存缓存：将 A 和 B 的子块加载到共享内存，减少全局内存访问
 * 3. 寄存器分片：每个线程计算 C 的一个小分片，使用寄存器存储中间结果
 * 4. 向量化加载：使用 float4 向量化加载，提高内存带宽利用率
 * 5. 双缓冲（可选）：重叠计算和内存加载，隐藏内存延迟
 * 
 * 模板参数说明：
 * @tparam BLOCK_SIZE_M 每个线程块计算的 C 块的高度
 * @tparam BLOCK_SIZE_K 每个线程块加载到共享内存的 A 块的宽度（也是 B 块的高度）
 * @tparam BLOCK_SIZE_N 每个线程块计算的 C 块的宽度
 * @tparam THREAD_SIZE_Y 每个线程计算的 C 分片的高度
 * @tparam THREAD_SIZE_X 每个线程计算的 C 分片的宽度
 * @tparam ENABLE_DOUBLE_BUFFER 是否启用双缓冲优化
 * 
 * 矩阵布局：
 * - K: 矩阵 A 的行宽（leading dimension of A）
 * - N: 矩阵 B 的行宽（leading dimension of B）
 */
template <
    const int BLOCK_SIZE_M,  // 每个线程块计算的 C 块的高度
    const int BLOCK_SIZE_K,  // 每个线程块加载到共享内存的 A 块的宽度
    const int BLOCK_SIZE_N,  // 每个线程块计算的 C 块的宽度
    const int THREAD_SIZE_Y, // 每个线程计算的 C 分片的高度
    const int THREAD_SIZE_X,  // 每个线程计算的 C 分片的宽度
    const bool ENABLE_DOUBLE_BUFFER // 是否启用双缓冲优化
    > 
__global__ void Sgemm( 
    float * __restrict__ A,  // 输入矩阵 A (M×K)
    float * __restrict__ B,  // 输入矩阵 B (K×N)
    float * __restrict__ C,  // 输出矩阵 C (M×N)
    const int M,             // 矩阵 A 的行数（也是 C 的行数）
    const int N,             // 矩阵 B 的列数（也是 C 的列数）
    const int K) {           // 矩阵 A 的列数（也是 B 的行数）
    // Block index
    int block_idx_x = blockIdx.x;
    int block_idx_y = blockIdx.y;

    // Thread index
    int thread_idx_x = threadIdx.x;
    int thread_idx_y = threadIdx.y;
    
    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // thread id in cur Block
    const int thread_idx = thread_idx_y * THREAD_X_PER_BLOCK + thread_idx_x;

    // shared memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];
    // registers load global memory
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4*ldg_num_a];
    float ldg_b_reg[4*ldg_num_b];

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = thread_idx / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = thread_idx / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = thread_idx % A_TILE_THREAD_PER_ROW * 4; 
    const int B_TILE_COL = thread_idx % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    A = &A[(BLOCK_SIZE_M * block_idx_y) * K];
    B = &B[BLOCK_SIZE_N * block_idx_x];

    //transfer first tile from global mem to shared mem
    // load A from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i, // row
            A_TILE_COL, // col
            K )]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
        As[0][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
    }
    // load B from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i, // row
                B_TILE_COL, // col
                N )]);
    }
    __syncthreads();
    // load A from shared memory to register
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * thread_idx_y + thread_y]);
    }
    // load B from shared memory to register
    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * thread_idx_x + thread_x]);
    }

    int write_stage_idx = 1;
    int tile_idx = 0;
    do{
        tile_idx += BLOCK_SIZE_K;
        // load next tile from global mem
        if(tile_idx< K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i, // row
                    A_TILE_COL + tile_idx, // col
                    K )]);
            }
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL, // col
                    N )]);
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for(int j=0; j<BLOCK_SIZE_K-1; ++j){
            // load next tile from shared mem to register 
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                FETCH_FLOAT4(frag_a[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][THREAD_SIZE_Y * thread_idx_y + thread_y]);
            }
            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][THREAD_SIZE_X * thread_idx_x + thread_x]);
            }
            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        if(tile_idx < K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
            }
            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            // use double buffer, only need one sync
            __syncthreads();
            // switch
            write_stage_idx ^= 1;
        }

        // load first tile from shared mem to register of next iter
        // load A from shared memory to register
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y * thread_idx_y + thread_y]);
        }
        // load B from shared memory to register
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X * thread_idx_x + thread_x]);
        }
        //compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    }while(tile_idx< K);

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * block_idx_y + thread_idx_y * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * block_idx_x + thread_idx_x * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    assert( M%8 == 0); 
    assert( N%8 == 0); 
    assert( K%8 == 0); 

    size_t bytes_matrix_a = sizeof(float) * M * K;
    size_t bytes_matrix_b = sizeof(float) * K * N;
    size_t bytes_matrix_c = sizeof(float) * M * N;
    float* host_matrix_a = (float*)malloc(bytes_matrix_a);
    float* host_matrix_b = (float*)malloc(bytes_matrix_b);
    float* host_matrix_c = (float*)malloc(bytes_matrix_c);
    float* host_matrix_c_ref = (float*)malloc(bytes_matrix_c);

    float* device_matrix_a;
    float* device_matrix_b;
    float* device_matrix_c;

    checkCudaErrors(cudaMalloc(&device_matrix_a, bytes_matrix_a));
    checkCudaErrors(cudaMalloc(&device_matrix_b, bytes_matrix_b));
    checkCudaErrors(cudaMalloc(&device_matrix_c, bytes_matrix_c));
    double msec_per_matrix_mul[2] = {0, 0};
    double giga_flops[2] = {0, 0};
    double flops_per_matrix_mul = 2.0 * M * N * K;

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    // generate A
    for( int element_idx = 0; element_idx < M * K; element_idx++ ){
        host_matrix_a[element_idx] = element_idx / 13;
    }

    // generate B
    for( int element_idx = 0; element_idx < K * N; element_idx++ ) {
        host_matrix_b[element_idx] = element_idx % 13;
    }

    checkCudaErrors(cudaMemcpy(device_matrix_a, host_matrix_a, bytes_matrix_a, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_matrix_b, host_matrix_b, bytes_matrix_b, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msec_total = 0;
    int num_iterations = 1000;

    checkCudaErrors(cudaMemcpy(device_matrix_c, host_matrix_c, bytes_matrix_c, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++ ) {
        dim3 block_dim(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 grid_dim(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        Sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> 
        <<< grid_dim, block_dim >>>(device_matrix_a, device_matrix_b, device_matrix_c, M, N, K);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec_total, start, stop));


    checkCudaErrors(cudaMemcpy(host_matrix_c, device_matrix_c, bytes_matrix_c, cudaMemcpyDeviceToHost));

    msec_per_matrix_mul[0] = msec_total / num_iterations;
    giga_flops[0] = (flops_per_matrix_mul * 1.0e-9f) / (msec_per_matrix_mul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        giga_flops[0],
        msec_per_matrix_mul[0],
        flops_per_matrix_mul);

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy(device_matrix_c, host_matrix_c, bytes_matrix_c, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++ ) {
        cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, 
            device_matrix_a, K, device_matrix_b, N, &beta, device_matrix_c, N
        );
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec_total, start, stop));

    checkCudaErrors(cudaMemcpy(host_matrix_c_ref, device_matrix_c, bytes_matrix_c, cudaMemcpyDeviceToHost));

    msec_per_matrix_mul[1] = msec_total / num_iterations;
    giga_flops[1] = (flops_per_matrix_mul * 1.0e-9f) / (msec_per_matrix_mul[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        giga_flops[1],
        msec_per_matrix_mul[1],
        flops_per_matrix_mul);

    cublasDestroy(blas_handle); 
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int element_idx = 0; element_idx < M * N; element_idx++) {
        int row = element_idx / N;
        int col = element_idx % N;
        double abs_err = fabs(host_matrix_c[element_idx] - host_matrix_c_ref[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(host_matrix_c[element_idx]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    element_idx, host_matrix_c[element_idx], host_matrix_c_ref[col * M + row], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", giga_flops[0] / giga_flops[1]);
    
    // Free Memory
    cudaFree(device_matrix_a);
    cudaFree(device_matrix_b);
    cudaFree(device_matrix_c);
    
    free(host_matrix_a);
    free(host_matrix_b);
    free(host_matrix_c);
    free(host_matrix_c_ref);
}
