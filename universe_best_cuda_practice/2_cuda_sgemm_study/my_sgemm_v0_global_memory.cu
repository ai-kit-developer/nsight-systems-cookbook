#include <cstdio>
// 矩阵访问宏定义：简化二维矩阵的索引计算
#define A(i, j) a[(i) * n + (j)]
#define B(i, j) b[(i) * n + (j)]

/**
 * 生成随机矩阵
 * 
 * @param m 矩阵行数
 * @param n 矩阵列数
 * @param a 矩阵数据指针
 */
void random_matrix(int m, int n, float *a)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
#if 1
            // 生成[-1, 1]范围内的随机浮点数
            A(i, j) = 2.0 * (float)drand48() - 1.0;
#else
            // 生成测试用的模式数据
            A(i, j) = (j - i) % 3;
#endif
}

/**
 * 比较两个矩阵，返回最大差值
 * 
 * @param m 矩阵行数
 * @param n 矩阵列数
 * @param a 第一个矩阵
 * @param b 第二个矩阵
 * @return 最大差值
 */
float compare_matrices(int m, int n, float *a, float *b)
{
    int i, j;
    float max_diff = 0.0, diff;
    int printed = 0;

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            diff = abs(A(i, j) - B(i, j));
            max_diff = (diff > max_diff ? diff : max_diff);
            // 只打印第一个超过阈值的错误
            if (0 == printed)
                if (max_diff > 0.5f || max_diff < -0.5f)
                {
                    printf("\n error: i %d  j %d diff %f  got %f  expect %f ", i, j, max_diff, A(i, j), B(i, j));
                    printed = 1;
                }
        }
    }
    return max_diff;
}

/**
 * CPU端实现的单精度矩阵乘法 C = A * B
 * 
 * @param A_ptr 矩阵A，大小为 M x K
 * @param B_ptr 矩阵B，大小为 K x N
 * @param C_ptr 结果矩阵C，大小为 M x N
 * @param M 矩阵A的行数
 * @param N 矩阵B的列数
 * @param K 矩阵A的列数（矩阵B的行数）
 */
void cpu_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float temp = 0.f;
            for (int k = 0; k < K; k++)
            {
                // C[m][n] = sum(A[m][k] * B[k][n])
                temp += A_ptr[m * K + k] * B_ptr[k * N + n];
            }
            C_ptr[m * N + n] = temp;
        }
    }
}

/**
 * CUDA SGEMM版本0：使用全局内存
 * 这是最基础的实现，直接从全局内存读取数据
 * 缺点：全局内存访问延迟高，没有数据重用，性能较差
 * 
 * @param A_ptr 矩阵A的全局内存指针，大小为 M x K
 * @param B_ptr 矩阵B的全局内存指针，大小为 K x N
 * @param C_ptr 结果矩阵C的全局内存指针，大小为 M x N
 * @param M 矩阵A的行数
 * @param N 矩阵B的列数
 * @param K 矩阵A的列数（矩阵B的行数）
 */
__global__ void cuda_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    // 计算当前线程负责的输出元素位置
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    // 计算当前block负责的A和B矩阵的起始位置
    float *A_ptr_start = A_ptr + blockDim.y * blockIdx.y * K;  // A矩阵的行起始位置
    float *B_ptr_start = B_ptr + blockDim.x * blockIdx.x;     // B矩阵的列起始位置
    
    // 计算点积：C[y][x] = sum(A[y][k] * B[k][x])
    float temp = 0.f;
    for (int k = 0; k < K; k++)
    {
        // 直接从全局内存读取A和B的元素并相乘累加
        temp += A_ptr_start[threadIdx.y * K + k] * B_ptr_start[k * N + threadIdx.x];
    }
    
    // 将结果写回全局内存
    C_ptr[x + y * N] = temp;
}

int main()
{
    int m = 512;
    int n = 512;
    int k = 512;
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);

    float *matrix_A_host = (float *)malloc(mem_size_A);
    float *matrix_B_host = (float *)malloc(mem_size_B);

    float *matrix_C_host_gpu_calc = (float *)malloc(mem_size_C);
    float *matrix_C_host_cpu_calc = (float *)malloc(mem_size_C);

    random_matrix(m, k, matrix_A_host);
    random_matrix(k, n, matrix_B_host);
    memset(matrix_C_host_gpu_calc, 0, mem_size_C);
    memset(matrix_C_host_cpu_calc, 0, mem_size_C);

    float *matrix_A_device, *matrix_B_device, *matrix_C_device;
    cudaMalloc((void **)&matrix_A_device, mem_size_A);
    cudaMalloc((void **)&matrix_B_device, mem_size_B);
    cudaMalloc((void **)&matrix_C_device, mem_size_C);

    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);

    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, m, n, k);

    constexpr int BLOCK = 8;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    cuda_sgemm<<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);

    float diff = compare_matrices(m, n, matrix_C_host_gpu_calc, matrix_C_host_cpu_calc);
    if (diff > 0.5f || diff < -0.5f)
    {
        printf("diff too big !\n");
        exit(-1);
    }
    else
    {
        printf("right\n");
    }

    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_cpu_calc);
    free(matrix_C_host_gpu_calc);

    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
    return 0;
}