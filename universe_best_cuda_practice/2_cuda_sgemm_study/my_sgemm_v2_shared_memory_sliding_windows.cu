#include <cstdio>
#define A(i, j) a[(i) * n + (j)]
#define B(i, j) b[(i) * n + (j)]

void random_matrix(int m, int n, float *a)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
#if 1
            A(i, j) = 2.0 * (float)drand48() - 1.0;
#else
            A(i, j) = (j - i) % 3;
#endif
}

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

void cpu_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float temp = 0.f;
            for (int k = 0; k < K; k++)
            {
                temp += A_ptr[m * K + k] * B_ptr[k * N + n];
            }
            C_ptr[m * N + n] = temp;
        }
    }
}
/**
 * CUDA SGEMM版本2：使用共享内存和滑动窗口
 * 相比v1版本，使用正确的共享内存大小，并采用滑动窗口方式分块计算
 * 优点：正确使用共享内存，提高数据重用，减少全局内存访问
 * 
 * @tparam BLOCK_SIZE block的大小（通常为16）
 * @param A_ptr 矩阵A的全局内存指针
 * @param B_ptr 矩阵B的全局内存指针
 * @param C_ptr 结果矩阵C的全局内存指针
 * @param M 矩阵A的行数
 * @param N 矩阵B的列数
 * @param K 矩阵A的列数（矩阵B的行数）
 */
template <unsigned int BLOCK_SIZE>
__global__ void cuda_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    // 计算当前线程负责的输出元素位置
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    // 计算当前block负责的A和B矩阵的起始位置
    float *A_ptr_start = A_ptr + blockDim.y * blockIdx.y * K;
    float *B_ptr_start = B_ptr + blockDim.x * blockIdx.x;

    // 声明共享内存：用于缓存A和B矩阵的块（大小与block匹配）
    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    // 累加变量：存储当前线程负责的输出元素的部分和
    float temp = 0.f;

    // 滑动窗口：分块处理K维度
    // 每次迭代处理BLOCK_SIZE个K维度元素
    for (int s = 0; s < K; s += blockDim.x)
    {
        // 第一步：将A和B的当前块从全局内存加载到共享内存
        a_shared[threadIdx.y][threadIdx.x] = A_ptr_start[threadIdx.y * K + threadIdx.x + s];
        b_shared[threadIdx.y][threadIdx.x] = B_ptr_start[threadIdx.x + (threadIdx.y + s) * N];
        
        // 同步所有线程，确保数据加载完成
        __syncthreads();
        
        // 第二步：在共享内存上进行矩阵乘法计算
        // 计算当前块对结果的贡献
        for (int k = 0; k < BLOCK_SIZE; k++)
            temp += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        
        // 同步所有线程，准备加载下一块数据
        __syncthreads();
    }

    // 将最终结果写回全局内存
    C_ptr[x + y * N] = temp;
}

int main()
{
    int m = 1024;
    int n = 1024;
    int k = 1024;
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

    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

    cuda_sgemm<BLOCK><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

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