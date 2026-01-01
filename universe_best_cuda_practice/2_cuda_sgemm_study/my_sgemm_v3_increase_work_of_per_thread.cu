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
 * CUDA SGEMM版本3：增加每个线程的工作量
 * 相比v2版本，每个线程计算多个输出元素（STRIDE x STRIDE个）
 * 优点：减少block数量，提高计算密度，更好地利用寄存器
 * 
 * @tparam BLOCK_SIZE block的大小（通常为16）
 * @tparam STRIDE 每个线程在M和N维度上处理的元素数量（通常为2）
 * @param A_ptr 矩阵A的全局内存指针
 * @param B_ptr 矩阵B的全局内存指针
 * @param C_ptr 结果矩阵C的全局内存指针
 * @param M 矩阵A的行数
 * @param N 矩阵B的列数
 * @param K 矩阵A的列数（矩阵B的行数）
 */
template <unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void cuda_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    // 计算每个block处理的元素数量
    constexpr int STEP = BLOCK_SIZE * STRIDE;
    
    // 线程在block内的索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 计算当前block负责的A和B矩阵的起始位置
    float *A_ptr_start = A_ptr + STEP * blockIdx.y * K;
    float *B_ptr_start = B_ptr + STEP * blockIdx.x;

    // 声明共享内存：每个block处理STEP x STEP大小的块
    __shared__ float a_shared[STEP][STEP];
    __shared__ float b_shared[STEP][STEP];
    
    // 每个线程计算STRIDE x STRIDE个输出元素，使用寄存器数组存储
    float temp[STRIDE][STRIDE] = {0.f};

    // 滑动窗口：分块处理K维度
    for (int s = 0; s < K; s += STEP)
    {
        // 第一步：协作加载数据到共享内存
        // 每个线程加载STRIDE x STRIDE个元素
        for (int i = 0; i < STRIDE; i++)
        {
            for (int j = 0; j < STRIDE; j++)
            {
                // 加载A矩阵的块
                a_shared[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = A_ptr_start[(ty + BLOCK_SIZE * i) * K + tx + BLOCK_SIZE * j + s];
                // 加载B矩阵的块
                b_shared[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = B_ptr_start[(ty + BLOCK_SIZE * i + s) * N + tx + BLOCK_SIZE * j];
            }
        }
        __syncthreads();
        
        // 第二步：在共享内存上进行矩阵乘法计算
        // 每个线程计算STRIDE x STRIDE个输出元素的部分和
        for (int i = 0; i < STRIDE; i++)
        {
            for (int j = 0; j < STRIDE; j++)
            {
                for (int k = 0; k < STEP; k++)
                {
                    temp[i][j] += a_shared[ty + i * BLOCK_SIZE][k] * b_shared[k][tx + j * BLOCK_SIZE];
                }
            }
        }
        __syncthreads();
    }

    // 第三步：将结果写回全局内存
    float *C_ptr_start = C_ptr + N * blockIdx.y * STEP + blockIdx.x * STEP;
    for (int i = 0; i < STRIDE; i++)
    {
        for (int j = 0; j < STRIDE; j++)
        {
            C_ptr_start[N * (ty + i * BLOCK_SIZE) + tx + j * BLOCK_SIZE] = temp[i][j];
        }
    }
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
    constexpr int STRIDE = 2;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);

    cuda_sgemm<BLOCK, STRIDE><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

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