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

// 宏定义：使用float4向量化加载，一次加载4个float（16字节）
// 这样可以提高内存访问效率，减少内存事务数量
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

/**
 * CUDA SGEMM版本4：使用float4向量化加载
 * 相比v3版本，使用float4一次加载4个元素，提高内存访问效率
 * 优点：减少内存事务数量，提高内存带宽利用率
 * 
 * @tparam M_NUM_PER_BLOCK 每个block在M维度处理的元素数量
 * @tparam N_NUM_PER_BLOCK 每个block在N维度处理的元素数量
 * @tparam K_NUM_PER_BLOCK 每个block在K维度处理的元素数量
 * @tparam NUM_PER_THREAD 每个线程处理的元素数量（必须是4的倍数）
 * @param A_ptr 矩阵A的全局内存指针
 * @param B_ptr 矩阵B的全局内存指针
 * @param C_ptr 结果矩阵C的全局内存指针
 * @param M 矩阵A的行数
 * @param N 矩阵B的列数
 * @param K 矩阵A的列数（矩阵B的行数）
 */
template <unsigned int M_NUM_PER_BLOCK,
          unsigned int N_NUM_PER_BLOCK,
          unsigned int K_NUM_PER_BLOCK,
          unsigned int NUM_PER_THREAD>
__global__ void cuda_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 计算当前block负责的A和B矩阵的起始位置
    float *A_ptr_start = A_ptr + blockIdx.y * M_NUM_PER_BLOCK * K;
    float *B_ptr_start = B_ptr + blockIdx.x * N_NUM_PER_BLOCK;

    // 声明共享内存
    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    
    // 每个线程计算NUM_PER_THREAD个输出元素的部分和
    float temp[NUM_PER_THREAD] = {0.f};

    // 滑动窗口：分块处理K维度
    for (int s = 0; s < K; s += K_NUM_PER_BLOCK)
    {
        // 使用float4向量化加载：一次加载4个元素
        // 这比逐个加载效率更高，减少内存事务数量
        FETCH_FLOAT4(a_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A_ptr_start[K * ty + s + tx * NUM_PER_THREAD]);
        // 注释掉的代码是原来的逐个加载方式：
        // a_shared[ty][tx * NUM_PER_THREAD] = A_ptr_start[K * ty + s + tx * NUM_PER_THREA];
        // a_shared[ty][tx * NUM_PER_THREAD + 1] = A_ptr_start[K * ty + s + tx * NUM_PER_THREAD + 1];
        // a_shared[ty][tx * NUM_PER_THREAD + 2] = A_ptr_start[K * ty + s + tx * NUM_PER_THREAD + 2];
        // a_shared[ty][tx * NUM_PER_THREAD + 3] = A_ptr_start[K * ty + s + tx * NUM_PER_THREAD + 3];
        
        FETCH_FLOAT4(b_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B_ptr_start[N * (ty + s) + tx * NUM_PER_THREAD]);
        // 注释掉的代码是原来的逐个加载方式：
        // b_shared[ty][tx * NUM_PER_THREAD] = B_ptr_start[N * ty + tx * NUM_PER_THREAD];
        // b_shared[ty][tx * NUM_PER_THREAD + 1] = B_ptr_start[N * (ty + s) + tx * NUM_PER_THREAD + 1];
        // b_shared[ty][tx * NUM_PER_THREAD + 2] = B_ptr_start[N * (ty + s) + tx * NUM_PER_THREAD + 2];
        // b_shared[ty][tx * NUM_PER_THREAD + 3] = B_ptr_start[N * (ty + s) + tx * NUM_PER_THREAD + 3];
        
        __syncthreads();

        // 在共享内存上进行矩阵乘法计算
        for (int i = 0; i < NUM_PER_THREAD; i++)
        {
            for (int k = 0; k < K_NUM_PER_BLOCK; k++)
            {
                temp[i] += a_shared[ty][k] *
                           b_shared[k][tx * NUM_PER_THREAD + i];
            }
        }

        __syncthreads();
    }

    // 将结果写回全局内存
    float *C_ptr_start = C_ptr + N * blockIdx.y * M_NUM_PER_BLOCK +
                         blockIdx.x * N_NUM_PER_BLOCK;
    for (int i = 0; i < NUM_PER_THREAD; i++)
    {
        C_ptr_start[ty * N + tx * NUM_PER_THREAD + i] = temp[i];
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

    constexpr int M_NUM_PER_BLOCK = 32;
    constexpr int N_NUM_PER_BLOCK = 32;
    constexpr int K_NUM_PER_BLOCK = 32;
    constexpr int NUM_PER_THREAD = 4;

    dim3 block(8, 32);
    dim3 grid(m / M_NUM_PER_BLOCK, n / N_NUM_PER_BLOCK);

    cuda_sgemm<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

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