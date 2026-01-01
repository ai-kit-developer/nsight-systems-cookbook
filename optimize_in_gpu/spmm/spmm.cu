#include <bits/stdc++.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime_api.h> 
#include <cuda_runtime.h>
#include <cusparse.h> 
#ifdef SPUTNIK_AVAILABLE
#include "sputnik/sputnik.h"
#endif

using namespace std;

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void read_file(std::string &file, int &num_rows, int &num_cols, int &num_nnz,
            std::vector<int> &row_offset, std::vector<int> &col_index,
            std::vector<float> &value, std::vector<float> &matrix_b,
            std::vector<float> &matrix_c)
{
    std::ifstream input;
    input.open("matrix/" + file + ".smtx");

    while (input.peek() == '%')
        input.ignore(2048, '\n');
    
    std::string line_str;
    getline(input, line_str);
    // parse the first line
    std::stringstream s_stream(line_str);
    std::string current_str;
    getline(s_stream, current_str, ',');
    num_rows = atoi(current_str.c_str());
    getline(s_stream, current_str, ',');
    num_cols = atoi(current_str.c_str());
    getline(s_stream, current_str, ',');
    num_nnz = atoi(current_str.c_str());

    row_offset.resize(num_rows + 1);
    col_index.resize(num_nnz);
    value.resize(num_nnz);
    for(int element_idx = 0; element_idx < num_rows + 1; element_idx++){
        input >> row_offset[element_idx];
    }
    for(int element_idx = 0; element_idx < num_nnz; element_idx++){
        input >> col_index[element_idx];
    }
    input.close();
    matrix_b.resize(num_cols * num_rows);
    matrix_c.resize(num_rows * num_rows);

    // init A
    for(int element_idx = 0; element_idx < value.size(); element_idx++){
        value[element_idx] = element_idx % 17;
    }

    // init B
    for(int element_idx = 0; element_idx < matrix_b.size(); element_idx++){
        matrix_b[element_idx] = element_idx % 13;
    }
}

template<typename T>
void vec_print(std::vector<T> array){
    for(auto x: array){
        cout<<x<<" ";
    }
    cout<<std::endl;
}

template <typename IndexType, typename ValueType>
void spmm_cpu_kernel(std::vector<IndexType> &row_offset,
                std::vector<IndexType> &col_index,
                std::vector<ValueType> &value,
                std::vector<ValueType> &matrix_b,
                std::vector<ValueType> &matrix_c,
                IndexType num_rows,
                IndexType num_cols)
{
    for(int row_idx = 0; row_idx < num_rows; row_idx++){
        for(int col_idx = 0; col_idx < num_rows; col_idx++){
            ValueType partial_sum = 0;
            IndexType num_nnz = row_offset[row_idx + 1] - row_offset[row_idx];
            for(int nnz_idx = 0; nnz_idx < num_nnz; nnz_idx++){
                IndexType nnz_index = row_offset[row_idx] + nnz_idx;
                IndexType current_col = col_index[nnz_index];
                partial_sum += value[nnz_index] * matrix_b[current_col * num_rows + col_idx];
            }
            matrix_c[row_idx * num_rows + col_idx] = partial_sum;
        }
    }
}

/**
 * My_spmm_csr_vector_kernel_v0: 稀疏矩阵-密集矩阵乘法（SpMM）内核 - 基础版本
 * 计算 C = A * B，其中 A 是稀疏矩阵（CSR 格式），B 和 C 是密集矩阵
 * 
 * 实现策略：
 * 1. 每个线程计算输出矩阵 C 的一个元素
 * 2. 线程通过 (blockIdx.y, blockIdx.x * THREAD_NUM_PER_BLOCK + threadIdx.x) 定位到 C 的元素
 * 3. 对于 C 的每一行，遍历 A 的对应行的非零元素
 * 4. 累加 A 的非零值乘以 B 的对应元素
 * 
 * 矩阵维度：
 * - A: (num_rows, col_num) - 稀疏矩阵，CSR 格式
 * - B: (col_num, ldb) - 密集矩阵
 * - C: (num_rows, ldc) - 输出密集矩阵
 * 
 * @tparam THREAD_NUM_PER_BLOCK 每个线程块中的线程数
 */
template <unsigned int THREAD_NUM_PER_BLOCK>
__global__ void My_spmm_csr_vector_kernel_v0(const int num_rows,        // 矩阵 A 的行数
    const int * A_row_offset,              // CSR 格式：行偏移数组
    const int * A_col_index,               // CSR 格式：列索引数组
    const float * A_value,                 // CSR 格式：非零值数组
    const float * B,                       // 输入密集矩阵 B
    float * C,                             // 输出密集矩阵 C
    const int ldb,                         // 矩阵 B 的行宽（leading dimension）
    const int ldc){                        // 矩阵 C 的行宽（leading dimension）
    // 线程块索引
    int block_idx_x = blockIdx.x;  // x 方向的线程块索引
    int block_idx_y = blockIdx.y;  // y 方向的线程块索引

    // 线程在块内的索引
    int thread_idx_x = threadIdx.x;

    // 计算当前线程处理的输出矩阵 C 的元素位置
    int c_row_idx = block_idx_y;  // C 的行索引
    int c_col_idx = block_idx_x * THREAD_NUM_PER_BLOCK + thread_idx_x;  // C 的列索引

    // 边界检查：确保索引在有效范围内
    if(c_row_idx < num_rows && c_col_idx < ldc){
        // 获取矩阵 A 当前行的非零元素范围
        int row_start = A_row_offset[c_row_idx];      // 当前行第一个非零元素的索引
        int row_end = A_row_offset[c_row_idx + 1];   // 下一行第一个非零元素的索引
        int num_nnz = row_end - row_start;            // 当前行的非零元素数量
        
        // 初始化累加器
        float partial_sum = 0.0;
        
        // 遍历当前行的所有非零元素
        for(int nnz_idx = 0; nnz_idx < num_nnz; nnz_idx++){
            int nnz_index = row_start + nnz_idx;                    // 非零元素在 CSR 数组中的索引
            int current_col = A_col_index[nnz_index];         // 非零元素在 A 中的列索引
            float current_val = A_value[nnz_index];           // 非零元素的值
            // 从矩阵 B 中加载对应元素：B[current_col][c_col_idx]
            float reg_b = B[current_col * ldb + c_col_idx];
            // 累加：A[c_row_idx][current_col] * B[current_col][c_col_idx]
            partial_sum += current_val * reg_b;
        }

        // 将结果写入输出矩阵 C
        C[c_row_idx * ldc + c_col_idx] = partial_sum;
    }
}

// dim3 dimBlock(THREAD_NUM_PER_BLOCK);
// dim3 dimGrid(row_num/THREAD_NUM_PER_BLOCK, row_num);
// useless optimize
template <
    const int BLOCK_SIZE_X,   
    const int BLOCK_SIZE_K,
    const int THREAD_NUM_PER_BLOCK
    > 
__global__ void My_spmm_csr_vector_kernel_v1(const int num_rows,
    const int * A_row_offset,
    const int * A_col_index,
    const float * A_value,
    const float * B,
    float * C,
    const int M,
    const int N,
    const int K){
    // Block index
    int block_idx_x = blockIdx.x;
    int block_idx_y = blockIdx.y;

    // Thread index
    int thread_idx_x = threadIdx.x;

    // matrix C row_index
    int c_row_idx = block_idx_y;
    int c_col_idx = block_idx_x * THREAD_NUM_PER_BLOCK + thread_idx_x;

    // shared mem for A 
    __shared__ int shared_col[BLOCK_SIZE_K];
    __shared__ float shared_value[BLOCK_SIZE_K];

    int num_a_per_thread = BLOCK_SIZE_K / THREAD_NUM_PER_BLOCK;

    if(c_row_idx < num_rows && c_col_idx < N){
        int row_start = A_row_offset[c_row_idx];
        int row_end = A_row_offset[c_row_idx + 1];
        int num_nnz = row_end - row_start;
        float partial_sum = 0.0;

        for(int k_offset = 0; k_offset < num_nnz; k_offset += BLOCK_SIZE_K){
            // store A to shared mem
            int global_index = row_start + k_offset * BLOCK_SIZE_K;
            int local_index = num_a_per_thread * thread_idx_x;
            for(int element_idx = 0; element_idx < num_a_per_thread; element_idx++){
                if(global_index + local_index + element_idx < row_end){
                    shared_col[local_index + element_idx] = A_col_index[global_index + local_index + element_idx];
                    shared_value[local_index + element_idx] = A_value[global_index + local_index + element_idx];
                }
                else{
                    shared_col[local_index + element_idx] = -1;
                    shared_value[local_index + element_idx] = 0.0;
                }
            }
            __syncthreads();
            // load A from shared mem
            for(int element_idx = 0; element_idx < BLOCK_SIZE_K; element_idx++){
                int current_col = shared_col[element_idx];
                float current_val = shared_value[element_idx];
                if(current_col != -1){
                    float reg_b = B[current_col * N + c_col_idx];
                    partial_sum += current_val * reg_b;
                }
            }
        }

        C[c_row_idx * N + c_col_idx] = partial_sum;
    }
}


// A(row_num,col_num)
// B(col_num,row_num)
// C(row_num,row_num)
int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("usage: ./spmm -f [matrix]\n");
        exit(0);
    }
    string file;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-f") == 0)
        {
            file = argv[i + 1];
        }
    }

    // load csr data from .smtx file
    int num_rows = 0;
    int num_cols = 0;
    int num_nnz = 0;
    std::vector<int> row_offset;
    std::vector<int> col_index;
    std::vector<float> value;
    std::vector<float> matrix_b;
    std::vector<float> matrix_c;
    read_file(file, num_rows, num_cols, num_nnz, row_offset, col_index, value, matrix_b, matrix_c);
    std::vector<float> matrix_c_cusparse(matrix_c.size());

    // used in sputnik
    // TODO: it's useless?
    std::vector<int> row_indices(num_rows);
    // init row_indices
    for(int row_idx = 0; row_idx < num_rows; row_idx++){
        row_indices[row_idx] = row_offset[row_idx + 1] - row_offset[row_idx];
    }

    //debug case
    /*
    int row_num = 4;
    int col_num = 4;
    int nnz_num = 9;
    int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    float hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              6.0f, 7.0f, 8.0f, 9.0f };
    float hB[]            = { 1.0f, 5.0f, 9.0f, 0.0f,
                              2.0f, 6.0f, 10.0f, 0.0f,
                              3.0f, 7.0f, 11.0f, 0.0f,
                              4.0f, 8.0f, 12.0f, 0.0f};
    std::vector<int> A_row_offset(hA_csrOffsets, hA_csrOffsets + sizeof(hA_csrOffsets));
    std::vector<int> A_col_index(hA_columns, hA_columns + sizeof(hA_columns));
    std::vector<float> A_value(hA_values, hA_values + sizeof(hA_values));
    std::vector<float> B(hB, hB + sizeof(hB));
    std::vector<float> C(16, 0);
    std::vector<float> C_cusparse(16, 0);
    */

    // check input
    std::cout<<"The num_rows is:" <<num_rows <<std::endl;
    std::cout<<"The num_cols is:" <<num_cols <<std::endl;
    std::cout<<"The num_nnz is:" <<num_nnz <<std::endl;

    // allocate memory in GPU device
    int* device_row_offset;
    int* device_col_index;
    float* device_value;
    float* device_matrix_b;
    float* device_matrix_c;
    float* device_matrix_c_cusparse;
    int* device_row_indices;
    int matrix_b_size = matrix_b.size();
    int matrix_c_size = matrix_c.size();

    checkCudaErrors(cudaMalloc(&device_row_offset, (num_rows + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&device_col_index, num_nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc(&device_value, num_nnz * sizeof(float)));
    checkCudaErrors(cudaMalloc(&device_matrix_b, matrix_b_size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&device_matrix_c, matrix_c_size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&device_matrix_c_cusparse, matrix_c_size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&device_row_indices, num_rows * sizeof(int)));
    checkCudaErrors(cudaMemcpy(device_row_offset, row_offset.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_col_index, col_index.data(), num_nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_value, value.data(), num_nnz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_matrix_b, matrix_b.data(), matrix_b_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_row_indices, row_indices.data(), num_rows * sizeof(int), cudaMemcpyHostToDevice));
    
    int num_iterations = 2000;
    // My spmm
    // cpu version
    // spmm_cpu_kernel<int,float>(row_offset, col_index, value, matrix_b, matrix_c, num_rows, num_cols);

    constexpr unsigned int THREAD_NUM_PER_BLOCK  = 128;
    
    dim3 block_dim(THREAD_NUM_PER_BLOCK);
    dim3 grid_dim(num_rows / THREAD_NUM_PER_BLOCK, num_rows);

    for(int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++){
        My_spmm_csr_vector_kernel_v1<128, 512, THREAD_NUM_PER_BLOCK> <<< grid_dim, block_dim >>> 
            (num_rows, device_row_offset, device_col_index, device_value, device_matrix_b, device_matrix_c, num_rows, num_rows, num_cols);
    }
    //checkCudaErrors(cudaMemcpy(matrix_c.data(), device_matrix_c, matrix_c_size*sizeof(float), cudaMemcpyDeviceToHost));

    // sputnik (optional, requires sputnik library)
    #ifdef SPUTNIK_AVAILABLE
    cudaStream_t stream = 0;
    for(int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++){
        sputnik::CudaSpmm(num_rows, num_rows, num_cols, 
                            num_nnz, device_row_indices, 
                            device_value, device_row_offset, device_col_index, 
                            device_matrix_b, device_matrix_c, stream);
    }
    cudaStreamSynchronize(stream);
    checkCudaErrors(cudaMemcpy(matrix_c.data(), device_matrix_c, matrix_c_size * sizeof(float), cudaMemcpyDeviceToHost));
    #else
    // Skip sputnik test if library is not available
    std::cout << "Note: Sputnik library not available, skipping sputnik test." << std::endl;
    #endif
    

    // cusparse spmm
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    int ldb = num_rows;
    int ldc = num_rows;
    float alpha           = 1.0f;
    float beta            = 0.0f;
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                device_buffer    = NULL;
    size_t               buffer_size = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, num_rows, num_cols, num_nnz,
                                      device_row_offset, device_col_index, device_value,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, num_cols, num_rows, ldb, device_matrix_b,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, num_rows, num_rows, ldc, device_matrix_c_cusparse,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size) )
    CHECK_CUDA( cudaMalloc(&device_buffer, buffer_size) )

    // execute SpMM
    for(int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++){
        CHECK_CUSPARSE( cusparseSpMM(handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                CUSPARSE_SPMM_ALG_DEFAULT, device_buffer) )
    }
    
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(matrix_c_cusparse.data(), device_matrix_c_cusparse, matrix_c_size * sizeof(float),
                           cudaMemcpyDeviceToHost) )

    bool check_result = true;
    for(int element_idx = 0; element_idx < matrix_c.size(); element_idx++){
        if(fabs(matrix_c[element_idx] - matrix_c_cusparse[element_idx]) > 1e-6){
            std::cout<<"The result is error!"<<std::endl;
            printf("The error case is (%d %d %f %f)\n", element_idx / num_rows, element_idx % num_rows, matrix_c[element_idx], matrix_c_cusparse[element_idx]);
            check_result = false;
            break;
        }
    }
    if(check_result){
        std::cout<<"The result is right!"<<std::endl;
    }

    // Free Memory
    cudaFree(device_row_offset);
    cudaFree(device_col_index);
    cudaFree(device_value);
    cudaFree(device_matrix_b);
    cudaFree(device_matrix_c);
    cudaFree(device_matrix_c_cusparse);

    return 0;
}