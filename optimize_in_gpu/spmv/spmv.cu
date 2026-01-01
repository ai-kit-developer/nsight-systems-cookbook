#include <bits/stdc++.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime_api.h> 
#include <cusparse.h> 

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

void add_edge(int src_node, int dst_node, float weight,
    int *head, int *edge_dst, int *next_edge, float *edge_weight, int &edge_idx)
{
    edge_dst[edge_idx] = dst_node;
    edge_weight[edge_idx] = weight;
    next_edge[edge_idx] = head[src_node];
    head[src_node] = edge_idx++;
}

void read_vertex_edges(int &is_weighted, int &num_vertices, int &num_vertices_symmetric, int &num_edges, std::string &file)
{
    std::ifstream input;
    input.open("matrix/" + file + ".mtx");

    while (input.peek() == '%')
        input.ignore(2048, '\n');

    input >> num_vertices >> num_vertices_symmetric >> num_edges;

    std::string str;
    input.ignore();
    getline(input, str);
    int space_count = 0;
    for(auto char_val : str){
        if(char_val == ' '){
            space_count++;
        }
    }
    if(space_count == 1){
        is_weighted = 0;
    }
    else if(space_count == 2){
        is_weighted = 1;
    }
    else{
        std::cout<<"error! you need to get right mtx input\n";
        exit(0);
    }
    input.close();
}

void read_mtx_file(int is_weighted, int num_vertices, int num_edges,
            int *row_offset, int *col_index, float *val,
            std::string &file)
{
    ifstream input;
    input.open("matrix/" + file + ".mtx");

    while (input.peek() == '%')
        input.ignore(2048, '\n');

    int num_vertices_symmetric;
    input >> num_vertices >> num_vertices_symmetric >> num_edges;
    int *head = (int *)malloc((num_vertices + 10) * sizeof(int));
    memset(head, -1, sizeof(int) * (num_vertices + 10));
    int *edge_dst = (int *)malloc((num_edges + 10) * sizeof(int));
    int *next_edge = (int *)malloc((num_edges + 10) * sizeof(int));
    float *edge_weight = (float *)malloc((num_edges + 10) * sizeof(float));
    int edge_idx = 0;

    int src_node, dst_node;
    double weight;
    srand((int)time(0));
    if(is_weighted == 0){
        while (input >> src_node >> dst_node)
        {
            src_node--;
            dst_node--;
            weight = src_node % 13;
            float weight_float = static_cast<float>(weight);
            add_edge(src_node, dst_node, weight_float, head, edge_dst, next_edge, edge_weight, edge_idx);
        }
    }
    else if(is_weighted == 1){
        while (input >> src_node >> dst_node >> weight)
        {
            src_node--;
            dst_node--;
            float weight_float = static_cast<float>(weight);
            add_edge(src_node, dst_node, weight_float, head, edge_dst, next_edge, edge_weight, edge_idx);
        }
    }
    else{
        std::cout<<"error! you need to get right mtx input\n";
        exit(0);
    }
    

    row_offset[0] = 0;
    int nnz_num = 0;

    for (int row_idx = 0; row_idx < num_vertices; row_idx++)
    {
        int count = 0;
        for (int edge_ptr = head[row_idx]; edge_ptr != -1; edge_ptr = next_edge[edge_ptr])
        {
            count++;
            int next_node = edge_dst[edge_ptr];
            float next_weight = edge_weight[edge_ptr];
            col_index[nnz_num] = next_node;
            val[nnz_num] = next_weight;
            nnz_num++;
        }
        row_offset[row_idx + 1] = row_offset[row_idx] + count;
    }

    input.close();
    free(head);
    free(edge_dst);
    free(next_edge);
    free(edge_weight);
}

/**
 * warp_reduce_sum: 使用 Shuffle 指令的 Warp 内归约函数
 * 优化：
 * 1. 使用 __shfl_down_sync 指令进行 warp 内数据交换
 * 2. 不需要共享内存，直接在寄存器间交换数据
 * 3. 延迟更低，带宽更高（寄存器访问比共享内存快）
 * 4. 使用 __forceinline__ 强制内联，减少函数调用开销
 */
template <unsigned int warp_size>
__device__ __forceinline__ float warp_reduce_sum(float sum) {
    if (warp_size >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (warp_size >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if (warp_size >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);   // 0-4, 1-5, 2-6, etc.
    if (warp_size >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);   // 0-2, 1-3, 4-6, 5-7, etc.
    if (warp_size >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);   // 0-1, 2-3, 4-5, etc.
    return sum;
}

/**
 * My_spmv_csr_kernel: 稀疏矩阵向量乘法（SpMV）内核
 * 计算 y = A * x，其中 A 是稀疏矩阵（CSR 格式），x 是密集向量，y 是输出向量
 * 
 * 实现策略：
 * 1. 每个线程组（vector）处理矩阵的一行
 * 2. 线程组内的线程并行处理该行的非零元素
 * 3. 使用 warp 内归约得到该行的最终结果
 * 
 * 优化点：
 * - 使用模板参数在编译时确定线程组大小
 * - 根据矩阵稀疏度动态选择线程组大小
 * - 使用 Shuffle 指令进行高效归约
 * 
 * @tparam IndexType 索引类型（通常是 int）
 * @tparam ValueType 值类型（通常是 float）
 * @tparam VECTORS_PER_BLOCK 每个线程块中的线程组数量
 * @tparam THREADS_PER_VECTOR 每个线程组中的线程数（通常是 2, 4, 8, 16, 32）
 */
template <typename IndexType, typename ValueType, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__global__ void My_spmv_csr_kernel(const IndexType num_rows,        // 矩阵行数
                       const IndexType * row_offset,              // CSR 格式：行偏移数组
                       const IndexType * col_index,               // CSR 格式：列索引数组
                       const ValueType * value,                   // CSR 格式：非零值数组
                       const ValueType * vector_x,                         // 输入向量 x
                       ValueType * vector_y)                                // 输出向量 y
{
    // 计算线程块中的线程总数
    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    // 全局线程索引
    const IndexType thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    // 线程在向量组内的索引（0 到 THREADS_PER_VECTOR-1）
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);
    // 当前线程组处理的行索引
    const IndexType row_id   = thread_id   /  THREADS_PER_VECTOR;

    // 边界检查：确保行索引在有效范围内
    if(row_id < num_rows){
        // 获取当前行的非零元素范围
        const IndexType row_start = row_offset[row_id];      // 当前行第一个非零元素的索引
        const IndexType row_end   = row_offset[row_id + 1];   // 下一行第一个非零元素的索引

        // 初始化局部累加器
        ValueType partial_sum = 0;

        // 并行累加：线程组内的线程并行处理该行的非零元素
        // 每个线程处理间隔为 THREADS_PER_VECTOR 的元素
        for(IndexType nnz_idx = row_start + thread_lane; nnz_idx < row_end; nnz_idx += THREADS_PER_VECTOR)
            partial_sum += value[nnz_idx] * vector_x[col_index[nnz_idx]];

        // 使用 warp 内归约将线程组内的部分结果归约
        partial_sum = warp_reduce_sum<THREADS_PER_VECTOR>(partial_sum);
        
        // 由线程组内的第一个线程（thread_lane == 0）写入最终结果
        if (thread_lane == 0){
            vector_y[row_id] = partial_sum;
        }   
    }
}

template<typename T>
void vec_print(vector<T> array){
    for(auto x: array){
        cout<<x<<" ";
    }
    cout<<std::endl;
}

template <typename IndexType, typename ValueType>
void spmv_cpu_kernel(vector<IndexType> &row_offset,
                vector<IndexType> &col_index,
                vector<ValueType> &value,
                vector<ValueType> &vector_x,
                vector<ValueType> &vector_y,
                IndexType num_rows)
{
    for(int row_idx = 0; row_idx < num_rows; row_idx++){
        ValueType partial_sum = 0;
        IndexType num_nnz = row_offset[row_idx + 1] - row_offset[row_idx];
        for(int nnz_idx = 0; nnz_idx < num_nnz; nnz_idx++){
            IndexType nnz_index = row_offset[row_idx] + nnz_idx;
            partial_sum += value[nnz_index] * vector_x[col_index[nnz_index]];
        }
        vector_y[row_idx] = partial_sum;
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("usage: ./spmv -f [matrix]\n");
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

    // read mtx file and convert to csr
    int is_weighted = -1;
    int num_rows;
    int num_cols;
    int num_nnz;
    read_vertex_edges(is_weighted, num_rows, num_cols, num_nnz, file);
    vector<int> row_offset(num_rows + 1);
    vector<int> col_index(num_nnz);
    vector<float> value(num_nnz);
    vector<float> vector_x(num_cols, 1.0);
    vector<float> vector_y(num_rows);
    vector<float> vector_y_res(num_rows);
    vector<float> vector_y_cusparse_res(num_rows);
    int num_iterations = 2000;
    read_mtx_file(is_weighted, num_rows, num_nnz, row_offset.data(), col_index.data(), value.data(), file);

    // check input
    // std::cout<<" The row_offset is: "<<std::endl;
    // vec_print<int>(row_offset);
    // std::cout<<" The col_index is: "<<std::endl;
    // vec_print<int>(col_index);
    // std::cout<<" The value is: "<<std::endl;
    // vec_print<float>(value);

    // allocate memory in GPU device
    int* device_row_offset;
    int* device_col_index;
    float* device_value;
    float* device_vector_x;
    float* device_vector_y;
    float* device_vector_y_cusparse;

    checkCudaErrors(cudaMalloc(&device_row_offset, (num_rows + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&device_col_index, num_nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc(&device_value, num_nnz * sizeof(float)));
    checkCudaErrors(cudaMalloc(&device_vector_x, num_cols * sizeof(float)));
    checkCudaErrors(cudaMalloc(&device_vector_y, num_rows * sizeof(float)));
    checkCudaErrors(cudaMalloc(&device_vector_y_cusparse, num_rows * sizeof(float)));
    checkCudaErrors(cudaMemcpy(device_row_offset, row_offset.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_col_index, col_index.data(), num_nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_value, value.data(), num_nnz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_vector_x, vector_x.data(), num_cols * sizeof(float), cudaMemcpyHostToDevice));
    
    // spmv
    // 32 thread for a row
    int mean_col_num = (num_nnz + (num_rows - 1)) / num_rows;
    std::cout<< "The average col num is: "<< mean_col_num << std::endl;

    // const int THREADS_PER_VECTOR = 32;
    // const unsigned int VECTORS_PER_BLOCK  = 256 / THREADS_PER_VECTOR;
    // const unsigned int THREADS_PER_BLOCK  = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    // const unsigned int NUM_BLOCKS = static_cast<unsigned int>((row_num + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
    // My_spmv_csr_kernel<int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> 
    //     (row_num, d_A_row_offset, d_A_col_index, d_A_value, d_x, d_y);
    
    for(int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++){
        if(mean_col_num <= 2){
            const int THREADS_PER_VECTOR = 2;
            const unsigned int VECTORS_PER_BLOCK  = 128;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((num_rows + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            My_spmv_csr_kernel<int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, 256>>> 
                (num_rows, device_row_offset, device_col_index, device_value, device_vector_x, device_vector_y);
        }
        else if(mean_col_num > 2 && mean_col_num <= 4){
            const int THREADS_PER_VECTOR = 4;
            const unsigned int VECTORS_PER_BLOCK  = 64;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((num_rows + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            My_spmv_csr_kernel<int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, 256>>> 
                (num_rows, device_row_offset, device_col_index, device_value, device_vector_x, device_vector_y);
        }
        else if(mean_col_num > 4 && mean_col_num <= 8){
            const int THREADS_PER_VECTOR = 8;
            const unsigned int VECTORS_PER_BLOCK  = 32;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((num_rows + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            My_spmv_csr_kernel<int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, 256>>> 
                (num_rows, device_row_offset, device_col_index, device_value, device_vector_x, device_vector_y);
        }
        else if(mean_col_num > 8 && mean_col_num <= 16){
            const int THREADS_PER_VECTOR = 16;
            const unsigned int VECTORS_PER_BLOCK  = 16;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((num_rows + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            My_spmv_csr_kernel<int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, 256>>> 
                (num_rows, device_row_offset, device_col_index, device_value, device_vector_x, device_vector_y);
        }
        else if(mean_col_num > 16){
            const int THREADS_PER_VECTOR = 32;
            const unsigned int VECTORS_PER_BLOCK  = 8;
            const unsigned int NUM_BLOCKS = static_cast<unsigned int>((num_rows + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
            My_spmv_csr_kernel<int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, 256>>> 
                (num_rows, device_row_offset, device_col_index, device_value, device_vector_x, device_vector_y);
        }
    }
    checkCudaErrors(cudaMemcpy(vector_y.data(), device_vector_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

    // cusparse spmv
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    float     alpha           = 1.0f;
    float     beta            = 0.0f;

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, num_rows, num_cols, num_nnz,
                                      device_row_offset, device_col_index, device_value,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, num_cols, device_vector_x, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, num_rows, device_vector_y_cusparse, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMV
    for(int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++){
        CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
    }

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(vector_y_cusparse_res.data(), device_vector_y_cusparse, num_rows * sizeof(float),
                           cudaMemcpyDeviceToHost) )

    bool check_result = true;
    for(int row_idx = 0; row_idx < num_rows; row_idx++){
        if(fabs(vector_y[row_idx] - vector_y_cusparse_res[row_idx]) > 1e-3){
            std::cout<<"The result is error!"<<std::endl;
            printf("The row is: %d the y is:%f and the cusparse_y is:%f\n", row_idx, vector_y[row_idx], vector_y_cusparse_res[row_idx]);
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
    cudaFree(device_vector_x);
    cudaFree(device_vector_y);

    return 0;
}
