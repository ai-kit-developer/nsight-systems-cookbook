# SGEMV 矩阵向量乘法优化

单精度矩阵向量乘法（SGEMV）计算 `y = A * x`，其中 A 是 M×N 矩阵，x 是 N 维向量，y 是 M 维向量。本目录展示了针对不同数据形状的优化策略。

## 📊 性能指标

在 **NVIDIA V100** GPU 上测试：

| 版本 | M | N | 我的实现 (ns) | cuBLAS (ns) | 性能比 |
|------|---|---|--------------|-------------|--------|
| v0 | 16384 | 32 | 10341 | 8386 | 81.1% |
| v1 | 16384 | 128 | 14284 | 15848 | **110.9%** |
| v2 | 16384 | 16 | 6903 | 7576 | **109.7%** |

## 📁 文件说明

- `Sgemv_v0.cu` - 基础版本，针对 n=32 的情况
- `Sgemv_v1.cu` - 优化版本，针对 n>32 的情况
- `Sgemv_v2.cu` - 优化版本，针对 n<32 的情况
- `ComplexHalfGemv.cu` - 复数半精度版本
- `cuHalfComplex.cuh` - 复数半精度工具头文件

## 🎯 核心优化思想

SGEMV 优化的核心在于**合理设计 block 和 thread 的配置**，**避免线程空闲**。

### 问题分析

SGEMV 的计算模式：
- 每个输出元素 `y[i]` 需要计算 `A[i, :] * x` 的点积
- 这是一个典型的 reduce 操作
- 关键是如何组织线程来高效地完成这个 reduce

### 优化策略

根据向量 x 的长度（N）不同，采用不同的优化策略：

#### 1. N = 32 (v0)

- 每个 warp（32 个线程）处理一行
- 每个线程处理一个元素
- 使用 warp shuffle 进行 reduce

#### 2. N > 32 (v1)

- 每个 block 处理多行
- 使用共享内存存储中间结果
- 多个 warp 协作完成 reduce

#### 3. N < 32 (v2)

- 每个线程处理多行
- 增加每个线程的工作量
- 减少 block 数量，提高占用率

## 🔧 实现细节

### v0: N = 32 的情况

```cpp
// 每个 warp 处理一行
// 每个线程处理一个元素
// 使用 warp shuffle 进行 reduce
__global__ void Sgemv_v0(float *A, float *x, float *y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        // 每个线程计算一个元素
        sum = A[row * N + threadIdx.y] * x[threadIdx.y];
        // 使用 warp shuffle 进行 reduce
        sum = warpReduceSum<32>(sum);
        if (threadIdx.y == 0) {
            y[row] = sum;
        }
    }
}
```

**特点：**
- 简单直接
- 适合 N 正好等于 warp size 的情况
- 性能：81.1% of cuBLAS

### v1: N > 32 的情况

**优化点：**
- 使用共享内存存储中间结果
- 多个 warp 协作完成 reduce
- 更好的负载均衡

**性能：110.9% of cuBLAS** ✅

### v2: N < 32 的情况

**优化点：**
- 每个线程处理多行
- 增加每个线程的工作量
- 减少线程空闲

**性能：109.7% of cuBLAS** ✅

## 💡 关键优化技巧

### 1. Warp Shuffle 指令

```cpp
template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (WarpSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (WarpSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (WarpSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (WarpSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}
```

**优势：**
- 不需要共享内存
- 延迟更低
- 带宽更高（寄存器访问比共享内存快）

### 2. 向量化加载

对于较大的 N，可以使用 `float4` 向量化加载：

```cpp
float4 vec_a = FETCH_FLOAT4(A[row * N + col]);
float4 vec_x = FETCH_FLOAT4(x[col]);
```

### 3. 共享内存使用

对于 N > 32 的情况，使用共享内存存储中间结果：

```cpp
__shared__ float sdata[BLOCK_SIZE];
// 每个 warp 将结果写入共享内存
if (lane_id == 0) {
    sdata[warp_id] = sum;
}
__syncthreads();
// 第一个 warp 进行最终的 reduce
```

## 📈 性能分析

### 使用 Nsight Systems 分析

```bash
# 编译
nvcc -o sgemv_v0 Sgemv_v0.cu -lcublas

# 性能分析
nsys profile --trace=cuda,nvtx --output=sgemv_profile.nsys-rep ./sgemv_v0

# 查看结果
nsys-ui sgemv_profile.nsys-rep
```

### 关键指标

- **内存带宽利用率**: 检查全局内存访问效率
- **占用率**: SM 占用率
- **Warp 效率**: Warp 内线程的利用率
- **共享内存使用**: Bank conflict 情况

## 🎓 学习要点

1. **理解数据形状对性能的影响**
   - 不同的 N 值需要不同的优化策略
   - 没有一种通用的优化方法适用于所有情况

2. **合理设计 block 和 thread**
   - 避免线程空闲
   - 平衡占用率和资源使用

3. **灵活使用 warp shuffle**
   - 对于小规模的 reduce，warp shuffle 比共享内存更高效
   - 减少共享内存使用，提高占用率

4. **针对特定场景优化**
   - 根据实际应用场景选择最合适的版本
   - 有时可以针对特定数据形状进行特殊优化

## 🔗 与其他算子的关系

- **Reduce**: SGEMV 本质上是一个 per-row 的 reduce 操作
- **SGEMM**: 可以理解为多个 SGEMV 的组合
- **Elementwise**: 某些优化技巧（如向量化）可以借鉴

## 📚 相关资源

- [NVIDIA cuBLAS SGEMV](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv)
- [CUDA Warp Shuffle](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
- [Matrix-Vector Multiplication Optimization](https://developer.nvidia.com/blog/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)

## 💻 编译和运行

```bash
# 编译所有版本
nvcc -o sgemv_v0 Sgemv_v0.cu -lcublas
nvcc -o sgemv_v1 Sgemv_v1.cu -lcublas
nvcc -o sgemv_v2 Sgemv_v2.cu -lcublas

# 运行
./sgemv_v0
./sgemv_v1
./sgemv_v2
```

---

**通过针对不同数据形状的优化，实现超越 cuBLAS 的性能！** 🚀
