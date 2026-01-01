# Elementwise 逐元素操作优化

逐元素操作（Elementwise）是最简单的 GPU 操作之一，但也是理解 GPU 内存访问优化的基础。本目录展示了如何通过向量化内存访问来优化逐元素操作。

## 📊 性能指标

在 **NVIDIA V100** GPU 上测试：

| 类型 | 带宽 (GB/s) | 利用率 |
|------|------------|--------|
| float | 827 | 91.9% |
| float2 | 838 | 93.1% |
| float4 | 844 | **93.8%** |

**结论**: 使用 `float4` 向量化访问可以获得最佳性能。

## 🎯 优化策略

### 核心思想

逐元素操作（如向量加法）的优化主要依赖于**向量化内存访问**。

### 为什么向量化有效？

1. **减少指令数**: 一次加载 4 个 float，而不是 4 次单独加载
2. **提高内存带宽利用率**: 更好地利用内存总线的宽度
3. **减少地址计算开销**: 减少地址计算和内存访问指令

## 🔧 实现对比

### 基础版本 (float)

```cpp
__global__ void elementwise_add_float(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**特点：**
- 每个线程处理一个元素
- 简单直接
- 性能：827 GB/s (91.9%)

### 优化版本 (float2)

```cpp
__global__ void elementwise_add_float2(float2 *a, float2 *b, float2 *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = make_float2(
            a[idx].x + b[idx].x,
            a[idx].y + b[idx].y
        );
    }
}
```

**特点：**
- 每个线程处理 2 个元素
- 使用 `float2` 向量类型
- 性能：838 GB/s (93.1%)

### 最佳版本 (float4)

```cpp
__global__ void elementwise_add_float4(float4 *a, float4 *b, float4 *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = make_float4(
            a[idx].x + b[idx].x,
            a[idx].y + b[idx].y,
            a[idx].z + b[idx].z,
            a[idx].w + b[idx].w
        );
    }
}
```

**特点：**
- 每个线程处理 4 个元素
- 使用 `float4` 向量类型
- 性能：844 GB/s (**93.8%**)

## 💡 关键技巧

### 1. 使用向量类型

CUDA 提供了内置的向量类型：
- `float2`: 2 个 float
- `float4`: 4 个 float
- `int2`, `int4`: 整数向量类型
- `double2`: 双精度向量类型

### 2. 内存对齐

向量类型需要内存对齐：
- `float2`: 8 字节对齐
- `float4`: 16 字节对齐

```cpp
// 使用 cudaMalloc 分配对齐的内存
float4 *d_a, *d_b, *d_c;
cudaMalloc(&d_a, n * sizeof(float4));
cudaMalloc(&d_b, n * sizeof(float4));
cudaMalloc(&d_c, n * sizeof(float4));
```

### 3. 处理边界情况

当数组大小不是 4 的倍数时，需要处理边界：

```cpp
__global__ void elementwise_add_float4_safe(float4 *a, float4 *b, float4 *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;
    
    // 处理完整的 float4
    if (vec_idx + 3 < n) {
        float4 va = a[idx];
        float4 vb = b[idx];
        c[idx] = make_float4(
            va.x + vb.x,
            va.y + vb.y,
            va.z + vb.z,
            va.w + vb.w
        );
    } else {
        // 处理剩余元素
        for (int i = 0; i < 4 && vec_idx + i < n; i++) {
            c[idx].x = a[idx].x + b[idx].x;  // 简化示例
        }
    }
}
```

## 📈 性能分析

### 使用 Nsight Systems 分析

```bash
# 编译
nvcc -o elementwise_add elementwise_add.cu

# 性能分析
nsys profile --trace=cuda,nvtx --output=elementwise_profile.nsys-rep ./elementwise_add

# 查看结果
nsys-ui elementwise_profile.nsys-rep
```

### 关键指标

- **内存带宽利用率**: 应该接近理论峰值（900 GB/s for V100）
- **内存事务数**: 向量化应该减少内存事务数
- **占用率**: 应该保持较高的 SM 占用率

## 🎓 学习要点

1. **向量化是基础优化**
   - 对于简单的逐元素操作，向量化是最有效的优化
   - 应该优先考虑向量化

2. **选择合适的向量大小**
   - `float4` 通常是最佳选择
   - 但需要考虑内存对齐和边界处理

3. **理解内存访问模式**
   - 向量化要求连续的内存访问
   - 确保数据布局支持向量化

4. **平衡复杂度和性能**
   - 向量化会增加代码复杂度
   - 需要权衡性能和可维护性

## 🔗 与其他算子的关系

- **Reduce**: 可以使用向量化加载数据，然后在 warp 内进行 reduce
- **SGEMM**: 使用 `float4` 加载矩阵块
- **SGEMV**: 可以使用向量化加载向量元素

## 📚 相关资源

- [CUDA Vector Types](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types)
- [Memory Coalescing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-coalescing)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## 💻 编译和运行

```bash
# 编译
nvcc -o elementwise_add elementwise_add.cu

# 运行
./elementwise_add
```

## 🚀 进一步优化

虽然向量化已经能获得很好的性能，但还可以考虑：

1. **使用共享内存**: 对于更复杂的操作，可以使用共享内存缓存数据
2. **循环展开**: 对于固定的循环次数，可以手动展开
3. **指令级优化**: 使用内联函数和编译器优化选项

---

**通过简单的向量化，就能获得接近理论峰值的性能！** 🚀
