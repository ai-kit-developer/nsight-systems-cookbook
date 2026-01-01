# SGEMM 矩阵乘法优化

单精度矩阵乘法（SGEMM）是高性能计算中最核心的操作之一。本目录展示了如何通过系统性的优化方法，将 SGEMM 性能优化到接近 cuBLAS 的水平。

## 📊 性能指标

在 **NVIDIA V100** GPU 上测试，矩阵大小 M=N=K=4096：

- **最终性能**: 达到 cuBLAS 的 **96.8%**
- **峰值浮点效率**: **93.6%**
- **基本达到 CUDA C 代码优化的极限**

## 📁 文件说明

- `sgemm_v1.cu` - 基础优化版本，包含分块、共享内存、寄存器分片等优化
- `sgemm_v3.cu` - 最终优化版本，包含 warp 级优化和改进的访问模式
- `asm/` - SASS 汇编代码优化（使用 CuAssembler）

## 🔧 优化策略

### v1 版本的主要优化

1. **分块计算（Tiling）**
   - 将大矩阵分成小块，每个线程块计算 C 的一个子块
   - 减少全局内存访问次数

2. **共享内存缓存**
   - 将 A 和 B 的子块加载到共享内存
   - 多个线程重用同一块数据

3. **寄存器分片**
   - 每个线程计算 C 的一个小分片（如 8×8）
   - 使用寄存器存储中间结果，减少共享内存访问

4. **向量化加载**
   - 使用 `float4` 向量化加载
   - 一次加载 4 个 float，提高内存带宽利用率

5. **双缓冲（可选）**
   - 重叠计算和内存加载
   - 隐藏内存延迟

### v3 版本的额外优化

1. **优化的共享内存访问模式**
   - 使用 warp 级别的索引计算
   - 减少 bank conflict

2. **改进的寄存器分片布局**
   - 使用 `a_tile_index` 和 `b_tile_index` 优化访问
   - 更好的数据局部性

3. **更高效的结果存储**
   - 分块存储结果
   - 提高写入带宽

## 🎯 关键优化技巧

### 1. 模板参数设计

```cpp
template <
    const int BLOCK_SIZE_M,  // C 块的高度（如 128）
    const int BLOCK_SIZE_K,  // A 块的宽度（如 8）
    const int BLOCK_SIZE_N,  // C 块的宽度（如 128）
    const int THREAD_SIZE_Y, // 每个线程计算的高度（如 8）
    const int THREAD_SIZE_X, // 每个线程计算的宽度（如 8）
    const bool ENABLE_DOUBLE_BUFFER  // 是否启用双缓冲
>
```

### 2. 共享内存布局

- **A 的共享内存**: `[BLOCK_SIZE_M][BLOCK_SIZE_K]`
- **B 的共享内存**: `[BLOCK_SIZE_K][BLOCK_SIZE_N]`
- 注意避免 bank conflict

### 3. 寄存器分片

每个线程维护一个 `[THREAD_SIZE_Y][THREAD_SIZE_X]` 的寄存器数组，用于存储 C 的部分结果。

### 4. 向量化加载

```cpp
// 使用 float4 一次加载 4 个 float
float4 vec_a = FETCH_FLOAT4(A[offset_a]);
float4 vec_b = FETCH_FLOAT4(B[offset_b]);
```

## 📈 性能分析

### 使用 Nsight Systems 分析

```bash
# 编译
nvcc -o sgemm_v3 sgemm_v3.cu -lcublas

# 性能分析
nsys profile --trace=cuda,nvtx --output=sgemm_profile.nsys-rep ./sgemm_v3

# 查看结果
nsys-ui sgemm_profile.nsys-rep
```

### 关键指标

- **内存带宽利用率**: 应该接近理论峰值
- **计算效率**: 浮点运算效率
- **占用率**: SM 占用率
- **共享内存使用**: Bank conflict 情况

## 🔬 SASS 级别优化

在 `asm/` 目录中，包含了 SASS 汇编代码的优化：

1. **寄存器重映射**: 优化寄存器分配
2. **指令重排**: 获得更好的 `.reuse` 标志布局
3. **使用 CuAssembler**: 手动优化汇编代码

### 查看汇编代码

```bash
cd asm
# 查看原始汇编
cat sgemm_pre.sm_70.cuasm

# 查看优化后的汇编
cat sgemm_final.sm_70.cuasm
```

## 💡 优化建议

### 参数调优

1. **BLOCK_SIZE_M 和 BLOCK_SIZE_N**
   - 通常选择 128 或 256
   - 需要平衡共享内存使用和占用率

2. **BLOCK_SIZE_K**
   - 通常选择 8 或 16
   - 影响共享内存使用和计算/内存访问比

3. **THREAD_SIZE_Y 和 THREAD_SIZE_X**
   - 通常选择 8×8 或 4×8
   - 影响寄存器使用和指令级并行

### 常见问题

1. **Bank Conflict**
   - 检查共享内存访问模式
   - 使用 padding 或调整访问模式

2. **寄存器溢出**
   - 减少 THREAD_SIZE
   - 优化寄存器使用

3. **占用率低**
   - 增加 BLOCK_SIZE
   - 减少共享内存使用

## 📚 相关资源

- [NVIDIA cuBLAS](https://docs.nvidia.com/cuda/cublas/)
- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA 的高性能 GEMM 库
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Matrix Multiplication on GPU](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)

## 🎓 学习路径

1. **理解基础版本** (`sgemm_v1.cu`)
   - 理解分块、共享内存、寄存器分片的基本概念
   - 运行代码，观察性能

2. **分析优化版本** (`sgemm_v3.cu`)
   - 对比 v1 和 v3 的差异
   - 理解 warp 级优化的作用

3. **深入 SASS 优化** (`asm/`)
   - 学习如何查看和分析 SASS 代码
   - 理解寄存器重映射和指令重排

4. **参考 CUTLASS**
   - 学习工业级的实现
   - 理解更高级的优化技巧

---

**通过系统性的优化，将 SGEMM 性能推向极限！** 🚀
