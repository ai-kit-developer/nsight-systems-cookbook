# CUDA 最佳实践学习指南

本目录包含系统性的 CUDA 编程学习路径，从基础算子优化到高级 Tensor Core 使用，帮助开发者深入理解 GPU 编程和优化技巧。

## 📚 目录结构

```
universe_best_cuda_practice/
├── 1_cuda_reduce_study/          # Reduce 算子优化研究（10个版本）
├── 2_cuda_sgemm_study/           # SGEMM 矩阵乘法优化（8个版本）
├── 3_kernel_profiling_guide/      # Kernel 性能分析和优化
├── 4_tensor_core_wmma/           # Tensor Core WMMA API 使用
├── 5_mma_and_swizzle/            # MMA 指令和内存 Swizzle 优化
├── 6_cutlass_study/              # CUTLASS 高性能库学习
└── flash_attention/              # Flash Attention 实现
```

## 🎯 学习路径

### 1. Reduce 算子优化研究 (`1_cuda_reduce_study/`)

**学习目标：** 深入理解 Reduce 算子的优化技巧，从基础到高级

**版本演进：**
- `v0_global_memory` - 使用全局内存的基础版本
- `v1_shared_memory` - 引入共享内存
- `v2_no_divergence_branch` - 消除 warp divergence
- `v3_no_bank_conflict` - 消除 bank 冲突
- `v4_add_during_load` - 加载时进行计算（两个方案）
- `v5_unroll_last_warp` - 展开最后一个 warp
- `v6_completely_unroll` - 完全展开循环
- `v7_mutli_add` - 多元素累加
- `v8_shuffle` - 使用 shuffle 指令

**关键优化技巧：**
- 共享内存的使用和 bank 冲突避免
- Warp divergence 的消除
- 线程利用率的提升
- Shuffle 指令的使用

**编译和运行：**
```bash
cd 1_cuda_reduce_study
mkdir build && cd build
cmake ..
make
./my_reduce_v0_global_memory
```

### 2. SGEMM 矩阵乘法优化 (`2_cuda_sgemm_study/`)

**学习目标：** 掌握矩阵乘法的系统优化方法

**版本演进：**
- `v0_global_memory` - 全局内存版本
- `v1_shared_memory` - 共享内存分块
- `v2_shared_memory_sliding_windows` - 滑动窗口优化
- `v3_increase_work_of_per_thread` - 增加每线程工作量
- `v4_using_float4` - Float4 向量化
- `v5_register_outer_product` - 寄存器外积
- `v6_register_outer_product_float4` - 寄存器外积 + Float4
- `v7_A_smem_transpose` - A 矩阵转置优化
- `v8_double_buffer` - 双缓冲技术

**关键优化技巧：**
- Tiling 和共享内存使用
- 向量化内存访问（float4）
- 寄存器级优化
- 双缓冲流水线

**编译和运行：**
```bash
cd 2_cuda_sgemm_study
mkdir build && cd build
cmake ..
make
./my_sgemm_v0_global_memory
```

### 3. Kernel 性能分析指南 (`3_kernel_profiling_guide/`)

**学习目标：** 学习如何分析和优化 Kernel 性能

**主要内容：**
- `my_transpose_v*.cu` - Transpose 算子的多个优化版本
- `roofline_model.cu` - Roofline 模型分析
- `combined_access.cu` - 合并访问模式

**优化版本：**
- `v1_naive` - 朴素实现
- `v2_float4` - Float4 向量化
- `v3_float2` - Float2 向量化
- `v4_float2_1x2` - Float2 优化布局
- `v5_shared_memory` - 共享内存版本
- `v6_no_bank_conflict` - 消除 bank 冲突
- `v7_increase_work_of_per_thread` - 增加每线程工作量

**关键概念：**
- Roofline 模型：理解计算和内存带宽的限制
- 内存访问模式优化
- 性能瓶颈识别

### 4. Tensor Core WMMA (`4_tensor_core_wmma/`)

**学习目标：** 学习如何使用 Tensor Core 进行混合精度计算

**版本演进：**
- `hgemm_v1_wmma_m16n16k16_naive_kernel` - 基础 WMMA 使用
- `hgemm_v2_wmma_m16n16k16_mma4x2_kernel` - 优化版本
- `hgemm_v3_wmma_m16n16k16_mma4x2_warp2x4_kernel` - 多 warp 优化
- `hgemm_v4_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel` - 异步双缓冲

**关键特性：**
- Half precision (FP16) 矩阵乘法
- WMMA API 使用
- 多 warp 协作
- 异步内存操作

**适用架构：** Volta (V100), Turing (T4), Ampere (A100) 及以上

### 5. MMA 和 Swizzle (`5_mma_and_swizzle/`)

**学习目标：** 学习高级的 MMA 指令和内存访问优化

**版本演进：**
- `v1_simple_wmma` - 简单 WMMA
- `v2_shared_memory_wmma` - 共享内存 WMMA
- `v3_shared_memory_wmma_padding` - Padding 优化
- `v4_shared_memory_mma` - MMA 指令使用
- `v5_shared_memory_mma_swizzle` - Swizzle 内存访问优化

**关键优化：**
- MMA (Matrix Multiply-Accumulate) 指令
- Shared memory swizzle 模式
- 内存访问模式优化

### 6. CUTLASS 学习 (`6_cutlass_study/`)

**学习目标：** 学习使用 NVIDIA CUTLASS 库实现高性能 GEMM

**内容：**
- `v1_print_half.cu` - Half 精度数据类型
- `v2_gemm_kernel.cu` - 基础 GEMM kernel
- `v3_turing_tensorop_gemm.cu` - Turing 架构 Tensor Core GEMM

**CUTLASS 特性：**
- 模块化的 GEMM 实现
- 支持多种数据类型和精度
- 针对不同 GPU 架构的优化

### 7. Flash Attention (`flash_attention/`)

**学习目标：** 学习 Flash Attention 的高效实现

**特性：**
- 使用共享内存避免 O(N²) 内存访问
- 约 100 行 CUDA 代码实现前向传播
- 相比标准实现有显著加速

**性能对比：**
- 标准 Attention: ~52ms
- Flash Attention: ~4ms (约 13x 加速)

**编译和运行：**
```bash
cd flash_attention
mkdir build && cd build
cmake ..
make
python bench.py
```

## 🛠️ 编译说明

### 使用 CMake

所有子目录都支持 CMake 编译：

```bash
# 在项目根目录
mkdir build && cd build
cmake ..
make

# 或编译特定模块
cd 1_cuda_reduce_study
mkdir build && cd build
cmake ..
make
```

### 编译选项

可以在 `CMakeLists.txt` 中调整：
- CUDA 架构（sm_70, sm_75, sm_80 等）
- 优化级别（-O2, -O3）
- 调试选项

## 📊 性能测试

所有代码都包含性能测试，建议使用 Nsight Systems 进行详细分析：

```bash
# 使用 nsys 分析
nsys profile --trace=cuda,nvtx --output=profile.nsys-rep ./your_kernel

# 查看结果
nsys-ui profile.nsys-rep
```

## 🎓 学习建议

### 初学者

1. 从 `1_cuda_reduce_study` 开始，理解基础的优化技巧
2. 学习 `3_kernel_profiling_guide`，掌握性能分析方法
3. 实践 `2_cuda_sgemm_study`，学习更复杂的优化

### 进阶

1. 学习 `4_tensor_core_wmma`，掌握 Tensor Core 使用
2. 深入 `5_mma_and_swizzle`，学习高级优化技巧
3. 研究 `6_cutlass_study`，了解工业级实现

### 高级

1. 实现 `flash_attention`，理解复杂算法的优化
2. 结合性能分析工具，优化自己的代码
3. 阅读 CUTLASS 源码，学习最佳实践

## 📖 相关资源

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)

## 🔗 与其他模块的关系

- **optimize_in_gpu/**: 本目录提供了更系统、更深入的学习路径
- **gpu_profile/**: 使用性能分析工具验证优化效果

## 📝 注意事项

1. **GPU 架构兼容性**：不同代码针对不同 GPU 架构，请根据你的 GPU 调整编译选项
2. **性能数据**：所有性能数据仅供参考，实际性能取决于硬件和配置
3. **学习顺序**：建议按照编号顺序学习，每个模块都建立在前一个的基础上

---

**开始你的 CUDA 深度学习之旅！** 🚀

