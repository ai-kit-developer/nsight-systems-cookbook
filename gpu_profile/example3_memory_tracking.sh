#!/bin/bash
# 示例 3: CUDA 内存跟踪
# 展示如何跟踪 GPU 内存使用情况

echo "=========================================="
echo "示例 3: CUDA 内存跟踪"
echo "=========================================="
echo ""

echo "3.1 启用 CUDA 内存使用跟踪"
echo "命令: nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --output=output.nsys-rep python script.py"
echo "说明: 跟踪 CUDA 内存分配和释放"
echo ""

echo "3.2 内存跟踪 + 采样频率"
echo "命令: nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --sampling-frequency=1000 --output=output.nsys-rep python script.py"
echo "说明: 设置采样频率为 1000Hz（更详细的内存使用信息）"
echo ""

echo "实际使用示例:"
echo "  nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --output=nsys_profiles/memory_tracking.nsys-rep python nvtx_tag_demo.py"
echo ""

