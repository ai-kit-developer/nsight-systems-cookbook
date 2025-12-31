#!/bin/bash
# 示例 2: 不同的跟踪选项
# 展示 --trace 参数的不同选项

echo "=========================================="
echo "示例 2: 不同的跟踪选项"
echo "=========================================="
echo ""

echo "2.1 跟踪 CUDA API 调用（默认）"
echo "命令: nsys profile --trace=cuda --output=output.nsys-rep python script.py"
echo "说明: 跟踪所有 CUDA API 调用"
echo ""

echo "2.2 跟踪 CUDA + NVTX 标记"
echo "命令: nsys profile --trace=cuda,nvtx --output=output.nsys-rep python script.py"
echo "说明: 同时跟踪 CUDA API 和 NVTX 标记（推荐用于有 NVTX 标记的代码）"
echo ""

echo "2.3 跟踪 CUDA + NVTX + OS Runtime"
echo "命令: nsys profile --trace=cuda,nvtx,osrt --output=output.nsys-rep python script.py"
echo "说明: 跟踪 CUDA、NVTX 和操作系统运行时（更全面的分析）"
echo ""

echo "2.4 跟踪所有可用选项"
echo "命令: nsys profile --trace=cuda,nvtx,osrt,opengl --output=output.nsys-rep python script.py"
echo "说明: 跟踪多个组件（注意：某些选项可能不适用于所有应用）"
echo ""

echo "实际使用示例:"
echo "  nsys profile --trace=cuda,nvtx --output=nsys_profiles/trace_cuda_nvtx.nsys-rep python nvtx_tag_demo.py"
echo ""

