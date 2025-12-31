#!/bin/bash
# 示例 1: 基本性能分析
# 最简单的 nsys profile 用法

echo "=========================================="
echo "示例 1: 基本性能分析"
echo "=========================================="
echo ""
echo "命令: nsys profile --output=output.nsys-rep python script.py"
echo ""
echo "说明:"
echo "  - 默认跟踪 CUDA API 调用"
echo "  - 生成 .nsys-rep 报告文件"
echo "  - 可以用 Nsight Systems GUI 打开查看"
echo ""

# 示例命令（注释掉，避免实际执行）
# nsys profile --output=nsys_profiles/basic_profile.nsys-rep python nvtx_tag_demo.py

echo "实际使用示例:"
echo "  nsys profile --output=nsys_profiles/basic_profile.nsys-rep python nvtx_tag_demo.py"
echo ""

