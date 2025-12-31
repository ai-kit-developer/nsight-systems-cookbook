#!/bin/bash
# 示例 7: 统计信息输出
# 展示如何获取统计信息

echo "=========================================="
echo "示例 7: 统计信息输出"
echo "=========================================="
echo ""

echo "7.1 显示统计信息"
echo "命令: nsys profile --stats=true --output=output.nsys-rep python script.py"
echo "说明: 在终端显示性能统计信息摘要"
echo ""

echo "7.2 统计信息 + 详细跟踪"
echo "命令: nsys profile --stats=true --trace=cuda,nvtx --output=output.nsys-rep python script.py"
echo "说明: 同时显示统计信息和生成详细报告"
echo ""

echo "实际使用示例:"
echo "  nsys profile --stats=true --trace=cuda,nvtx --output=nsys_profiles/with_stats.nsys-rep python nvtx_tag_demo.py"
echo ""

