#!/bin/bash
# 示例 5: 持续时间控制
# 展示如何控制分析的时间范围

echo "=========================================="
echo "示例 5: 持续时间控制"
echo "=========================================="
echo ""

echo "5.1 设置分析持续时间"
echo "命令: nsys profile --duration=10 --output=output.nsys-rep python script.py"
echo "说明: 只分析前 10 秒（适合长时间运行的程序）"
echo ""

echo "5.2 等待后开始分析"
echo "命令: nsys profile --wait=5 --duration=10 --output=output.nsys-rep python script.py"
echo "说明: 等待 5 秒后开始分析，然后分析 10 秒"
echo "      适用于跳过初始化阶段"
echo ""

echo "5.3 程序退出时停止"
echo "命令: nsys profile --stop-on-exit=true --output=output.nsys-rep python script.py"
echo "说明: 程序正常退出时自动停止分析（默认行为）"
echo ""

echo "实际使用示例:"
echo "  nsys profile --trace=cuda,nvtx --duration=5 --output=nsys_profiles/duration_5s.nsys-rep python nvtx_tag_demo.py"
echo ""

