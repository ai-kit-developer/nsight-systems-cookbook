#!/bin/bash
# 示例 4: 采样频率设置
# 展示如何调整 CPU 和 GPU 采样频率

echo "=========================================="
echo "示例 4: 采样频率设置"
echo "=========================================="
echo ""

echo "4.1 设置 CPU 采样频率"
echo "命令: nsys profile --sampling-frequency=1000 --output=output.nsys-rep python script.py"
echo "说明: CPU 采样频率设置为 1000Hz（默认通常较低）"
echo "     更高的频率 = 更详细但文件更大"
echo ""

echo "4.2 设置 GPU 指标采样频率"
echo "命令: nsys profile --gpu-metrics-frequency=100 --output=output.nsys-rep python script.py"
echo "说明: GPU 指标（如利用率、内存使用）采样频率为 100Hz"
echo ""

echo "4.3 同时设置 CPU 和 GPU 采样频率"
echo "命令: nsys profile --sampling-frequency=1000 --gpu-metrics-frequency=100 --output=output.nsys-rep python script.py"
echo "说明: 同时优化 CPU 和 GPU 采样频率"
echo ""

echo "实际使用示例:"
echo "  nsys profile --trace=cuda,nvtx --sampling-frequency=1000 --gpu-metrics-frequency=100 --output=nsys_profiles/high_freq.nsys-rep python nvtx_tag_demo.py"
echo ""

