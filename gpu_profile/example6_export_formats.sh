#!/bin/bash
# 示例 6: 导出不同格式
# 展示如何导出为不同格式的报告

echo "=========================================="
echo "示例 6: 导出不同格式"
echo "=========================================="
echo ""

echo "6.1 导出为 SQLite 数据库"
echo "命令: nsys profile --export=sqlite --output=output.nsys-rep python script.py"
echo "说明: 导出为 SQLite 格式，可以用工具分析"
echo ""

echo "6.2 导出为 JSON 格式"
echo "命令: nsys profile --export=json --output=output.nsys-rep python script.py"
echo "说明: 导出为 JSON 格式，便于程序化分析"
echo ""

echo "6.3 导出为 Chrome 跟踪格式"
echo "命令: nsys profile --export=chrome-trace --output=output.nsys-rep python script.py"
echo "说明: 导出为 Chrome tracing 格式，可在 chrome://tracing 中查看"
echo ""

echo "6.4 导出多个格式"
echo "命令: nsys profile --export=sqlite,json --output=output.nsys-rep python script.py"
echo "说明: 同时导出为多种格式"
echo ""

echo "6.5 强制覆盖已存在的导出文件"
echo "命令: nsys profile --export=sqlite --force-export=true --output=output.nsys-rep python script.py"
echo "说明: 如果导出文件已存在，强制覆盖"
echo ""

echo "实际使用示例:"
echo "  nsys profile --trace=cuda,nvtx --export=sqlite,json --output=nsys_profiles/exported.nsys-rep python nvtx_tag_demo.py"
echo ""

