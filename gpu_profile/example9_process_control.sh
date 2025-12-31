#!/bin/bash
# 示例 9: 进程控制
# 展示如何控制被分析的进程

echo "=========================================="
echo "示例 9: 进程控制"
echo "=========================================="
echo ""

echo "9.1 附加到运行中的进程"
echo "命令: nsys profile --attach=<PID> --output=output.nsys-rep"
echo "说明: 附加到指定进程 ID 的进程进行分析"
echo "      不需要指定要运行的程序"
echo ""

echo "9.2 附加进程 + 持续时间"
echo "命令: nsys profile --attach=<PID> --duration=30 --output=output.nsys-rep"
echo "说明: 附加到进程并分析 30 秒"
echo ""

echo "9.3 杀死进程后停止"
echo "命令: nsys profile --kill=none --output=output.nsys-rep python script.py"
echo "说明: 不杀死进程（默认行为）"
echo "      选项: none, sigterm, sigkill"
echo ""

echo "实际使用示例:"
echo "  # 首先运行程序获取 PID"
echo "  python nvtx_tag_demo.py &"
echo "  PID=\$!"
echo "  # 然后附加分析"
echo "  nsys profile --attach=\$PID --duration=5 --output=nsys_profiles/attached.nsys-rep"
echo ""

