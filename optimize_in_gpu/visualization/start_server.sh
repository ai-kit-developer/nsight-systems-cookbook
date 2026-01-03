#!/bin/bash
# CUDA GPU 性能优化算法可视化服务器启动脚本

cd "$(dirname "$0")"
echo "正在启动 CUDA GPU 性能优化算法可视化服务器..."
python3 server.py
