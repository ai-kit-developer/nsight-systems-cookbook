#!/bin/bash
# Nsight Systems (nsys) Profile 示例脚本集合
# 展示不同参数和用法的多个示例

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Nsight Systems Profile 示例脚本${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 检查 nsys 是否安装
if ! command -v nsys &> /dev/null; then
    echo -e "${YELLOW}警告: nsys 未找到，请确保已安装 NVIDIA Nsight Systems${NC}"
    echo "安装方法: https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="nsys_profiles"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}输出目录: $OUTPUT_DIR${NC}\n"

