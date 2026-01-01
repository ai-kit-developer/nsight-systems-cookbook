#!/bin/bash
###############################################################################
# Profile script for example1_memory_allocation.py
# GPU 内存分配问题分析脚本
#
# 功能:
#   - 使用 nsys 分析 GPU 内存分配性能
#   - 生成 .nsys-rep 文件供 Nsight Systems 分析
#   - 自动检查依赖和文件存在性
#
# 用法:
#   ./profile_example1_memory_allocation.sh
#
# 输出:
#   - example1_memory_allocation.nsys-rep: nsys 分析报告
###############################################################################

set -e  # 遇到错误立即退出

# 配置变量
SCRIPT_NAME="example1_memory_allocation.py"
OUTPUT_FILE="example1_memory_allocation.nsys-rep"
PYTHON_CMD="python3"  # 可以根据需要改为 python

# 颜色输出（如果支持）
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'  # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Profiling: $SCRIPT_NAME${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查 nsys 是否安装
if ! command -v nsys &> /dev/null; then
    echo -e "${RED}错误: nsys 未找到${NC}"
    echo "请安装 NVIDIA Nsight Systems:"
    echo "  - Ubuntu/Debian: sudo apt install nsight-systems"
    echo "  - 或访问: https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# 检查 Python 文件是否存在
if [ ! -f "$SCRIPT_NAME" ]; then
    echo -e "${RED}错误: $SCRIPT_NAME 未找到${NC}"
    echo "请确保在正确的目录下运行此脚本"
    exit 1
fi

# 检查 Python 是否可用
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo -e "${RED}错误: $PYTHON_CMD 未找到${NC}"
    exit 1
fi

# 显示将要执行的命令
echo -e "${YELLOW}执行命令:${NC}"
echo "  nsys profile --trace=cuda,nvtx \\"
echo "    --cuda-memory-usage=true \\"
echo "    --force-overwrite=true \\"
echo "    --output=$OUTPUT_FILE \\"
echo "    $PYTHON_CMD $SCRIPT_NAME"
echo ""

# 执行 profile
# 使用 || true 确保即使脚本执行失败也能继续
echo -e "${GREEN}开始分析...${NC}"
if nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output="$OUTPUT_FILE" \
    "$PYTHON_CMD" "$SCRIPT_NAME"; then
    echo ""
    echo -e "${GREEN}✓ 分析完成！${NC}"
else
    echo ""
    echo -e "${RED}✗ 分析过程中出现错误${NC}"
    exit 1
fi

# 检查输出文件是否生成
echo ""
if [ -f "$OUTPUT_FILE" ]; then
    file_size=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo -e "${GREEN}✓ 生成文件: $OUTPUT_FILE (大小: $file_size)${NC}"
    echo ""
    echo -e "${BLUE}查看结果:${NC}"
    echo "  - 使用 Nsight Systems GUI:"
    echo "    nsys-ui $OUTPUT_FILE"
    echo ""
    echo "  - 查看统计信息:"
    echo "    nsys stats $OUTPUT_FILE"
    echo ""
    echo "  - 导出为 SQLite 数据库:"
    echo "    nsys export --type=sqlite --output=profile.sqlite $OUTPUT_FILE"
else
    echo -e "${YELLOW}⚠ 警告: 文件未生成，请检查错误信息${NC}"
    exit 1
fi
echo ""

