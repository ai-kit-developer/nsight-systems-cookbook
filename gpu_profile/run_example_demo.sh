#!/bin/bash
# 实际运行示例：演示如何使用 nsys profile 分析 nvtx_tag_demo.py

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Nsys Profile 实际运行示例${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 检查依赖
if ! command -v nsys &> /dev/null; then
    echo -e "${RED}错误: nsys 未找到${NC}"
    echo "请安装 NVIDIA Nsight Systems"
    exit 1
fi

if [ ! -f "nvtx_tag_demo.py" ]; then
    echo -e "${RED}错误: nvtx_tag_demo.py 未找到${NC}"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="nsys_profiles"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}开始运行示例...${NC}\n"

# 示例 1: 基本分析
echo -e "${YELLOW}[示例 1] 基本性能分析${NC}"
echo "命令: nsys profile --trace=cuda,nvtx --output=$OUTPUT_DIR/example1_basic.nsys-rep python nvtx_tag_demo.py"
nsys profile --trace=cuda,nvtx --output="$OUTPUT_DIR/example1_basic.nsys-rep" python nvtx_tag_demo.py
echo -e "${GREEN}✓ 完成\n${NC}"

# 示例 2: 带内存跟踪
echo -e "${YELLOW}[示例 2] 带内存跟踪的分析${NC}"
echo "命令: nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --output=$OUTPUT_DIR/example2_memory.nsys-rep python nvtx_tag_demo.py"
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --output="$OUTPUT_DIR/example2_memory.nsys-rep" python nvtx_tag_demo.py
echo -e "${GREEN}✓ 完成\n${NC}"

# 示例 3: 带统计信息
echo -e "${YELLOW}[示例 3] 带统计信息的分析${NC}"
echo "命令: nsys profile --trace=cuda,nvtx --stats=true --output=$OUTPUT_DIR/example3_stats.nsys-rep python nvtx_tag_demo.py"
nsys profile --trace=cuda,nvtx --stats=true --output="$OUTPUT_DIR/example3_stats.nsys-rep" python nvtx_tag_demo.py
echo -e "${GREEN}✓ 完成\n${NC}"

# 示例 4: 导出多种格式
echo -e "${YELLOW}[示例 4] 导出多种格式${NC}"
echo "命令: nsys profile --trace=cuda,nvtx --export=sqlite,json --output=$OUTPUT_DIR/example4_export.nsys-rep python nvtx_tag_demo.py"
nsys profile --trace=cuda,nvtx --export=sqlite,json --output="$OUTPUT_DIR/example4_export.nsys-rep" python nvtx_tag_demo.py
echo -e "${GREEN}✓ 完成\n${NC}"

# 示例 5: 高频率采样
echo -e "${YELLOW}[示例 5] 高频率采样分析${NC}"
echo "命令: nsys profile --trace=cuda,nvtx --sampling-frequency=1000 --gpu-metrics-frequency=100 --output=$OUTPUT_DIR/example5_highfreq.nsys-rep python nvtx_tag_demo.py"
nsys profile --trace=cuda,nvtx --sampling-frequency=1000 --gpu-metrics-frequency=100 --output="$OUTPUT_DIR/example5_highfreq.nsys-rep" python nvtx_tag_demo.py
echo -e "${GREEN}✓ 完成\n${NC}"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}所有示例运行完成！${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${GREEN}生成的文件:${NC}"
ls -lh "$OUTPUT_DIR"/*.nsys-rep 2>/dev/null || echo "无文件"

echo -e "\n${CYAN}查看结果:${NC}"
echo "1. 使用 Nsight Systems GUI: nsys-ui $OUTPUT_DIR/example1_basic.nsys-rep"
echo "2. 查看统计信息: nsys stats $OUTPUT_DIR/example3_stats.nsys-rep"
echo "3. 导出文件位置: $OUTPUT_DIR/"

