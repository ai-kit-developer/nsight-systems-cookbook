#!/bin/bash
# 运行所有 nsys profile 示例说明
# 这个脚本会显示所有示例的说明，但不会实际执行（避免生成大量文件）

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Nsight Systems Profile 完整示例集合${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 检查 nsys
if ! command -v nsys &> /dev/null; then
    echo -e "${YELLOW}警告: nsys 未找到${NC}"
    echo "请安装 NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems"
    exit 1
fi

echo -e "${GREEN}nsys 版本信息:${NC}"
nsys --version
echo ""

# 创建输出目录
OUTPUT_DIR="nsys_profiles"
mkdir -p "$OUTPUT_DIR"

echo -e "${CYAN}所有示例说明:${NC}\n"

# 运行所有示例脚本（只显示说明，不实际执行）
for script in example*.sh; do
    if [ -f "$script" ]; then
        echo -e "${YELLOW}--- 运行 $script ---${NC}"
        bash "$script"
        echo ""
    fi
done

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}所有示例说明已显示完成！${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${CYAN}提示:${NC}"
echo "1. 查看各个 example*.sh 文件了解详细用法"
echo "2. 取消注释示例命令中的实际执行部分来运行"
echo "3. 使用 Nsight Systems GUI 打开 .nsys-rep 文件查看分析结果"
echo "4. 使用 nsys stats 命令查看统计信息: nsys stats output.nsys-rep"
echo ""

