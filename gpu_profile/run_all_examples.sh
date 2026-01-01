#!/bin/bash
###############################################################################
# 运行所有 nsys profile 示例脚本
#
# 功能:
#   - 检查所有 profile 脚本的可用性
#   - 显示所有示例的说明和用法
#   - 可以选择性地执行所有示例
#
# 用法:
#   ./run_all_examples.sh              # 只显示说明
#   ./run_all_examples.sh --run        # 执行所有示例（会生成大量文件）
#
# 输出:
#   - 显示所有示例脚本的信息
#   - 可选：生成所有 .nsys-rep 文件
###############################################################################

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

# 检查是否要实际执行
RUN_ALL=false
if [ "$1" == "--run" ] || [ "$1" == "-r" ]; then
    RUN_ALL=true
    echo -e "${YELLOW}警告: 将执行所有示例，这会生成大量文件！${NC}"
    read -p "继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Nsight Systems Profile 完整示例集合${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 检查 nsys
if ! command -v nsys &> /dev/null; then
    echo -e "${RED}错误: nsys 未找到${NC}"
    echo "请安装 NVIDIA Nsight Systems:"
    echo "  - Ubuntu/Debian: sudo apt install nsight-systems"
    echo "  - 或访问: https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# 显示 nsys 版本信息
echo -e "${GREEN}nsys 版本信息:${NC}"
nsys --version
echo ""

# 创建输出目录
OUTPUT_DIR="nsys_profiles"
mkdir -p "$OUTPUT_DIR"
echo -e "${CYAN}输出目录: $OUTPUT_DIR${NC}\n"

# 查找所有 profile 脚本
PROFILE_SCRIPTS=()
for script in profile_example*.sh; do
    if [ -f "$script" ]; then
        PROFILE_SCRIPTS+=("$script")
    fi
done

if [ ${#PROFILE_SCRIPTS[@]} -eq 0 ]; then
    echo -e "${YELLOW}警告: 未找到任何 profile 脚本${NC}"
    exit 1
fi

echo -e "${CYAN}找到 ${#PROFILE_SCRIPTS[@]} 个示例脚本:${NC}\n"

# 处理每个脚本
SUCCESS_COUNT=0
FAIL_COUNT=0

for script in "${PROFILE_SCRIPTS[@]}"; do
    echo -e "${YELLOW}----------------------------------------${NC}"
    echo -e "${YELLOW}脚本: $script${NC}"
    echo -e "${YELLOW}----------------------------------------${NC}"
    
    if [ "$RUN_ALL" = true ]; then
        # 实际执行脚本
        echo -e "${GREEN}执行中...${NC}"
        if bash "$script" 2>&1; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo -e "${GREEN}✓ 完成${NC}"
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo -e "${RED}✗ 失败${NC}"
        fi
    else
        # 只显示脚本信息
        echo "  (使用 --run 参数来实际执行)"
        if [ -f "$script" ]; then
            # 显示脚本的前几行注释
            head -n 10 "$script" | grep -E "^#|^$" | sed 's/^#/  /' || true
        fi
    fi
    echo ""
done

# 显示总结
echo -e "${BLUE}========================================${NC}"
if [ "$RUN_ALL" = true ]; then
    echo -e "${GREEN}执行完成！${NC}"
    echo -e "成功: ${GREEN}$SUCCESS_COUNT${NC}, 失败: ${RED}$FAIL_COUNT${NC}"
else
    echo -e "${GREEN}所有示例脚本已列出！${NC}"
fi
echo -e "${BLUE}========================================${NC}\n"

# 显示使用提示
echo -e "${CYAN}使用提示:${NC}"
echo "1. 查看各个 profile_example*.sh 文件了解详细用法"
if [ "$RUN_ALL" = false ]; then
    echo "2. 使用 --run 参数执行所有示例: $0 --run"
fi
echo "3. 使用 Nsight Systems GUI 打开 .nsys-rep 文件查看分析结果:"
echo "   nsys-ui <output_file>.nsys-rep"
echo "4. 查看统计信息: nsys stats <output_file>.nsys-rep"
echo "5. 导出为 SQLite: nsys export --type=sqlite --output=profile.sqlite <output_file>.nsys-rep"
echo ""

