#!/bin/bash
###############################################################################
# 查询 SQLite 文件中的 NVTX 数据
#
# 功能:
#   - 从 nsys 导出的 SQLite 数据库中查询 NVTX 事件数据
#   - 显示 NVTX 标记的统计信息和示例
#   - 帮助理解代码中的性能标记
#
# 用法:
#   ./query_nvtx_data.sh <sqlite_file>
#   示例: ./query_nvtx_data.sh profile.sqlite
#
# 前置条件:
#   - 需要 sqlite3 命令行工具
#   - SQLite 文件需要包含 NVTX 数据（使用 --trace=nvtx 参数）
###############################################################################

set -e  # 遇到错误立即退出

# 颜色输出（如果支持）
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m'  # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
fi

# 检查参数
if [ $# -eq 0 ]; then
    echo -e "${RED}错误: 缺少参数${NC}"
    echo "用法: $0 <sqlite_file>"
    echo "示例: $0 profile.sqlite"
    echo ""
    echo "提示: 首先需要将 .nsys-rep 文件导出为 SQLite 格式："
    echo "  nsys export --type=sqlite --output=profile.sqlite profile.nsys-rep"
    exit 1
fi

SQLITE_FILE="$1"

# 检查文件是否存在
if [ ! -f "$SQLITE_FILE" ]; then
    echo -e "${RED}错误: 文件不存在: $SQLITE_FILE${NC}"
    exit 1
fi

# 检查 sqlite3 是否可用
if ! command -v sqlite3 &> /dev/null; then
    echo -e "${RED}错误: sqlite3 未找到${NC}"
    echo "请安装 sqlite3: sudo apt install sqlite3"
    exit 1
fi

# 检查文件是否为有效的 SQLite 数据库
if ! sqlite3 "$SQLITE_FILE" "SELECT 1;" &> /dev/null; then
    echo -e "${RED}错误: 文件不是有效的 SQLite 数据库: $SQLITE_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}查询 NVTX 数据: $SQLITE_FILE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查 NVTX_EVENTS 表是否存在
if sqlite3 "$SQLITE_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='NVTX_EVENTS';" | grep -q "NVTX_EVENTS"; then
    echo -e "${GREEN}✓ 找到 NVTX_EVENTS 表${NC}"
    echo ""
    
    # 1. NVTX 事件数量
    echo -e "${CYAN}1. NVTX 事件数量：${NC}"
    event_count=$(sqlite3 "$SQLITE_FILE" "SELECT COUNT(*) FROM NVTX_EVENTS;")
    echo "   $event_count 个事件"
    echo ""
    
    # 2. NVTX 事件示例
    echo -e "${CYAN}2. NVTX 事件示例（前 5 条）：${NC}"
    sqlite3 -header -column "$SQLITE_FILE" "SELECT * FROM NVTX_EVENTS LIMIT 5;" 2>/dev/null || echo "   (无法显示事件详情)"
    echo ""
    
    # 3. NVTX 事件表结构
    echo -e "${CYAN}3. NVTX 事件表结构：${NC}"
    sqlite3 "$SQLITE_FILE" ".schema NVTX_EVENTS" | sed 's/^/   /'
    echo ""
else
    echo -e "${YELLOW}✗ 未找到 NVTX_EVENTS 表${NC}"
    echo "提示: 确保使用了 --trace=nvtx 参数进行 profile"
    echo "     并且代码中使用了 nvtx.annotate() 标记"
    echo ""
fi

# 4. StringIds 中的标记名称（包含常见关键词）
echo -e "${CYAN}4. StringIds 中的标记名称（包含常见关键词）：${NC}"
if sqlite3 "$SQLITE_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='StringIds';" | grep -q "StringIds"; then
    result=$(sqlite3 -header -column "$SQLITE_FILE" \
        "SELECT id, value FROM StringIds \
         WHERE value LIKE '%数据%' OR value LIKE '%计算%' OR value LIKE '%内存%' \
            OR value LIKE '%传输%' OR value LIKE '%同步%' OR value LIKE '%分配%' \
         LIMIT 20;" 2>/dev/null)
    if [ -n "$result" ]; then
        echo "$result" | sed 's/^/   /'
    else
        echo "   未找到匹配的字符串"
    fi
else
    echo "   StringIds 表不存在"
fi
echo ""

# 5. 所有 StringIds 记录数
echo -e "${CYAN}5. 所有 StringIds 记录数：${NC}"
if sqlite3 "$SQLITE_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='StringIds';" | grep -q "StringIds"; then
    string_count=$(sqlite3 "$SQLITE_FILE" "SELECT COUNT(*) FROM StringIds;")
    echo "   $string_count 条记录"
else
    echo "   StringIds 表不存在"
fi
echo ""

# 6. 显示所有可用的表
echo -e "${CYAN}6. 数据库中的所有表：${NC}"
sqlite3 "$SQLITE_FILE" "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;" | sed 's/^/   /'
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}提示：${NC}"
echo "- 如果 NVTX_EVENTS 为空，检查是否使用了 --trace=nvtx"
echo "- 如果 StringIds 中没有标记名称，检查代码中的 NVTX 标记"
echo "- 使用 nsys-ui profile.nsys-rep 可以更好地查看 NVTX 标记"
echo ""
echo "导出 SQLite 文件的方法："
echo "  nsys export --type=sqlite --output=profile.sqlite profile.nsys-rep"
echo -e "${BLUE}========================================${NC}"
