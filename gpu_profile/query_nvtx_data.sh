#!/bin/bash
# 查询 SQLite 文件中的 NVTX 数据

if [ $# -eq 0 ]; then
    echo "用法: $0 <sqlite_file>"
    echo "示例: $0 profile.sqlite"
    exit 1
fi

SQLITE_FILE="$1"

if [ ! -f "$SQLITE_FILE" ]; then
    echo "错误: 文件不存在: $SQLITE_FILE"
    exit 1
fi

echo "=========================================="
echo "查询 NVTX 数据: $SQLITE_FILE"
echo "=========================================="
echo ""

# 检查 NVTX_EVENTS 表是否存在
if sqlite3 "$SQLITE_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='NVTX_EVENTS';" | grep -q "NVTX_EVENTS"; then
    echo "✓ 找到 NVTX_EVENTS 表"
    echo ""
    
    echo "1. NVTX 事件数量："
    sqlite3 "$SQLITE_FILE" "SELECT COUNT(*) FROM NVTX_EVENTS;"
    echo ""
    
    echo "2. NVTX 事件示例（前 5 条）："
    sqlite3 "$SQLITE_FILE" "SELECT * FROM NVTX_EVENTS LIMIT 5;"
    echo ""
    
    echo "3. NVTX 事件表结构："
    sqlite3 "$SQLITE_FILE" ".schema NVTX_EVENTS"
    echo ""
else
    echo "✗ 未找到 NVTX_EVENTS 表"
    echo "提示: 确保使用了 --trace=nvtx 参数"
    echo ""
fi

echo "4. StringIds 中的标记名称（包含常见关键词）："
sqlite3 "$SQLITE_FILE" "SELECT id, value FROM StringIds WHERE value LIKE '%数据%' OR value LIKE '%计算%' OR value LIKE '%内存%' OR value LIKE '%传输%' LIMIT 20;" 2>/dev/null || echo "未找到匹配的字符串"
echo ""

echo "5. 所有 StringIds 记录数："
sqlite3 "$SQLITE_FILE" "SELECT COUNT(*) FROM StringIds;" 2>/dev/null || echo "StringIds 表不存在"
echo ""

echo "=========================================="
echo "提示："
echo "- 如果 NVTX_EVENTS 为空，检查是否使用了 --trace=nvtx"
echo "- 如果 StringIds 中没有标记名称，检查代码中的 NVTX 标记"
echo "- 使用 nsys-ui profile.nsys-rep 可以更好地查看 NVTX 标记"
echo "=========================================="
