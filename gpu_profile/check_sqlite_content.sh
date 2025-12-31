#!/bin/bash
# 检查 SQLite 文件内容的脚本

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
echo "检查 SQLite 文件: $SQLITE_FILE"
echo "=========================================="
echo ""

echo "1. 所有表："
sqlite3 "$SQLITE_FILE" ".tables"
echo ""

echo "2. 表数量："
sqlite3 "$SQLITE_FILE" "SELECT COUNT(*) as total_tables FROM sqlite_master WHERE type='table';"
echo ""

echo "3. 查找 NVTX 相关表："
sqlite3 "$SQLITE_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND (name LIKE '%NVTX%' OR name LIKE '%MARK%' OR name LIKE '%RANGE%' OR name LIKE '%STRING%');"
echo ""

echo "4. StringIds 表内容（前 10 条）："
sqlite3 "$SQLITE_FILE" "SELECT * FROM StringIds LIMIT 10;" 2>/dev/null || echo "StringIds 不存在或为空"
echo ""

echo "5. StringIds 总记录数："
sqlite3 "$SQLITE_FILE" "SELECT COUNT(*) FROM StringIds;" 2>/dev/null || echo "StringIds 不存在"
echo ""

echo "6. 查找包含 'NVTX' 或常见标记的字符串："
sqlite3 "$SQLITE_FILE" "SELECT * FROM StringIds WHERE value LIKE '%NVTX%' OR value LIKE '%数据%' OR value LIKE '%计算%' LIMIT 10;" 2>/dev/null || echo "未找到匹配的字符串"
echo ""

echo "7. NVTX_EVENTS 表结构："
sqlite3 "$SQLITE_FILE" ".schema NVTX_EVENTS" 2>/dev/null || echo "NVTX_EVENTS 表不存在"
echo ""

echo "8. NVTX_EVENTS 记录数："
sqlite3 "$SQLITE_FILE" "SELECT COUNT(*) FROM NVTX_EVENTS;" 2>/dev/null || echo "NVTX_EVENTS 不存在"
echo ""

echo "9. NVTX_EVENTS 内容（前 5 条）："
sqlite3 "$SQLITE_FILE" "SELECT * FROM NVTX_EVENTS LIMIT 5;" 2>/dev/null || echo "NVTX_EVENTS 为空或不存在"
echo ""

echo "7. 事件相关表："
sqlite3 "$SQLITE_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%EVENT%';"
echo ""

echo "8. CUDA 相关表："
sqlite3 "$SQLITE_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND (name LIKE '%CUDA%' OR name LIKE '%KERNEL%');"
echo ""

echo "=========================================="
echo "提示："
echo "- 如果找不到 NVTX 数据，确保使用了 --trace=nvtx"
echo "- 使用 nsys-ui profile.nsys-rep 可以更好地查看 NVTX 标记"
echo "=========================================="

