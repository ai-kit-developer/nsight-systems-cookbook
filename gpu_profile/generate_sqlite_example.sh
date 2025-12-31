#!/bin/bash
# 示例：如何生成 SQLite 文件

echo "=========================================="
echo "生成包含 SQLite 的性能分析文件"
echo "=========================================="
echo ""

OUTPUT_FILE="profile_with_sqlite.nsys-rep"

echo "运行 nsys profile 并导出 SQLite..."
nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --export=sqlite \
  --output="$OUTPUT_FILE" \
  python example1_memory_allocation.py

echo ""
echo "检查生成的文件："
ls -lh "${OUTPUT_FILE%.nsys-rep}"* 2>/dev/null || echo "文件未找到"

echo ""
echo "如果生成了 SQLite 文件，可以查看："
echo "  sqlite3 ${OUTPUT_FILE%.nsys-rep}.sqlite '.tables'"
