#!/bin/bash
# 示例 8: 高级选项
# 展示一些高级配置选项

echo "=========================================="
echo "示例 8: 高级选项"
echo "=========================================="
echo ""

echo "8.1 强制覆盖输出文件"
echo "命令: nsys profile --force-overwrite=true --output=output.nsys-rep python script.py"
echo "说明: 如果输出文件已存在，强制覆盖（不提示）"
echo ""

echo "8.2 设置输出目录"
echo "命令: nsys profile --output-dir=./profiles --output=output.nsys-rep python script.py"
echo "说明: 指定输出目录（注意：--output 参数仍需要）"
echo ""

echo "8.3 完整的高级配置示例"
echo "命令: nsys profile \\"
echo "  --trace=cuda,nvtx,osrt \\"
echo "  --cuda-memory-usage=true \\"
echo "  --sampling-frequency=1000 \\"
echo "  --gpu-metrics-frequency=100 \\"
echo "  --stats=true \\"
echo "  --force-overwrite=true \\"
echo "  --output=output.nsys-rep \\"
echo "  python script.py"
echo ""
echo "说明: 综合使用多个选项进行详细分析"
echo ""

echo "实际使用示例:"
echo "  nsys profile \\"
echo "    --trace=cuda,nvtx \\"
echo "    --cuda-memory-usage=true \\"
echo "    --sampling-frequency=1000 \\"
echo "    --stats=true \\"
echo "    --force-overwrite=true \\"
echo "    --output=nsys_profiles/advanced.nsys-rep \\"
echo "    python nvtx_tag_demo.py"
echo ""

