#!/bin/bash
# 安装最新版本的 NVIDIA Nsight Systems (nsys)

set -e

echo "=========================================="
echo "安装最新版本的 NVIDIA Nsight Systems"
echo "=========================================="
echo ""

# 检查当前版本
if command -v nsys &> /dev/null; then
    CURRENT_VERSION=$(nsys --version 2>&1 | head -1)
    echo "当前安装的版本: $CURRENT_VERSION"
else
    echo "未检测到 nsys，将进行全新安装"
fi

echo ""
echo "最新版本信息:"
echo "根据 NVIDIA 官网，最新版本是 2025.6.1"
echo ""

# 创建临时目录
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

echo "请选择安装方式："
echo "1. 从 NVIDIA 官网手动下载并安装（推荐）"
echo "2. 尝试自动下载（可能需要登录）"
echo ""
read -p "请选择 (1/2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "请按照以下步骤操作："
    echo "1. 访问: https://developer.nvidia.com/nsight-systems/get-started"
    echo "2. 下载适用于 Linux 的安装包（.run 文件）"
    echo "3. 将下载的文件放到当前目录: $TMP_DIR"
    echo "4. 运行以下命令安装："
    echo "   sudo sh ./nsight-systems-linux-*.run"
    echo ""
    echo "或者，如果您已经下载了安装包，请输入文件路径："
    read -p "安装包路径: " installer_path
    
    if [ -f "$installer_path" ]; then
        echo ""
        echo "开始安装..."
        sudo sh "$installer_path" --accept-eula --quiet
        echo "安装完成！"
    else
        echo "错误: 文件不存在: $installer_path"
        exit 1
    fi
    
elif [ "$choice" = "2" ]; then
    echo ""
    echo "尝试自动下载..."
    echo "注意: 这可能需要 NVIDIA 开发者账号登录"
    echo ""
    
    # 尝试下载（可能需要登录）
    DOWNLOAD_URL="https://developer.download.nvidia.com/nsight-systems/nsight-systems-linux-latest.run"
    
    if wget --user-agent="Mozilla/5.0" "$DOWNLOAD_URL" -O nsight-systems-linux-latest.run 2>&1 | grep -q "200 OK"; then
        echo "下载成功！"
        chmod +x nsight-systems-linux-latest.run
        echo "开始安装..."
        sudo sh ./nsight-systems-linux-latest.run --accept-eula --quiet
        echo "安装完成！"
    else
        echo "自动下载失败，请使用方式 1 手动下载"
        exit 1
    fi
else
    echo "无效的选择"
    exit 1
fi

# 验证安装
echo ""
echo "验证安装..."
if command -v nsys &> /dev/null; then
    NEW_VERSION=$(nsys --version 2>&1 | head -1)
    echo "✓ 安装成功！"
    echo "新版本: $NEW_VERSION"
    echo ""
    echo "nsys 路径: $(which nsys)"
else
    echo "✗ 安装失败，nsys 命令未找到"
    exit 1
fi

# 清理
cd /
rm -rf "$TMP_DIR"

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="

