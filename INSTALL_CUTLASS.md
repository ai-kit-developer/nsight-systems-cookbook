# CUTLASS 安装指南

## 当前状态

由于网络连接问题，自动克隆可能失败。请按照以下方法之一手动安装 CUTLASS。

## 方法 1: 使用 Git 克隆（如果网络正常）

```bash
cd /data
git clone https://github.com/NVIDIA/cutlass.git
```

如果网络较慢，可以使用浅克隆：
```bash
cd /data
git clone --depth 1 https://github.com/NVIDIA/cutlass.git
```

## 方法 2: 下载压缩包（推荐，如果 Git 失败）

1. **访问 GitHub 发布页面**：
   - 打开浏览器访问：https://github.com/NVIDIA/cutlass/releases
   - 或直接下载最新版本：https://github.com/NVIDIA/cutlass/archive/refs/heads/main.zip

2. **下载并解压**：
```bash
cd /data
# 如果使用 wget（需要先下载到本地）
wget https://github.com/NVIDIA/cutlass/archive/refs/heads/main.zip -O cutlass-main.zip
unzip cutlass-main.zip
mv cutlass-main cutlass

# 或使用 curl
curl -L https://github.com/NVIDIA/cutlass/archive/refs/heads/main.zip -o cutlass-main.zip
unzip cutlass-main.zip
mv cutlass-main cutlass
```

## 方法 3: 使用镜像源

如果 GitHub 访问困难，可以尝试：

```bash
cd /data
# 使用 Gitee 镜像（如果可用）
git clone https://gitee.com/mirrors/cutlass.git

# 或使用其他镜像源
```

## 配置环境变量

安装完成后，设置环境变量：

### 临时设置（当前会话有效）
```bash
export CUTLASS_PATH=/data/cutlass/include
export CUTLASS_UTIL_PATH=/data/cutlass/tools/util/include
```

### 永久设置（推荐）

将以下内容添加到 `~/.bashrc` 或 `~/.profile`：

```bash
# CUTLASS 配置
export CUTLASS_PATH=/data/cutlass/include
export CUTLASS_UTIL_PATH=/data/cutlass/tools/util/include
```

然后执行：
```bash
source ~/.bashrc
```

## 验证安装

```bash
# 检查目录是否存在
ls -d $CUTLASS_PATH
ls -d $CUTLASS_UTIL_PATH

# 检查关键文件
ls $CUTLASS_PATH/cutlass/cutlass.h
ls $CUTLASS_UTIL_PATH/cutlass/util/

# 测试构建
cd /data/code/gpu-performance-optimization-cookbook/universe_best_cuda_practice
make 6_cutlass_study
```

## 目录结构

安装成功后，应该有以下目录结构：

```
/data/cutlass/
├── include/
│   └── cutlass/
│       ├── cutlass.h
│       ├── gemm/
│       ├── arch/
│       └── ...
├── tools/
│   └── util/
│       └── include/
│           └── cutlass/
│               └── util/
│                   ├── host_tensor.h
│                   └── ...
└── ...
```

## 快速安装脚本

如果网络正常，可以运行以下脚本：

```bash
#!/bin/bash
cd /data

# 尝试克隆
if git clone https://github.com/NVIDIA/cutlass.git 2>/dev/null; then
    echo "✓ CUTLASS 克隆成功"
else
    echo "✗ Git 克隆失败，尝试下载压缩包..."
    if wget https://github.com/NVIDIA/cutlass/archive/refs/heads/main.zip -O cutlass.zip 2>/dev/null; then
        unzip -q cutlass.zip
        mv cutlass-main cutlass
        rm cutlass.zip
        echo "✓ CUTLASS 下载成功"
    else
        echo "✗ 下载失败，请手动安装"
        exit 1
    fi
fi

# 设置环境变量
export CUTLASS_PATH=/data/cutlass/include
export CUTLASS_UTIL_PATH=/data/cutlass/tools/util/include

# 验证
if [ -d "$CUTLASS_PATH" ] && [ -d "$CUTLASS_UTIL_PATH" ]; then
    echo "✓ CUTLASS 安装成功！"
    echo "CUTLASS_PATH=$CUTLASS_PATH"
    echo "CUTLASS_UTIL_PATH=$CUTLASS_UTIL_PATH"
else
    echo "✗ CUTLASS 安装验证失败"
    exit 1
fi
```

## 注意事项

- CUTLASS 是一个大型库，完整克隆约 500MB+
- 确保有足够的磁盘空间
- 如果使用代理，可能需要配置 git 代理：
  ```bash
  git config --global http.proxy http://proxy.example.com:8080
  git config --global https.proxy https://proxy.example.com:8080
  ```
- 如果网络不稳定，建议使用方法 2（下载压缩包）

## 故障排除

### 问题 1: 克隆超时
**解决方案**: 使用压缩包下载（方法 2）

### 问题 2: 找不到 include 目录
**检查**: 
```bash
ls -la /data/cutlass/
```
如果只有 `.git` 目录，说明克隆未完成，需要重新克隆或下载

### 问题 3: 构建时找不到头文件
**检查环境变量**:
```bash
echo $CUTLASS_PATH
echo $CUTLASS_UTIL_PATH
```
确保路径正确且目录存在
