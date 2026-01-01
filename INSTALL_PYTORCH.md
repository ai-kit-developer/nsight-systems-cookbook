# PyTorch 安装指南

## 快速安装

### 方法 1: 使用安装脚本（推荐）

```bash
cd /data
bash install_pytorch.sh
```

脚本会自动：
- 安装 PyTorch Python 包
- 下载并安装 libtorch (C++ API)
- 验证安装

### 方法 2: 手动安装

#### 2.1 安装 PyTorch Python 包

```bash
# CUDA 12.1 版本（与 CUDA 12.5 兼容）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 如果网络较慢，可以使用国内镜像
pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 2.2 安装 libtorch (C++ API)

```bash
cd /data

# 下载 libtorch (PyTorch 2.5.1 + CUDA 12.1)
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip -O libtorch.zip

# 解压
unzip libtorch.zip
mv libtorch /data/libtorch
rm libtorch.zip

# 设置环境变量
export TORCH_PATH=/data/libtorch
```

## 验证安装

### 验证 Python 包

```bash
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 验证 libtorch

```bash
# 检查目录结构
ls -d /data/libtorch/include
ls -d /data/libtorch/lib

# 测试编译
export TORCH_PATH=/data/libtorch
cd universe_best_cuda_practice/flash_attention
make example-app
```

## 环境变量配置

### 临时设置（当前会话）

```bash
export TORCH_PATH=/data/libtorch
```

### 永久设置（推荐）

```bash
echo 'export TORCH_PATH=/data/libtorch' >> ~/.bashrc
source ~/.bashrc
```

## 版本信息

- **PyTorch 版本**: 2.5.1
- **CUDA 版本**: 12.1 (与系统 CUDA 12.5 兼容)
- **Python 版本**: 3.13.11
- **libtorch 路径**: /data/libtorch

## 使用示例

### Python 使用

```python
import torch

# 创建张量
x = torch.rand(5, 3)
print(x)

# CUDA 张量
if torch.cuda.is_available():
    x_cuda = x.cuda()
    print(x_cuda)
```

### C++ 使用 (flash_attention)

```bash
# 设置环境变量
export TORCH_PATH=/data/libtorch

# 构建
cd universe_best_cuda_practice/flash_attention
make example-app
make flash-atten-main

# 运行
./bin/example-app
./bin/flash-atten-main
```

## 故障排除

### 问题 1: pip 安装失败

**解决方案**:
```bash
# 使用国内镜像
pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用 conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 问题 2: libtorch 下载失败

**解决方案**:
1. 检查网络连接
2. 手动下载: 访问 https://pytorch.org/get-started/locally/
3. 选择 "LibTorch" 标签页，下载对应版本

### 问题 3: 编译时找不到 PyTorch

**检查环境变量**:
```bash
echo $TORCH_PATH
```

**确保路径正确**:
```bash
ls -d $TORCH_PATH/include
ls -d $TORCH_PATH/lib
```

### 问题 4: CUDA 版本不匹配

PyTorch 2.5.1 支持 CUDA 12.1，与 CUDA 12.5 兼容。如果遇到问题，可以：
1. 检查 CUDA 版本: `nvcc --version`
2. 使用对应的 PyTorch 版本

## 相关链接

- PyTorch 官网: https://pytorch.org/
- PyTorch 下载: https://pytorch.org/get-started/locally/
- libtorch 文档: https://pytorch.org/cppdocs/

