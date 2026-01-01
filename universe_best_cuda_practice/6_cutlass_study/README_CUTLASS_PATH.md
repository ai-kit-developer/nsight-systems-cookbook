# CUTLASS 路径查找说明

## 自动查找机制

构建脚本会自动在以下位置查找 CUTLASS：

### 优先级顺序

1. **环境变量**（最高优先级）
   - `CUTLASS_PATH` - 如果设置了此环境变量，将优先使用
   - `CUTLASS_UTIL_PATH` - 如果设置了此环境变量，将优先使用
   - 如果只设置了 `CUTLASS_PATH`，会自动尝试推断 `CUTLASS_UTIL_PATH`

2. **常见安装位置**（按顺序查找）
   - `/data/cutlass`
   - `$HOME/cutlass`
   - `/usr/local/cutlass`
   - `/opt/cutlass`
   - `$HOME/workspace/cutlass`
   - `$HOME/projects/cutlass`

## 使用方法

### 方法 1: 使用环境变量（推荐）

```bash
export CUTLASS_PATH=/path/to/cutlass/include
export CUTLASS_UTIL_PATH=/path/to/cutlass/tools/util/include
cd universe_best_cuda_practice
make 6_cutlass_study
```

### 方法 2: 安装到常见位置

将 CUTLASS 克隆到以下任一位置：
- `/data/cutlass`
- `$HOME/cutlass`
- `/usr/local/cutlass`
- `/opt/cutlass`

然后直接构建：
```bash
cd universe_best_cuda_practice
make 6_cutlass_study
```

## 验证路径

构建时会显示使用的路径：
```
Using CUTLASS_PATH=/data/cutlass/include
Using CUTLASS_UTIL_PATH=/data/cutlass/tools/util/include
```

## 故障排除

如果找不到 CUTLASS，构建脚本会显示警告信息，提示：
1. 如何克隆 CUTLASS
2. 如何设置环境变量
3. 支持的安装位置列表

