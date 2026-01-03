# CUDA GPU 性能优化算法可视化集合

这是一个完整的 CUDA GPU 性能优化算法可视化系统，包含所有主要算法的交互式可视化演示。

## 📁 文件结构

```
visualization/
├── index.html                    # 主页面（标签页导航）
├── server.py                     # Web 服务器脚本
├── start_server.sh               # 启动服务器脚本
├── generate_visualizations.py    # 生成可视化页面脚本
├── README.md                     # 使用说明文档
├── README_VISUALIZATION.md       # Reduce 可视化详细说明
├── FILE_LIST.md                  # 文件清单
│
├── Reduce 归约算法（8个版本 + 索引页）
├── reduce_index.html            # Reduce 版本选择页面
├── reduce_visualization.html     # Reduce 基础版本可视化
├── reduce_v0_visualization.html  # Reduce v0 Baseline 可视化
├── reduce_v1_visualization.html  # Reduce v1 消除分支发散 可视化
├── reduce_v2_visualization.html  # Reduce v2 消除 Bank 冲突 可视化
├── reduce_v3_visualization.html  # Reduce v3 加载时加法 可视化
├── reduce_v4_visualization.html  # Reduce v4 展开最后一个 Warp 可视化
├── reduce_v5_visualization.html  # Reduce v5 完全展开循环 可视化
├── reduce_v6_visualization.html  # Reduce v6 多元素处理 可视化
└── reduce_v7_visualization.html  # Reduce v7 Shuffle 指令 可视化
│
├── 其他算法
├── elementwise.html              # Elementwise 逐元素操作可视化
├── spmv.html                     # SpMV 稀疏矩阵-向量乘法可视化
├── spmm.html                     # SpMM 稀疏矩阵-矩阵乘法可视化
├── sgemm.html                    # SGEMM 矩阵-矩阵乘法可视化
└── sgemv.html                    # SGEMV 矩阵-向量乘法可视化
```

## 🚀 快速开始

### 方法 1: 使用提供的服务器脚本（推荐）

```bash
cd /data/code/gpu-performance-optimization-cookbook/optimize_in_gpu/visualization
./start_server.sh
# 或者
python3 server.py
```

然后在浏览器中打开: http://localhost:8000

### 方法 2: 使用 Python 内置服务器

```bash
cd /data/code/gpu-performance-optimization-cookbook/optimize_in_gpu/visualization
python3 -m http.server 8000
```

然后在浏览器中打开: http://localhost:8000

### 方法 3: 直接打开文件

直接在浏览器中打开 `index.html` 文件（某些功能可能受限）

## 📖 使用说明

### 主页面 (index.html)

主页面使用标签页（Tabs）组织所有算法的可视化：

- **🔬 Reduce 归约**: 包含8个优化版本的完整学习路径（v0-v7）
- **⚡ Elementwise 逐元素操作**: 向量化内存访问优化演示
- **🔢 SpMV 稀疏矩阵-向量乘法**: CSR格式与Warp归约
- **🔢 SpMM 稀疏矩阵-矩阵乘法**: CSR格式与分块计算
- **🔢 SGEMM 矩阵-矩阵乘法**: 分块与共享内存优化
- **🔢 SGEMV 矩阵-向量乘法**: Warp归约优化

### 功能特性

1. **标签页导航**: 点击顶部标签页切换不同算法
2. **懒加载**: 只在切换到标签页时加载对应的可视化内容
3. **PC端优化**: 专为PC端宽屏显示设计（最大宽度1920px）
4. **响应式设计**: 适配不同屏幕尺寸

### 每个可视化页面包含

1. **算法说明**: 详细的算法说明和优化点
2. **交互控制**:
   - 参数调整滑块
   - 播放/暂停按钮
   - 重置按钮
   - 单步执行按钮
3. **实时可视化**:
   - Canvas 绘制的动态图形
   - 实时统计信息
   - 图例说明
4. **性能指标**: 显示关键性能数据

## 🎯 支持的算法

### 1. Reduce 归约（8个版本）
- **v0 Baseline**: 基础树形归约算法
- **v1**: 消除分支发散
- **v2**: 消除 Bank 冲突
- **v3**: 加载时加法
- **v4**: 展开最后一个 Warp
- **v5**: 完全展开循环
- **v6**: 多元素处理
- **v7**: Shuffle 指令（最优版本）
- 每个版本包含：树形归约过程可视化、共享内存状态展示、线程执行状态、性能对比

### 2. Elementwise 逐元素操作
- 向量化内存访问模式（float, float2, float4）
- 线程执行状态
- 性能对比表

### 3. SpMV 稀疏矩阵-向量乘法
- CSR格式稀疏矩阵展示
- 线程组执行与Warp归约过程
- 部分和计算可视化

### 4. SpMM 稀疏矩阵-矩阵乘法
- 矩阵计算过程可视化
- 线程执行状态
- CSR格式使用演示

### 5. SGEMM 矩阵-矩阵乘法
- 分块计算策略
- 共享内存加载和使用
- K维度迭代过程

### 6. SGEMV 矩阵-向量乘法
- 矩阵-向量乘法过程
- Warp归约过程可视化
- 每个warp处理一行的策略

## 💡 使用技巧

1. **调整参数**: 使用滑块调整算法参数，观察可视化变化
2. **单步执行**: 使用单步执行按钮逐步理解算法执行过程
3. **播放动画**: 使用播放按钮观看完整的算法执行动画
4. **切换算法**: 使用顶部标签页快速切换不同算法

## 🔧 技术栈

- **HTML5**: 页面结构
- **CSS3**: 样式和动画
- **JavaScript**: 交互逻辑和Canvas绘制
- **Canvas API**: 图形绘制

## 📝 注意事项

1. 建议使用现代浏览器（Chrome, Firefox, Edge, Safari）
2. 某些浏览器可能需要启用本地文件访问权限
3. 使用本地服务器可以获得最佳体验
4. 建议在PC端使用，以获得最佳的宽屏显示效果

## 🎓 学习建议

1. **从 Reduce 算法开始**：
   - 从 v0 Baseline 开始，理解基础的并行归约概念
   - 按顺序学习 v1-v7，理解每个版本的优化技巧
   - 对比不同版本的可视化，理解优化带来的改进
   - 查看性能对比表，了解优化效果
2. **学习 Elementwise 操作**：理解向量化内存访问
3. **深入理解稀疏矩阵操作**：SpMV, SpMM
4. **掌握密集矩阵操作**：SGEMM, SGEMV

## 📊 Reduce 版本性能对比

| 版本 | 优化技巧 | 耗时 (ns) | 带宽 (GB/s) | 加速比 |
|------|---------|-----------|-------------|--------|
| v0 | Baseline | 804,187 | 159.2 | 1.0x |
| v1 | 消除分支发散 | 506,365 | 252.8 | 1.59x |
| v2 | 消除 Bank 冲突 | 397,886 | 321.7 | 1.27x |
| v3 | 加载时加法 | 213,182 | 600.4 | 1.86x |
| v4 | 展开最后一个 Warp | 169,215 | 756.4 | 1.26x |
| v5 | 完全展开循环 | 166,752 | 767.6 | 1.01x |
| v6 | 多元素处理 | 166,656 | 768.1 | 1.00x |
| v7 | Shuffle 指令 | 166,175 | 770.3 | 1.00x |

*性能数据基于 V100 GPU，32MB 数据大小测试*

---

**通过交互式可视化，深入理解 GPU 并行计算优化技术！** 🚀

