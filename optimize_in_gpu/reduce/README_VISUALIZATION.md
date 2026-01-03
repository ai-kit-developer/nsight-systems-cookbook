# CUDA Reduce 可视化系统

这是一个完整的 CUDA Reduce（归约）算法可视化系统，包含 8 个不同优化版本的交互式可视化页面。

## 📁 文件结构

```
reduce/
├── index.html                    # 主页面，展示所有版本
├── reduce_v0_visualization.html  # v0 Baseline 可视化
├── reduce_v1_visualization.html  # v1 消除分支发散 可视化
├── reduce_v2_visualization.html  # v2 消除 Bank 冲突 可视化
├── reduce_v3_visualization.html  # v3 加载时加法 可视化
├── reduce_v4_visualization.html  # v4 展开最后一个 Warp 可视化
├── reduce_v5_visualization.html  # v5 完全展开循环 可视化
├── reduce_v6_visualization.html  # v6 多元素处理 可视化
├── reduce_v7_visualization.html  # v7 Shuffle 指令 可视化
├── server.py                     # Web 服务器脚本
└── generate_visualizations.py   # 可视化页面生成脚本
```

## 🚀 快速开始

### 方法 1: 使用 Python 服务器（推荐）

```bash
cd /data/code/gpu-performance-optimization-cookbook/optimize_in_gpu/reduce
python3 server.py
```

然后在浏览器中打开: http://localhost:8000

### 方法 2: 使用 Python 内置服务器

```bash
cd /data/code/gpu-performance-optimization-cookbook/optimize_in_gpu/reduce
python3 -m http.server 8000
```

然后在浏览器中打开: http://localhost:8000

### 方法 3: 直接打开文件

直接在浏览器中打开 `index.html` 文件（某些功能可能受限）

## 📖 使用说明

### 主页面 (index.html)

主页面展示了所有 8 个 Reduce 版本的概览，包括：
- 每个版本的优化技巧说明
- 已知问题和优化特性
- 性能对比表
- 点击卡片可跳转到对应的可视化页面

### 可视化页面

每个可视化页面包含：

1. **算法说明**: 该版本的详细说明和优化点
2. **交互控制**:
   - 线程数滑块：调整线程数量（8-256）
   - 速度滑块：调整动画播放速度（1-10）
   - 开始/暂停按钮：播放或暂停动画
   - 重置按钮：重置到初始状态
   - 单步执行按钮：逐步执行归约过程

3. **可视化内容**:
   - **树形归约过程**: 展示归约的树形结构，包括每层的线程状态
   - **共享内存状态**: 以柱状图显示每个内存位置的值和访问状态

4. **实时统计**:
   - 当前步长 (Stride)
   - 迭代次数
   - 活跃线程数
   - 完成度百分比

## 🔧 重新生成可视化页面

如果需要修改可视化页面，可以编辑 `generate_visualizations.py` 中的配置，然后运行：

```bash
python3 generate_visualizations.py
```

这将重新生成所有版本的可视化页面。

## 📊 版本对比

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

## 🎯 学习路径建议

1. **v0 Baseline**: 理解基础的树形归约算法
2. **v1**: 学习如何消除分支发散问题
3. **v2**: 学习如何解决 bank 冲突
4. **v3**: 学习如何提高线程利用率
5. **v4-v7**: 学习高级优化技巧

## 💡 提示

- 建议从 v0 开始，逐步理解每个版本的优化点
- 使用单步执行功能可以更清楚地观察每一步的变化
- 调整线程数可以观察不同规模下的归约过程
- 对比不同版本的可视化，理解优化带来的改进

## 📝 注意事项

- 性能数据基于 V100 GPU，32MB 数据大小测试
- 可视化页面使用 Canvas 绘制，需要现代浏览器支持
- 建议使用 Chrome、Firefox 或 Edge 浏览器

## 🔗 相关资源

- [NVIDIA CUDA Reduce 优化文档](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- CUDA 编程最佳实践
- GPU 性能优化技巧

---

**享受学习 CUDA 优化的过程！** 🚀
