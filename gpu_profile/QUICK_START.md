# NVTX 性能分析快速入门指南

## 5 分钟快速开始

### 步骤 1: 运行一个示例
```bash
conda activate python3.12
cd /data/code/gpu_profile
python example1_memory_allocation.py
```

### 步骤 2: 收集性能数据
```bash
nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output=my_profile.nsys-rep \
  python example1_memory_allocation.py
```

### 步骤 3: 查看结果
```bash
# 打开 GUI 查看时间线
nsys-ui my_profile.nsys-rep

# 或查看统计信息
nsys stats my_profile.nsys-rep
```

---

## 分析流程速查

### 1. 内存分配问题
**如何识别**: 内存使用曲线频繁波动，大量短时间分配操作
**查看位置**: GPU Memory 时间线
**优化方法**: 预分配内存，重用缓冲区

### 2. 数据传输问题
**如何识别**: 频繁的 `cudaMemcpy` 调用，传输时间占比高
**查看位置**: CUDA API 时间线
**优化方法**: 批量传输，减少传输次数

### 3. 同步问题
**如何识别**: 频繁的同步操作，GPU 利用率低
**查看位置**: CUDA API 时间线中的同步调用
**优化方法**: 延迟同步，批量处理

### 4. 内核开销问题
**如何识别**: 大量短时间内核，启动开销占比高
**查看位置**: GPU Kernels 时间线
**优化方法**: 合并小内核为大内核

### 5. 负载不均衡问题
**如何识别**: 任务时间差异大，部分资源空闲
**查看位置**: 任务标记的时间分布
**优化方法**: 负载均衡，工作窃取

### 6. 内存访问问题
**如何识别**: 内存带宽利用率低，访问延迟高
**查看位置**: GPU Metrics 中的内存带宽
**优化方法**: 连续访问，优化数据布局

---

## 常用命令

### 基本分析
```bash
nsys profile --trace=cuda,nvtx --output=profile.nsys-rep python script.py
```

### 详细分析（推荐）
```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sampling-frequency=1000 \
  --gpu-metrics-frequency=100 \
  --stats=true \
  --output=profile.nsys-rep \
  python script.py
```

### 查看结果
```bash
nsys-ui profile.nsys-rep          # GUI 查看
nsys stats profile.nsys-rep        # 统计信息
```

---

## 分析检查清单

- [ ] 添加了 NVTX 标记到关键代码段
- [ ] 标记了优化前后的代码
- [ ] 收集了性能数据
- [ ] 查看了统计信息
- [ ] 分析了时间线
- [ ] 识别了性能瓶颈
- [ ] 应用了优化
- [ ] 验证了改进效果

---

## 下一步

详细的分析流程和示例，请参考 [README_examples.md](README_examples.md)

