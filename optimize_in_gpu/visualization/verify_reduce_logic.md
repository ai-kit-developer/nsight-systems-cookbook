# Reduce v0 归约逻辑验证

## 归约过程分析

### 对于32个线程的归约过程（forward循环）

根据CUDA代码：
```cuda
for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    if (thread_idx % (2 * stride) == 0) {
        shared_data[thread_idx] += shared_data[thread_idx + stride];
    }
}
```

**归约步骤**：
1. **初始状态** (layerStates[0]): 每个线程有自己的值
2. **第1次归约** (stride=1, iteration=0): 
   - 线程0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30参与
   - 结果保存到 layerStates[1]
3. **第2次归约** (stride=2, iteration=1):
   - 线程0,4,8,12,16,20,24,28参与
   - 结果保存到 layerStates[2]
4. **第3次归约** (stride=4, iteration=2):
   - 线程0,8,16,24参与
   - 结果保存到 layerStates[3]
5. **第4次归约** (stride=8, iteration=3):
   - 线程0,16参与
   - 结果保存到 layerStates[4]
6. **第5次归约** (stride=16, iteration=4):
   - 线程0参与
   - 结果保存到 layerStates[5]
7. **完成**: stride变成32，32 >= 32，停止

**总层数**: 6层（初始层 + 5层归约）

## 可视化显示对应关系

- **Level 0**: 初始层，显示 layerStates[0]
- **Level 1**: Stride 1，显示 layerStates[1] (iteration 0的结果)
- **Level 2**: Stride 2，显示 layerStates[2] (iteration 1的结果)
- **Level 3**: Stride 4，显示 layerStates[3] (iteration 2的结果)
- **Level 4**: Stride 8，显示 layerStates[4] (iteration 3的结果)
- **Level 5**: Stride 16，显示 layerStates[5] (iteration 4的结果)

## 关键检查点

1. ✅ **初始化**: layerStates[0] 应该保存初始状态
2. ✅ **每次归约后**: layerStates[iteration + 1] 应该保存归约后的状态
3. ✅ **层数计算**: levels = Math.ceil(Math.log2(32)) + 1 = 6
4. ✅ **显示逻辑**: 每层应该显示对应的layerStates值

## 可能的问题

1. **状态保存时机**: 确保在执行归约后立即保存
2. **层数计算**: 确保计算正确的总层数
3. **显示逻辑**: 确保每层都能找到对应的layerStates值
4. **未开始层**: 确保未开始的层也能显示预测值

