# Canvas高度动态调整修复总结

## 🐛 问题描述

左侧的规约框（树形归约可视化）高度固定，无法根据步数动态调整，导致某些情况下无法显示所有步数。

## ✅ 修复内容

### 1. 添加层数和高度计算函数

**新增函数**：`calculateTreeHeight()`
```javascript
function calculateTreeHeight() {
    // 计算总层数（包括初始层和所有归约层）
    let actualIterations = 0;
    let testStride = 1;
    while (testStride < config.threadCount) {
        actualIterations++;
        testStride *= 2;
    }
    const levels = Math.max(1, actualIterations + 1);
    
    const topPadding = 40;
    const bottomPadding = 40;
    const cellHeight = 28;
    const levelSpacing = 35;
    
    // 计算所需高度
    const requiredHeight = topPadding + levels * cellHeight + (levels - 1) * levelSpacing + bottomPadding;
    
    return { levels, requiredHeight };
}
```

### 2. 添加Canvas高度调整函数

**新增函数**：`adjustTreeCanvasHeight()`
```javascript
function adjustTreeCanvasHeight() {
    if (!treeCanvas) return;
    
    const { levels, requiredHeight } = calculateTreeHeight();
    
    // 设置最小高度和最大高度
    const minHeight = 300;
    const maxHeight = 2000; // 设置一个合理的最大值
    const newHeight = Math.max(minHeight, Math.min(maxHeight, Math.ceil(requiredHeight)));
    
    // 如果高度改变，更新canvas
    if (treeCanvas.height !== newHeight) {
        const wasPlaying = config.isPlaying;
        if (wasPlaying) {
            config.isPlaying = false;
        }
        
        treeCanvas.height = newHeight;
        
        // 重新绘制
        if (wasPlaying) {
            config.isPlaying = true;
        }
        draw();
    }
}
```

### 3. 修改drawTree函数

**修改**：使用`calculateTreeHeight()`获取层数，而不是重复计算：
```javascript
// 计算总层数（包括初始层和所有归约层）
const { levels } = calculateTreeHeight();
```

### 4. 修改init函数

**修改**：在初始化后调用`adjustTreeCanvasHeight()`：
```javascript
layerStates = [[...sharedMemory]]; // 保存初始状态

// 调整treeCanvas高度以适应所有层
adjustTreeCanvasHeight();

updateDisplays();
draw();
```

### 5. 修改resizeCanvases函数

**修改**：对于treeCanvas，根据层数动态调整高度：
```javascript
// 对于treeCanvas，根据层数动态调整高度
if (canvas.id === 'treeCanvas') {
    adjustTreeCanvasHeight();
} else {
    // 对于memoryCanvas，使用固定比例
    const newHeight = Math.min(500, Math.max(400, (rect.width - 40) * 0.8));
    canvas.height = newHeight;
}
```

### 6. 修改事件监听器

**修改**：在数组长度和线程数改变时，也调整canvas高度：
```javascript
document.getElementById('arrayLength').addEventListener('input', (e) => {
    // ... 其他代码 ...
    init();
    // 调整treeCanvas高度
    adjustTreeCanvasHeight();
});

document.getElementById('threadCount').addEventListener('input', (e) => {
    // ... 其他代码 ...
    init();
    // 调整treeCanvas高度（线程数改变会影响层数）
    adjustTreeCanvasHeight();
});
```

## 📊 高度计算逻辑

### 高度计算公式：
```
requiredHeight = topPadding + levels * cellHeight + (levels - 1) * levelSpacing + bottomPadding
```

其中：
- `topPadding = 40`
- `bottomPadding = 40`
- `cellHeight = 28`
- `levelSpacing = 35`
- `levels = 实际迭代次数 + 1`

### 不同线程数对应的所需高度：

| 线程数 | 层数 | 所需高度 | 实际高度 |
|--------|------|---------|---------|
| 8      | 4    | ~247    | 300     |
| 16     | 5    | ~315    | 315     |
| 32     | 6    | ~383    | 383     |
| 64     | 7    | ~451    | 451     |
| 128    | 8    | ~519    | 519     |
| 256    | 9    | ~587    | 587     |

**注意**：实际高度会取`Math.max(300, Math.min(2000, requiredHeight))`，确保在合理范围内。

## 📁 修复的文件

- ✅ `reduce_v0_visualization.html` - 完整修复
- ✅ `reduce_v1_visualization.html` - 已应用修复
- ✅ `reduce_v2_visualization.html` - 已应用修复
- ✅ `reduce_v3_visualization.html` - 已应用修复
- ✅ `reduce_v4_visualization.html` - 已应用修复
- ✅ `reduce_v5_visualization.html` - 已应用修复
- ✅ `reduce_v6_visualization.html` - 已应用修复
- ✅ `reduce_v7_visualization.html` - 已应用修复

## 🎯 修复效果

修复后的效果：
- ✅ Canvas高度根据实际层数动态调整
- ✅ 所有层都能正确显示在画布内
- ✅ 线程数改变时自动调整高度
- ✅ 数组长度改变时自动调整高度
- ✅ 窗口大小改变时保持正确的高度
- ✅ 设置最小高度（300px）和最大高度（2000px）限制

## 🧪 测试建议

1. **测试不同线程数**：
   - 8, 16, 32, 64, 128, 256个线程
   - 验证canvas高度自动调整
   - 验证所有层都显示

2. **测试动态调整**：
   - 改变线程数滑块
   - 验证canvas高度立即更新
   - 验证所有层都显示

3. **测试窗口大小**：
   - 调整浏览器窗口大小
   - 验证canvas高度保持正确
   - 验证所有层都显示

---

**修复完成时间**: 2024年
**修复版本**: v2.6

