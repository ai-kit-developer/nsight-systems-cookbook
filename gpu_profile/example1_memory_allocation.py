"""
示例 1: GPU 内存分配问题

问题描述：
----------
频繁的内存分配和释放是 GPU 编程中常见的性能问题。

问题表现：
1. 每次迭代都分配新内存，导致内存分配器频繁工作
2. 内存分配和释放的开销累积，占用大量时间
3. 可能导致内存碎片化，影响后续分配效率
4. GPU 内存管理需要与 CPU 同步，增加延迟

性能影响：
- 内存分配时间可能占总时间的 20-50%
- 频繁分配导致 GPU 流水线中断
- 内存碎片化降低可用内存利用率

优化思路：
- 预先分配足够大的内存缓冲区
- 在循环中重用已分配的内存
- 使用内存池管理策略
"""

import nvtx
import numpy as np
import time

def get_color(name):
    """颜色辅助函数"""
    colors = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "orange": (1.0, 0.5, 0.0),
    }
    return colors.get(name, (0.5, 0.5, 0.5))

def bad_practice_frequent_allocation(size=1024, iterations=100):
    """
    不好的做法：频繁分配和释放内存
    
    问题分析：
    ----------
    这个函数展示了典型的"频繁内存分配"反模式：
    
    1. 在循环中分配内存（第 1 个问题点）
       - 每次迭代都调用 np.random.rand() 创建新数组
       - 每次分配都需要：
         * 向 GPU 内存管理器请求内存
         * 可能触发内存碎片整理
         * 需要同步 CPU-GPU
    
    2. 立即释放内存（第 2 个问题点）
       - del data 后内存立即释放
       - 释放操作也需要同步
       - 频繁的分配-释放循环导致开销累积
    
    3. 性能影响
       - 分配开销：每次分配可能需要 0.1-1ms
       - 同步开销：CPU 需要等待 GPU 内存操作完成
       - 碎片化：频繁分配释放可能导致内存碎片
    
    在 nsys 时间线中你会看到：
    - 大量短时间的"分配内存"标记
    - GPU Memory 时间线频繁波动
    - 内存分配操作占用大量时间
    """
    print("=== 不好的做法：频繁内存分配 ===")
    print("问题：每次迭代都分配新内存，导致大量分配开销\n")
    
    with nvtx.annotate("频繁内存分配示例", color=get_color("red")):
        for i in range(iterations):
            # ❌ 问题点 1: 在循环中分配内存
            # 每次迭代都创建新数组，触发内存分配
            # 在 GPU 上，这需要：
            #   - 调用内存分配器
            #   - 可能触发内存碎片整理
            #   - CPU-GPU 同步开销
            with nvtx.annotate(f"迭代 {i}: 分配内存", color=get_color("orange")):
                data = np.random.rand(size, size).astype(np.float32)
            
            # 实际计算操作（这里用 sum 模拟）
            with nvtx.annotate(f"迭代 {i}: 计算", color=get_color("yellow")):
                result = np.sum(data)
            
            # ❌ 问题点 2: 立即释放内存
            # 释放操作也需要同步，增加开销
            # 频繁的分配-释放循环导致性能下降
            del data

def good_practice_reuse_allocation(size=1024, iterations=100):
    """
    好的做法：重用已分配的内存
    
    优化分析：
    ----------
    这个函数展示了正确的内存管理方式：
    
    1. 预分配内存（优化点 1）
       - 在循环外一次性分配内存
       - 只触发一次分配操作
       - 避免循环中的分配开销
    
    2. 重用内存（优化点 2）
       - 使用 data[:] = ... 重用已分配的内存
       - 不需要重新分配，只需要填充数据
       - 避免了分配-释放循环
    
    3. 性能提升
       - 分配开销：只有一次，而不是 N 次
       - 无同步开销：循环中不需要同步
       - 无碎片化：内存连续使用
    
    在 nsys 时间线中你会看到：
    - 只有一个"预分配内存"标记
    - GPU Memory 时间线平滑稳定
    - 大部分时间用于实际计算
    """
    print("=== 好的做法：重用内存 ===")
    print("优化：预先分配内存，循环中重用，避免频繁分配\n")
    
    with nvtx.annotate("内存重用示例", color=get_color("green")):
        # ✅ 优化点 1: 预先分配内存
        # 在循环外一次性分配，只触发一次分配操作
        # 避免了循环中的分配开销
        with nvtx.annotate("预分配内存", color=get_color("blue")):
            data = np.empty((size, size), dtype=np.float32)
        
        for i in range(iterations):
            # ✅ 优化点 2: 重用已分配的内存
            # 使用 data[:] = ... 填充数据，而不是创建新数组
            # 这样只需要一次分配，后续都是数据填充操作
            with nvtx.annotate(f"迭代 {i}: 填充数据", color=get_color("orange")):
                data[:] = np.random.rand(size, size).astype(np.float32)
            
            # 实际计算操作
            with nvtx.annotate(f"迭代 {i}: 计算", color=get_color("yellow")):
                result = np.sum(data)
        
        # 注意：内存在整个函数结束后才释放，而不是每次迭代

if __name__ == "__main__":
    print("GPU 内存分配性能分析示例\n")
    
    # 不好的做法
    start = time.time()
    bad_practice_frequent_allocation(size=512, iterations=50)
    bad_time = time.time() - start
    print(f"频繁分配耗时: {bad_time:.4f} 秒\n")
    
    # 好的做法
    start = time.time()
    good_practice_reuse_allocation(size=512, iterations=50)
    good_time = time.time() - start
    print(f"内存重用耗时: {good_time:.4f} 秒\n")
    
    print(f"性能提升: {bad_time/good_time:.2f}x")
    print("\n使用 nsys profile 查看详细的内存分配时间线：")
    print("nsys profile --trace=cuda,nvtx --cuda-memory-usage=true python example1_memory_allocation.py")

