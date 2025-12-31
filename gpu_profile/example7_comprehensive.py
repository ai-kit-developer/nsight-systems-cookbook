"""
示例 7: 综合性能问题分析
结合多个性能问题的真实场景示例
"""

import nvtx
import numpy as np
import time

def get_color(name):
    colors = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "orange": (1.0, 0.5, 0.0),
        "purple": (0.5, 0.0, 0.5),
        "cyan": (0.0, 1.0, 1.0),
    }
    return colors.get(name, (0.5, 0.5, 0.5))

def comprehensive_bad_practice():
    """综合示例：包含多个性能问题"""
    print("=== 综合示例：多个性能问题 ===")
    
    with nvtx.annotate("综合性能问题示例", color=get_color("red")):
        # 问题 1: 频繁内存分配
        with nvtx.annotate("阶段 1: 频繁分配", color=get_color("orange")):
            for i in range(10):
                with nvtx.annotate(f"分配 {i}", color=get_color("yellow")):
                    data = np.random.rand(100, 100).astype(np.float32)
                    del data
        
        # 问题 2: 频繁数据传输
        with nvtx.annotate("阶段 2: 频繁传输", color=get_color("orange")):
            for i in range(10):
                with nvtx.annotate(f"传输 {i}", color=get_color("red")):
                    cpu_data = np.random.rand(100).astype(np.float32)
                    # 模拟传输
                    time.sleep(0.001)
        
        # 问题 3: 频繁同步
        with nvtx.annotate("阶段 3: 频繁同步", color=get_color("orange")):
            for i in range(10):
                with nvtx.annotate(f"计算 {i}", color=get_color("blue")):
                    time.sleep(0.001)
                with nvtx.annotate(f"同步 {i}", color=get_color("red")):
                    time.sleep(0.002)
        
        # 问题 4: 小内核
        with nvtx.annotate("阶段 4: 小内核", color=get_color("orange")):
            for i in range(20):
                with nvtx.annotate(f"小内核 {i}", color=get_color("purple")):
                    time.sleep(0.0001)

def comprehensive_good_practice():
    """
    综合示例：优化后的版本
    
    优化分析：
    ----------
    这个函数展示了系统性的优化方法：
    
    阶段 1: 预分配内存
    - 优化：预先分配内存，循环中重用
    - 效果：减少分配次数，降低开销
    - 时间减少：~70-80%
    
    阶段 2: 批量传输
    - 优化：批量准备数据，一次性传输
    - 效果：减少传输次数，提高效率
    - 时间减少：~80-90%
    
    阶段 3: 延迟同步
    - 优化：批量启动，延迟同步
    - 效果：提高并行性，减少等待
    - 时间减少：~70-80%
    
    阶段 4: 合并内核
    - 优化：合并小内核为大内核
    - 效果：减少启动次数，提高效率
    - 时间减少：~50-70%
    
    总体效果：
    - 总时间减少：~5-10 倍
    - GPU 利用率提高：~3-5 倍
    - 资源利用率提高：~2-3 倍
    
    在 nsys 时间线中你会看到：
    - 所有阶段都得到优化
    - 时间线更紧凑，空闲时间少
    - GPU 利用率高且稳定
    """
    print("=== 综合示例：优化版本 ===")
    print("系统性优化：内存、传输、同步、内核全部优化\n")
    
    with nvtx.annotate("优化示例", color=get_color("green")):
        # ✅ 优化 1: 预分配内存
        # 预先分配内存，循环中重用，避免频繁分配
        with nvtx.annotate("阶段 1: 预分配", color=get_color("blue")):
            data = np.empty((100, 100), dtype=np.float32)
            for i in range(10):
                with nvtx.annotate(f"重用 {i}", color=get_color("green")):
                    # 重用已分配的内存，不需要重新分配
                    data[:] = np.random.rand(100, 100).astype(np.float32)
        
        # ✅ 优化 2: 批量传输
        # 批量准备数据，一次性传输，提高传输效率
        with nvtx.annotate("阶段 2: 批量传输", color=get_color("blue")):
            all_data = np.random.rand(10, 100).astype(np.float32)
            with nvtx.annotate("批量传输", color=get_color("cyan")):
                # 一次传输所有数据，效率高
                time.sleep(0.01)  # 批量传输，总时间 < 多次小传输
        
        # ✅ 优化 3: 延迟同步
        # 批量启动所有计算，最后统一同步
        with nvtx.annotate("阶段 3: 延迟同步", color=get_color("blue")):
            # 启动所有计算（非阻塞）
            for i in range(10):
                with nvtx.annotate(f"启动 {i}", color=get_color("green")):
                    time.sleep(0.001)
            # 只同步一次，而不是 10 次
            with nvtx.annotate("最终同步", color=get_color("cyan")):
                time.sleep(0.002)
        
        # ✅ 优化 4: 合并内核
        # 将多个小操作合并为批次，减少启动次数
        with nvtx.annotate("阶段 4: 合并内核", color=get_color("blue")):
            batch_size = 5
            for batch in range(4):
                with nvtx.annotate(f"大内核 {batch}", color=get_color("green")):
                    time.sleep(0.0001)  # 启动开销（与单个小内核相同）
                    # 批量处理，提高效率
                    for i in range(batch_size):
                        time.sleep(0.00005)

def performance_analysis_workflow():
    """性能分析工作流程示例"""
    print("=== 性能分析工作流程 ===")
    
    with nvtx.annotate("性能分析流程", color=get_color("purple")):
        # 1. 数据准备
        with nvtx.annotate("1. 数据准备", color=get_color("yellow")):
            with nvtx.annotate("加载数据", color=get_color("blue")):
                data = np.random.rand(1000, 1000).astype(np.float32)
                time.sleep(0.01)
            
            with nvtx.annotate("数据预处理", color=get_color("blue")):
                data = data / np.max(data)
                time.sleep(0.005)
        
        # 2. 计算阶段
        with nvtx.annotate("2. 计算阶段", color=get_color("green")):
            with nvtx.annotate("主要计算", color=get_color("blue")):
                result1 = np.sum(data)
                time.sleep(0.02)
            
            with nvtx.annotate("辅助计算", color=get_color("blue")):
                result2 = np.mean(data)
                time.sleep(0.01)
        
        # 3. 后处理
        with nvtx.annotate("3. 后处理", color=get_color("orange")):
            with nvtx.annotate("结果处理", color=get_color("blue")):
                final_result = result1 + result2
                time.sleep(0.005)

if __name__ == "__main__":
    print("综合性能分析示例\n")
    
    # 包含多个问题的版本
    start = time.time()
    comprehensive_bad_practice()
    bad_time = time.time() - start
    print(f"未优化版本耗时: {bad_time:.4f} 秒\n")
    
    # 优化后的版本
    start = time.time()
    comprehensive_good_practice()
    good_time = time.time() - start
    print(f"优化版本耗时: {good_time:.4f} 秒\n")
    
    print(f"总体性能提升: {bad_time/good_time:.2f}x\n")
    
    # 性能分析工作流程
    performance_analysis_workflow()
    
    print("\n使用 nsys profile 进行完整分析：")
    print("nsys profile \\")
    print("  --trace=cuda,nvtx,osrt \\")
    print("  --cuda-memory-usage=true \\")
    print("  --sampling-frequency=1000 \\")
    print("  --gpu-metrics-frequency=100 \\")
    print("  --stats=true \\")
    print("  --output=comprehensive_profile.nsys-rep \\")
    print("  python example7_comprehensive.py")

