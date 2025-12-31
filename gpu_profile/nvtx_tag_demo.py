"""
NVTX (NVIDIA Tools Extension) 完整示例
展示所有 NVTX 标记用法，包括多层嵌套

本示例包含实际的 GPU 计算，可用于 CUDA 性能分析。
支持 PyTorch 和 CuPy 两种方式。
"""

import nvtx
import numpy as np
import time

# 尝试导入 GPU 计算库
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        torch.cuda.init()
        DEVICE = torch.device('cuda')
        print(f"✓ PyTorch 可用，使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        TORCH_AVAILABLE = False
        print("⚠ PyTorch 可用但 CUDA 不可用，将使用 CPU")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch 不可用")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print(f"✓ CuPy 可用")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠ CuPy 不可用")

# 选择可用的 GPU 库
USE_TORCH = TORCH_AVAILABLE
USE_CUPY = CUPY_AVAILABLE and not TORCH_AVAILABLE  # 优先使用 PyTorch

if not (USE_TORCH or USE_CUPY):
    print("⚠ 警告：没有可用的 GPU 库，将使用 CPU 模拟（性能分析可能不准确）")

# 颜色映射：将非标准颜色名称映射为 RGB 元组 (R, G, B)
# 注意：nvtx 要求 RGB 值在 0-1 范围内，所以需要归一化
COLOR_RGB = {
    "spring_green": (0.0, 1.0, 0.498),
    "lime": (0.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "sky_blue": (0.529, 0.808, 0.922),
    "violet": (0.933, 0.510, 0.933),
    "magenta": (1.0, 0.0, 1.0),
    "peach": (1.0, 0.855, 0.725),
    "light_green": (0.565, 0.933, 0.565),
    "lavender": (0.902, 0.902, 0.980),
    "light_red": (1.0, 0.714, 0.757),
    "coral": (1.0, 0.498, 0.314),
    "light_blue": (0.678, 0.847, 0.902),
    "steel_blue": (0.275, 0.510, 0.706),
    "lime_green": (0.196, 0.804, 0.196),
    "tan": (0.824, 0.706, 0.549),
    "khaki": (0.941, 0.902, 0.549),
    "pale_green": (0.596, 0.984, 0.596),
    "teal": (0.0, 0.502, 0.502),
    "mint": (0.741, 0.988, 0.788),
}

def get_color(color_name):
    """获取颜色值，非标准颜色返回 RGB 元组，标准颜色返回字符串"""
    if color_name in COLOR_RGB:
        return COLOR_RGB[color_name]
    return color_name


def gpu_matrix_multiply(size=1024):
    """GPU 矩阵乘法计算"""
    if USE_TORCH:
        with nvtx.annotate("GPU: 创建矩阵", color="blue"):
            a = torch.randn(size, size, device=DEVICE)
            b = torch.randn(size, size, device=DEVICE)
        with nvtx.annotate("GPU: 矩阵乘法", color="green"):
            c = torch.matmul(a, b)
        with nvtx.annotate("GPU: 同步", color="yellow"):
            torch.cuda.synchronize()
        return c
    elif USE_CUPY:
        with nvtx.annotate("GPU: 创建矩阵", color="blue"):
            a = cp.random.randn(size, size, dtype=cp.float32)
            b = cp.random.randn(size, size, dtype=cp.float32)
        with nvtx.annotate("GPU: 矩阵乘法", color="green"):
            c = cp.matmul(a, b)
        with nvtx.annotate("GPU: 同步", color="yellow"):
            cp.cuda.Stream.null.synchronize()
        return c
    else:
        # CPU fallback
        time.sleep(0.01)
        return None

def demo_basic_range():
    """基本范围标记：使用 push/pop"""
    print("=== 基本范围标记 ===")
    
    # 使用 push/pop 标记代码范围
    nvtx.push_range("基本范围标记")
    # 执行 GPU 计算
    gpu_matrix_multiply(512)
    nvtx.pop_range()
    
    print("基本范围标记完成\n")


def demo_range_with_label():
    """带标签的范围标记"""
    print("=== 带标签的范围标记 ===")
    
    # 带标签的范围（将标签合并到消息中）
    nvtx.push_range("数据处理: 数据预处理阶段")
    # GPU 数据预处理
    if USE_TORCH:
        with nvtx.annotate("GPU: 数据归一化", color="cyan"):
            data = torch.randn(1000, 1000, device=DEVICE)
            data = (data - data.mean()) / data.std()
            torch.cuda.synchronize()
    elif USE_CUPY:
        with nvtx.annotate("GPU: 数据归一化", color="cyan"):
            data = cp.random.randn(1000, 1000, dtype=cp.float32)
            data = (data - data.mean()) / data.std()
            cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.05)
    nvtx.pop_range()
    
    print("带标签的范围标记完成\n")


def demo_range_with_color():
    """带颜色的范围标记"""
    print("=== 带颜色的范围标记 ===")
    
    # 使用颜色字符串（nvtx 支持直接使用颜色名称）
    nvtx.push_range("红色范围", color="red")
    gpu_matrix_multiply(256)
    nvtx.pop_range()
    
    nvtx.push_range("绿色范围", color="green")
    gpu_matrix_multiply(256)
    nvtx.pop_range()
    
    nvtx.push_range("蓝色范围", color="blue")
    gpu_matrix_multiply(256)
    nvtx.pop_range()
    
    print("带颜色的范围标记完成\n")


def demo_mark():
    """标记特定点"""
    print("=== 标记特定点 ===")
    
    # 在代码中标记特定点
    nvtx.mark("开始处理")
    gpu_matrix_multiply(512)
    
    nvtx.mark("中间检查点", color="yellow")
    if USE_TORCH:
        with nvtx.annotate("GPU: 检查点计算", color="orange"):
            x = torch.randn(512, 512, device=DEVICE)
            result = torch.sum(x ** 2)
            torch.cuda.synchronize()
    elif USE_CUPY:
        with nvtx.annotate("GPU: 检查点计算", color="orange"):
            x = cp.random.randn(512, 512, dtype=cp.float32)
            result = cp.sum(x ** 2)
            cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.05)
    
    nvtx.mark("处理完成")
    
    print("标记特定点完成\n")


def demo_context_manager():
    """使用上下文管理器"""
    print("=== 上下文管理器 ===")
    
    # 使用 with 语句自动管理范围（nvtx.annotate 是上下文管理器）
    with nvtx.annotate("上下文管理器范围", color="purple"):
        # GPU 计算
        gpu_matrix_multiply(1024)
        print("在上下文管理器范围内")
    
    print("上下文管理器完成\n")


def demo_domain():
    """使用域（Domain）组织标记"""
    print("=== 使用域 ===")
    
    # 创建不同的域
    data_domain = nvtx.Domain("数据处理域")
    compute_domain = nvtx.Domain("计算域")
    
    # 在特定域中标记（需要使用 get_event_attributes 创建 EventAttributes）
    attrs1 = data_domain.get_event_attributes(message="数据加载")
    handle1 = data_domain.start_range(attrs1)
    # GPU 数据加载
    if USE_TORCH:
        with nvtx.annotate("GPU: 从CPU传输数据", color="yellow"):
            cpu_data = torch.randn(1000, 1000)
            gpu_data = cpu_data.to(DEVICE)
            torch.cuda.synchronize()
    elif USE_CUPY:
        with nvtx.annotate("GPU: 从CPU传输数据", color="yellow"):
            cpu_data = np.random.randn(1000, 1000).astype(np.float32)
            gpu_data = cp.asarray(cpu_data)
            cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.05)
    data_domain.end_range(handle1)
    
    attrs2 = compute_domain.get_event_attributes(message="矩阵计算")
    handle2 = compute_domain.start_range(attrs2)
    # GPU 矩阵计算
    gpu_matrix_multiply(512)
    compute_domain.end_range(handle2)
    
    print("域使用完成\n")


def demo_nested_ranges():
    """嵌套范围标记 - 两层"""
    print("=== 嵌套范围标记（两层）===")
    
    nvtx.push_range("外层范围", color="orange")
    # 外层 GPU 操作
    if USE_TORCH:
        x = torch.randn(256, 256, device=DEVICE)
        torch.cuda.synchronize()
    elif USE_CUPY:
        x = cp.random.randn(256, 256, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.02)
    
    nvtx.push_range("内层范围1", color=get_color("spring_green"))
    gpu_matrix_multiply(256)
    nvtx.pop_range()
    
    nvtx.push_range("内层范围2", color=get_color("violet"))
    gpu_matrix_multiply(256)
    nvtx.pop_range()
    
    nvtx.pop_range()
    
    print("两层嵌套完成\n")


def demo_deeply_nested_ranges():
    """深层嵌套范围标记 - 多层"""
    print("=== 深层嵌套范围标记（多层）===")
    
    # 第一层
    nvtx.push_range("第1层：主流程", color="red")
    if USE_TORCH:
        main_data = torch.randn(512, 512, device=DEVICE)
        torch.cuda.synchronize()
    elif USE_CUPY:
        main_data = cp.random.randn(512, 512, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.01)
    
    # 第二层
    nvtx.push_range("第2层：初始化阶段", color="orange")
    if USE_TORCH:
        init_data = torch.randn(256, 256, device=DEVICE)
        torch.cuda.synchronize()
    elif USE_CUPY:
        init_data = cp.random.randn(256, 256, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.01)
    
    # 第三层
    nvtx.push_range("第3层：内存分配", color="yellow")
    gpu_matrix_multiply(128)
    nvtx.pop_range()
    
    nvtx.push_range("第3层：参数设置", color=get_color("lime"))
    gpu_matrix_multiply(128)
    nvtx.pop_range()
    
    nvtx.pop_range()  # 结束第2层
    
    # 第二层 - 另一个分支
    nvtx.push_range("第2层：计算阶段", color=get_color("spring_green"))
    if USE_TORCH:
        compute_data = torch.randn(256, 256, device=DEVICE)
        torch.cuda.synchronize()
    elif USE_CUPY:
        compute_data = cp.random.randn(256, 256, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.01)
    
    # 第三层
    nvtx.push_range("第3层：数据预处理", color=get_color("cyan"))
    gpu_matrix_multiply(128)
    
    # 第四层
    nvtx.push_range("第4层：归一化", color=get_color("sky_blue"))
    if USE_TORCH:
        with nvtx.annotate("GPU: 归一化计算", color="blue"):
            normalized = (compute_data - compute_data.mean()) / compute_data.std()
            torch.cuda.synchronize()
    elif USE_CUPY:
        with nvtx.annotate("GPU: 归一化计算", color="blue"):
            normalized = (compute_data - compute_data.mean()) / compute_data.std()
            cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.02)
    nvtx.pop_range()
    
    nvtx.push_range("第4层：特征提取", color=get_color("violet"))
    gpu_matrix_multiply(128)
    nvtx.pop_range()
    
    nvtx.pop_range()  # 结束第3层
    
    nvtx.push_range("第3层：核心计算", color=get_color("magenta"))
    gpu_matrix_multiply(256)
    nvtx.pop_range()
    
    nvtx.pop_range()  # 结束第2层
    
    # 第二层 - 最后一个分支
    nvtx.push_range("第2层：后处理阶段", color=get_color("peach"))
    if USE_TORCH:
        post_data = torch.randn(256, 256, device=DEVICE)
        torch.cuda.synchronize()
    elif USE_CUPY:
        post_data = cp.random.randn(256, 256, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.01)
    
    nvtx.push_range("第3层：结果验证", color=get_color("light_green"))
    if USE_TORCH:
        with nvtx.annotate("GPU: 验证计算", color="green"):
            result = torch.sum(post_data ** 2)
            torch.cuda.synchronize()
    elif USE_CUPY:
        with nvtx.annotate("GPU: 验证计算", color="green"):
            result = cp.sum(post_data ** 2)
            cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.02)
    nvtx.pop_range()
    
    nvtx.push_range("第3层：输出格式化", color=get_color("lavender"))
    gpu_matrix_multiply(128)
    nvtx.pop_range()
    
    nvtx.pop_range()  # 结束第2层
    nvtx.pop_range()  # 结束第1层
    
    print("多层嵌套完成\n")


def demo_mixed_nested_with_domains():
    """混合使用：域 + 嵌套范围"""
    print("=== 混合使用：域 + 嵌套范围 ===")
    
    # 创建域
    io_domain = nvtx.Domain("IO域")
    compute_domain = nvtx.Domain("计算域")
    
    # IO 域中的嵌套
    attrs1 = io_domain.get_event_attributes(message="IO操作", color=get_color("light_red"))
    handle1 = io_domain.start_range(attrs1)
    # GPU 数据传输（模拟 IO）
    if USE_TORCH:
        with nvtx.annotate("GPU: CPU->GPU 传输", color="yellow"):
            cpu_data = torch.randn(1000, 1000)
            gpu_data = cpu_data.to(DEVICE)
            torch.cuda.synchronize()
    elif USE_CUPY:
        with nvtx.annotate("GPU: CPU->GPU 传输", color="yellow"):
            cpu_data = np.random.randn(1000, 1000).astype(np.float32)
            gpu_data = cp.asarray(cpu_data)
            cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.01)
    
    attrs2 = io_domain.get_event_attributes(message="读取文件", color=get_color("coral"))
    handle2 = io_domain.start_range(attrs2)
    gpu_matrix_multiply(256)
    io_domain.end_range(handle2)
    
    attrs3 = io_domain.get_event_attributes(message="解析数据", color=get_color("light_red"))
    handle3 = io_domain.start_range(attrs3)
    if USE_TORCH:
        with nvtx.annotate("GPU: 数据解析", color="cyan"):
            parsed = gpu_data.reshape(-1, 100)
            torch.cuda.synchronize()
    elif USE_CUPY:
        with nvtx.annotate("GPU: 数据解析", color="cyan"):
            parsed = gpu_data.reshape(-1, 100)
            cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.02)
    io_domain.end_range(handle3)
    
    io_domain.end_range(handle1)
    
    # 计算域中的嵌套
    attrs4 = compute_domain.get_event_attributes(message="计算操作", color=get_color("light_blue"))
    handle4 = compute_domain.start_range(attrs4)
    if USE_TORCH:
        compute_input = torch.randn(512, 512, device=DEVICE)
        torch.cuda.synchronize()
    elif USE_CUPY:
        compute_input = cp.random.randn(512, 512, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.01)
    
    attrs5 = compute_domain.get_event_attributes(message="矩阵运算", color=get_color("light_blue"))
    handle5 = compute_domain.start_range(attrs5)
    if USE_TORCH:
        matrix_a = torch.randn(256, 256, device=DEVICE)
        torch.cuda.synchronize()
    elif USE_CUPY:
        matrix_a = cp.random.randn(256, 256, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.01)
    
    attrs6 = compute_domain.get_event_attributes(message="矩阵乘法", color=get_color("light_blue"))
    handle6 = compute_domain.start_range(attrs6)
    gpu_matrix_multiply(256)
    compute_domain.end_range(handle6)
    
    attrs7 = compute_domain.get_event_attributes(message="矩阵转置", color=get_color("light_blue"))
    handle7 = compute_domain.start_range(attrs7)
    if USE_TORCH:
        with nvtx.annotate("GPU: 矩阵转置", color="blue"):
            transposed = matrix_a.t()
            torch.cuda.synchronize()
    elif USE_CUPY:
        with nvtx.annotate("GPU: 矩阵转置", color="blue"):
            transposed = matrix_a.T
            cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.02)
    compute_domain.end_range(handle7)
    
    compute_domain.end_range(handle5)
    
    attrs8 = compute_domain.get_event_attributes(message="向量运算", color=get_color("light_blue"))
    handle8 = compute_domain.start_range(attrs8)
    if USE_TORCH:
        with nvtx.annotate("GPU: 向量运算", color="green"):
            vec = torch.randn(1000, device=DEVICE)
            result = torch.sum(vec ** 2)
            torch.cuda.synchronize()
    elif USE_CUPY:
        with nvtx.annotate("GPU: 向量运算", color="green"):
            vec = cp.random.randn(1000, dtype=cp.float32)
            result = cp.sum(vec ** 2)
            cp.cuda.Stream.null.synchronize()
    else:
        time.sleep(0.02)
    compute_domain.end_range(handle8)
    
    compute_domain.end_range(handle4)
    
    print("混合使用完成\n")


def demo_real_world_example():
    """真实世界示例：模拟深度学习训练流程"""
    print("=== 真实世界示例：深度学习训练流程 ===")
    
    # 主训练循环
    nvtx.push_range("训练循环", color=get_color("steel_blue"))
    
    # 初始化模型参数（模拟）
    if USE_TORCH:
        with nvtx.annotate("GPU: 初始化模型", color="blue"):
            weight1 = torch.randn(64, 32, device=DEVICE, requires_grad=True)
            weight2 = torch.randn(32, 10, device=DEVICE, requires_grad=True)
            torch.cuda.synchronize()
    elif USE_CUPY:
        with nvtx.annotate("GPU: 初始化模型", color="blue"):
            weight1 = cp.random.randn(64, 32, dtype=cp.float32)
            weight2 = cp.random.randn(32, 10, dtype=cp.float32)
            cp.cuda.Stream.null.synchronize()
    
    for epoch in range(2):  # 只运行2个epoch作为示例
        nvtx.push_range(f"Epoch {epoch+1}", color=get_color("lime_green"))
        
        # 训练阶段
        nvtx.push_range("训练阶段", color=get_color("tan"))
        
        for batch in range(3):  # 只运行3个batch作为示例
            nvtx.push_range(f"Batch {batch+1}", color=get_color("khaki"))
            
            # 前向传播
            nvtx.push_range("前向传播", color=get_color("light_red"))
            if USE_TORCH:
                with nvtx.annotate("GPU: 输入数据", color="yellow"):
                    x = torch.randn(32, 64, device=DEVICE)
                    torch.cuda.synchronize()
                
                nvtx.push_range("卷积层1", color=get_color("coral"))
                with nvtx.annotate("GPU: Conv1", color="green"):
                    h1 = torch.matmul(x, weight1)
                    h1 = torch.relu(h1)
                    torch.cuda.synchronize()
                nvtx.pop_range()
                
                nvtx.push_range("卷积层2", color=get_color("coral"))
                with nvtx.annotate("GPU: Conv2", color="green"):
                    h2 = torch.matmul(h1, weight2)
                    torch.cuda.synchronize()
                nvtx.pop_range()
                
                nvtx.push_range("全连接层", color=get_color("coral"))
                with nvtx.annotate("GPU: FC", color="green"):
                    output = torch.softmax(h2, dim=1)
                    torch.cuda.synchronize()
                nvtx.pop_range()
            elif USE_CUPY:
                with nvtx.annotate("GPU: 输入数据", color="yellow"):
                    x = cp.random.randn(32, 64, dtype=cp.float32)
                    cp.cuda.Stream.null.synchronize()
                
                nvtx.push_range("卷积层1", color=get_color("coral"))
                with nvtx.annotate("GPU: Conv1", color="green"):
                    h1 = cp.matmul(x, weight1)
                    h1 = cp.maximum(h1, 0)  # ReLU
                    cp.cuda.Stream.null.synchronize()
                nvtx.pop_range()
                
                nvtx.push_range("卷积层2", color=get_color("coral"))
                with nvtx.annotate("GPU: Conv2", color="green"):
                    h2 = cp.matmul(h1, weight2)
                    cp.cuda.Stream.null.synchronize()
                nvtx.pop_range()
                
                nvtx.push_range("全连接层", color=get_color("coral"))
                with nvtx.annotate("GPU: FC", color="green"):
                    # 简化的 softmax
                    exp_h2 = cp.exp(h2 - cp.max(h2, axis=1, keepdims=True))
                    output = exp_h2 / cp.sum(exp_h2, axis=1, keepdims=True)
                    cp.cuda.Stream.null.synchronize()
                nvtx.pop_range()
            else:
                time.sleep(0.01)
                time.sleep(0.005)  # Conv1
                time.sleep(0.005)  # Conv2
                time.sleep(0.005)  # FC
            
            nvtx.pop_range()
            
            # 损失计算
            nvtx.push_range("损失计算", color=get_color("pale_green"))
            if USE_TORCH:
                with nvtx.annotate("GPU: 计算损失", color="green"):
                    target = torch.randint(0, 10, (32,), device=DEVICE)
                    loss = torch.nn.functional.cross_entropy(h2, target)
                    torch.cuda.synchronize()
            elif USE_CUPY:
                with nvtx.annotate("GPU: 计算损失", color="green"):
                    # 简化的损失计算
                    loss = cp.sum((h2 - cp.random.randn(32, 10, dtype=cp.float32)) ** 2)
                    cp.cuda.Stream.null.synchronize()
            else:
                time.sleep(0.005)
            nvtx.pop_range()
            
            # 反向传播
            nvtx.push_range("反向传播", color=get_color("light_blue"))
            if USE_TORCH:
                with nvtx.annotate("GPU: 反向传播", color="purple"):
                    loss.backward()
                    torch.cuda.synchronize()
                
                nvtx.push_range("梯度计算", color=get_color("light_blue"))
                with nvtx.annotate("GPU: 梯度", color="cyan"):
                    grad1 = weight1.grad
                    grad2 = weight2.grad
                    torch.cuda.synchronize()
                nvtx.pop_range()
                
                nvtx.push_range("参数更新", color=get_color("light_blue"))
                with nvtx.annotate("GPU: 更新参数", color="orange"):
                    # 简单的 SGD 更新
                    with torch.no_grad():
                        weight1 -= 0.01 * grad1
                        weight2 -= 0.01 * grad2
                    torch.cuda.synchronize()
                nvtx.pop_range()
            elif USE_CUPY:
                with nvtx.annotate("GPU: 反向传播", color="purple"):
                    # 简化的梯度计算
                    grad1 = cp.random.randn(64, 32, dtype=cp.float32)
                    grad2 = cp.random.randn(32, 10, dtype=cp.float32)
                    cp.cuda.Stream.null.synchronize()
                
                nvtx.push_range("梯度计算", color=get_color("light_blue"))
                cp.cuda.Stream.null.synchronize()
                nvtx.pop_range()
                
                nvtx.push_range("参数更新", color=get_color("light_blue"))
                with nvtx.annotate("GPU: 更新参数", color="orange"):
                    weight1 -= 0.01 * grad1
                    weight2 -= 0.01 * grad2
                    cp.cuda.Stream.null.synchronize()
                nvtx.pop_range()
            else:
                time.sleep(0.01)
                time.sleep(0.005)  # 梯度
                time.sleep(0.005)  # 更新
            
            nvtx.pop_range()
            
            nvtx.pop_range()  # Batch结束
        
        nvtx.pop_range()  # 训练阶段结束
        
        # 验证阶段
        nvtx.push_range("验证阶段", color=get_color("teal"))
        if USE_TORCH:
            with nvtx.annotate("GPU: 验证计算", color="teal"):
                val_input = torch.randn(16, 64, device=DEVICE)
                val_output = torch.matmul(torch.matmul(val_input, weight1), weight2)
                torch.cuda.synchronize()
        elif USE_CUPY:
            with nvtx.annotate("GPU: 验证计算", color="teal"):
                val_input = cp.random.randn(16, 64, dtype=cp.float32)
                val_output = cp.matmul(cp.matmul(val_input, weight1), weight2)
                cp.cuda.Stream.null.synchronize()
        else:
            time.sleep(0.02)
        nvtx.pop_range()
        
        nvtx.pop_range()  # Epoch结束
    
    nvtx.pop_range()  # 训练循环结束
    
    print("真实世界示例完成\n")


def demo_start_end_range():
    """使用 start_range/end_range API"""
    print("=== start_range/end_range API ===")
    
    # 使用 start_range 返回一个句柄
    handle1 = nvtx.start_range("范围1", color=get_color("light_red"))
    gpu_matrix_multiply(512)
    nvtx.end_range(handle1)
    
    handle2 = nvtx.start_range("范围2", color=get_color("mint"))
    gpu_matrix_multiply(512)
    nvtx.end_range(handle2)
    
    print("start_range/end_range 完成\n")


def main():
    """主函数：运行所有示例"""
    print("=" * 50)
    print("NVTX 完整示例演示")
    print("=" * 50)
    print()
    
    # 基本用法
    demo_basic_range()
    demo_range_with_label()
    demo_range_with_color()
    demo_mark()
    demo_context_manager()
    
    # 域的使用
    demo_domain()
    
    # 嵌套示例
    demo_nested_ranges()
    demo_deeply_nested_ranges()
    demo_mixed_nested_with_domains()
    
    # 其他API
    demo_start_end_range()
    
    # 真实世界示例
    demo_real_world_example()
    
    print("=" * 50)
    print("所有示例完成！")
    print("=" * 50)
    print("\n提示：使用 NVIDIA Nsight Systems 或 Nsight Compute 查看 NVTX 标记")


if __name__ == "__main__":
    main()

