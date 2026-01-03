#!/usr/bin/env python3
"""
批量修复 v2-v7 可视化文件
主要修复：
1. 添加线程到数据索引的映射
2. 修复树形展示：显示线程而不是数据索引
3. 修复共享内存展示：显示数据并标识线程
4. 修复 stepReduce 和 stepBtn 中的逻辑
"""

import re
import os

def fix_visualization_file(filename):
    """修复单个可视化文件"""
    print(f"\n处理 {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. 添加线程到数据索引的映射变量
    if 'threadToDataIndex' not in content:
        pattern = r'(let activeThreads = new Set\(\);[\s\S]*?let readingIndices = new Set\(\);)'
        replacement = r'''let activeThreads = new Set(); // 存储活跃的线程索引（thread_idx）
        let computingThreads = new Set(); // 存储正在计算的线程索引（thread_idx）
        let readingIndices = new Set(); // 存储正在读取的数据索引（data_index）
        
        // 线程到数据索引的映射（thread_idx -> data_index）
        let threadToDataIndex = new Map(); // Map<thread_idx, data_index>
        // 数据索引到线程的映射（data_index -> thread_idx）
        let dataIndexToThread = new Map(); // Map<data_index, thread_idx>'''
        content = re.sub(pattern, replacement, content)
    
    # 2. 修复 stepReduce 函数中的线程选择逻辑
    # 查找 backward 循环部分
    backward_pattern = r'(if \(CONFIG\.reduceLoop === \'backward\'\) \{[\s\S]*?// 反向循环：只有前 stride 个线程参与[\s\S]*?for \(let i = 0; i < config\.currentStride; i\+\+\) \{[\s\S]*?readingIndices\.add\(i \+ config\.currentStride\);[\s\S]*?\})'
    
    backward_replacement = '''if (CONFIG.reduceLoop === 'backward') {
                // v2 backward: 反向循环，只有前 stride 个线程参与
                // thread_idx 直接作为数据索引
                for (let threadIdx = 0; threadIdx < config.currentStride; threadIdx++) {
                    if (threadIdx + config.currentStride < config.threadCount) {
                        const dataIndex = threadIdx; // v2 backward: thread_idx = data_index
                        activeThreads.add(threadIdx); // 存储线程索引
                        computingThreads.add(threadIdx); // 存储线程索引
                        readingIndices.add(threadIdx + config.currentStride); // 存储数据索引
                        threadToDataIndex.set(threadIdx, dataIndex);
                        dataIndexToThread.set(dataIndex, threadIdx);
                    }
                }'''
    
    content = re.sub(backward_pattern, backward_replacement, content)
    
    # 修复正向循环中的 v1+ 逻辑
    forward_v1_pattern = r'(} else \{[\s\S]*?// v1\+: 使用连续索引[\s\S]*?for \(let i = 0; i < config\.threadCount; i\+\+\) \{[\s\S]*?const index = 2 \* config\.currentStride \* i;[\s\S]*?readingIndices\.add\(index \+ config\.currentStride\);[\s\S]*?\})'
    
    forward_v1_replacement = '''} else {
                    // v1+: 使用连续索引 - 消除分支发散
                    for (let threadIdx = 0; threadIdx < config.threadCount; threadIdx++) {
                        const dataIndex = 2 * config.currentStride * threadIdx;
                        if (dataIndex < config.threadCount && dataIndex + config.currentStride < config.threadCount) {
                            activeThreads.add(threadIdx); // 存储线程索引
                            computingThreads.add(threadIdx); // 存储线程索引
                            readingIndices.add(dataIndex + config.currentStride); // 存储数据索引
                            threadToDataIndex.set(threadIdx, dataIndex);
                            dataIndexToThread.set(dataIndex, threadIdx);
                        }
                    }'''
    
    content = re.sub(forward_v1_pattern, forward_v1_replacement, content)
    
    # 修复执行归约部分
    exec_pattern = r'(// 执行归约（模拟）[\s\S]*?for \(let i of activeThreads\) \{[\s\S]*?sharedMemory\[i\] \+= sharedMemory\[i \+ config\.currentStride\];[\s\S]*?\})'
    exec_replacement = '''// 执行归约（模拟，使用线程索引）
            for (let threadIdx of activeThreads) {
                const dataIndex = threadToDataIndex.get(threadIdx);
                if (dataIndex !== undefined && dataIndex + config.currentStride < config.threadCount) {
                    sharedMemory[dataIndex] += sharedMemory[dataIndex + config.currentStride];
                }
            }'''
    
    content = re.sub(exec_pattern, exec_replacement, content)
    
    # 3. 修复清除状态部分，添加映射清除
    clear_pattern = r'(activeThreads\.clear\(\);[\s\S]*?readingIndices\.clear\(\);)'
    clear_replacement = '''activeThreads.clear();
            computingThreads.clear();
            readingIndices.clear();
            threadToDataIndex.clear();
            dataIndexToThread.clear();'''
    
    content = re.sub(clear_pattern, clear_replacement, content)
    
    # 检查是否有修改
    if content != original_content:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {filename}: 已更新")
        return True
    else:
        print(f"ℹ️  {filename}: 无需更新或已更新")
        return False

# 修复所有 v2-v7 文件
files = [f'reduce_v{i}_visualization.html' for i in range(2, 8)]
fixed_count = 0

for filename in files:
    if os.path.exists(filename):
        if fix_visualization_file(filename):
            fixed_count += 1
    else:
        print(f"⚠️  {filename}: 文件不存在")

print(f"\n✨ 完成！共修复 {fixed_count} 个文件")
print("\n注意：树形展示和共享内存展示的修复需要手动应用，因为涉及复杂的逻辑修改。")

