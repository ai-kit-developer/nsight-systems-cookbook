#!/usr/bin/env python3
"""
批量更新所有reduce可视化文件，添加数组长度控制和样式优化
"""

import os
import re
from pathlib import Path

# 要更新的文件列表
REDUCE_FILES = [
    'reduce_v0_visualization.html',
    'reduce_v1_visualization.html',
    'reduce_v2_visualization.html',
    'reduce_v3_visualization.html',
    'reduce_v4_visualization.html',
    'reduce_v5_visualization.html',
    'reduce_v6_visualization.html',
    'reduce_v7_visualization.html',
]

def update_file(filepath):
    """更新单个文件"""
    if not os.path.exists(filepath):
        print(f"⚠️  文件不存在: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # 1. 添加数组长度控制到controls区域
    if '数组长度' not in content:
        # 查找controls区域，在threadCount之前添加arrayLength
        pattern = r'(<div class="control-group">\s*<label>线程数:</label>)'
        replacement = r'''<div class="control-group">
                    <label>数组长度:</label>
                    <input type="range" id="arrayLength" min="16" max="512" step="16" value="64">
                    <span id="arrayLengthValue">64</span>
                </div>
                <div class="control-group">
                    <label>线程数:</label>'''
        
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append("添加数组长度控制")
    
    # 2. 更新config对象，添加arrayLength
    if 'arrayLength:' not in content:
        pattern = r'(let config = \{[^}]*threadCount: \d+)'
        replacement = r'\1,\n            arrayLength: 64'
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append("添加arrayLength到config")
    
    # 3. 添加globalMemory变量
    if 'let globalMemory' not in content:
        pattern = r'(// 状态\s*let sharedMemory = \[\];)'
        replacement = r'\1\n        let globalMemory = []; // 全局内存数组'
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append("添加globalMemory变量")
    
    # 4. 更新init函数，添加数组长度初始化逻辑
    if 'config.arrayLength = parseInt' not in content:
        pattern = r'(function init\(\) \{[^}]*config\.threadCount = parseInt\(document\.getElementById\(\'threadCount\'\)\.value\);)'
        replacement = r'''function init() {
            config.arrayLength = parseInt(document.getElementById('arrayLength').value);
            config.threadCount = parseInt(document.getElementById('threadCount').value);
            
            // 确保线程数不超过数组长度
            if (config.threadCount > config.arrayLength) {
                config.threadCount = config.arrayLength;
                document.getElementById('threadCount').value = config.threadCount;
            }
            
            // 初始化全局内存数组
            globalMemory = new Array(config.arrayLength).fill(0).map((_, i) => i + 1);
            
            // 初始化共享内存（每个线程块处理一部分数据）
            const elementsPerThread = Math.ceil(config.arrayLength / config.threadCount);
            sharedMemory = new Array(config.threadCount).fill(0);
            
            // 每个线程加载对应的全局内存数据到共享内存
            for (let i = 0; i < config.threadCount; i++) {
                const globalIndex = i * elementsPerThread;
                if (globalIndex < config.arrayLength) {
                    sharedMemory[i] = globalMemory[globalIndex];
                    // 如果线程处理多个元素，累加
                    for (let j = 1; j < elementsPerThread && globalIndex + j < config.arrayLength; j++) {
                        sharedMemory[i] += globalMemory[globalIndex + j];
                    }
                }
            }
            
            config.threadCount = parseInt(document.getElementById('threadCount').value);'''
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append("更新init函数")
    
    # 5. 更新updateDisplays函数，添加arrayLength显示
    if 'arrayLengthValue' not in content:
        pattern = r'(function updateDisplays\(\) \{)'
        replacement = r'''function updateDisplays() {
            document.getElementById('arrayLengthValue').textContent = config.arrayLength;
            document.getElementById('threadCountValue').textContent = config.threadCount;'''
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append("更新updateDisplays函数")
    
    # 6. 添加数组长度事件监听器
    if 'arrayLength.*addEventListener' not in content:
        # 在threadCount事件监听器之前添加
        pattern = r'(// 事件监听\s*document\.getElementById\(\'threadCount\')'
        replacement = r'''// 事件监听
        document.getElementById('arrayLength').addEventListener('input', (e) => {
            config.arrayLength = parseInt(e.target.value);
            document.getElementById('arrayLengthValue').textContent = config.arrayLength;
            // 如果数组长度小于线程数，调整线程数
            if (config.arrayLength < config.threadCount) {
                config.threadCount = config.arrayLength;
                document.getElementById('threadCount').value = config.threadCount;
                document.getElementById('threadCountValue').textContent = config.threadCount;
            }
            // 更新线程数的最大值
            document.getElementById('threadCount').max = config.arrayLength;
            init();
        });

        document.getElementById('threadCount').addEventListener('input', (e) => {
            const newThreadCount = parseInt(e.target.value);
            if (newThreadCount <= config.arrayLength) {
                config.threadCount = newThreadCount;
                document.getElementById('threadCountValue').textContent = config.threadCount;
                init();
            } else {
                // 如果超过数组长度，重置
                e.target.value = config.threadCount;
            }
        });

        document.getElementById('threadCount')'''
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append("添加数组长度事件监听器")
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {os.path.basename(filepath)}: {', '.join(changes)}")
        return True
    else:
        print(f"ℹ️  {os.path.basename(filepath)}: 无需更新（可能已包含更改）")
        return False

def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("🚀 开始批量更新reduce可视化文件...\n")
    
    updated_count = 0
    for filename in REDUCE_FILES:
        filepath = os.path.join(script_dir, filename)
        if update_file(filepath):
            updated_count += 1
    
    print(f"\n✨ 完成！共更新 {updated_count}/{len(REDUCE_FILES)} 个文件")
    print("\n💡 注意：某些复杂的更新可能需要手动检查和调整")

if __name__ == '__main__':
    main()

