#!/usr/bin/env python3
"""
修复reduce可视化文件中的重复代码问题
"""

import os
import re
from pathlib import Path

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

def fix_file(filepath):
    """修复单个文件"""
    if not os.path.exists(filepath):
        print(f"⚠️  文件不存在: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # 1. 修复init函数中的重复代码
    # 删除重复的 threadCount 和 sharedMemory 初始化
    pattern = r'(// 每个线程加载对应的全局内存数据到共享内存[\s\S]*?}\s*}\s*)\s*config\.threadCount = parseInt\(document\.getElementById\(\'threadCount\'\)\.value\);\s*sharedMemory = new Array\(config\.threadCount\)\.fill\(0\)\.map\(\(_, i\) => i \+ 1\);'
    
    if re.search(pattern, content):
        content = re.sub(pattern, r'\1', content)
        changes.append("删除重复的threadCount和sharedMemory初始化")
    
    # 2. 修复updateDisplays函数中的重复代码
    # 删除重复的函数体
    pattern = r'(function updateDisplays\(\) \{[\s\S]*?document\.getElementById\(\'completion\'\)\.textContent = completion \+ \'%\';\s*\})\s*document\.getElementById\(\'threadCountValue\'\)\.textContent = config\.threadCount;[\s\S]*?document\.getElementById\(\'completion\'\)\.textContent = completion \+ \'%\';\s*\}'
    
    if re.search(pattern, content):
        content = re.sub(pattern, r'\1', content)
        changes.append("删除重复的updateDisplays函数体")
    
    # 3. 确保updateDisplays函数正确调用arrayLengthValue
    if 'arrayLengthValue' in content and 'document.getElementById(\'arrayLengthValue\')' not in content:
        # 检查updateDisplays函数
        pattern = r'(function updateDisplays\(\) \{[\s\S]*?)(document\.getElementById\(\'threadCountValue\'))'
        replacement = r'\1document.getElementById(\'arrayLengthValue\').textContent = config.arrayLength;\n            \2'
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append("添加arrayLengthValue到updateDisplays")
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {os.path.basename(filepath)}: {', '.join(changes)}")
        return True
    else:
        print(f"ℹ️  {os.path.basename(filepath)}: 无需修复")
        return False

def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("🔧 开始修复reduce可视化文件...\n")
    
    fixed_count = 0
    for filename in REDUCE_FILES:
        filepath = os.path.join(script_dir, filename)
        if fix_file(filepath):
            fixed_count += 1
    
    print(f"\n✨ 完成！共修复 {fixed_count}/{len(REDUCE_FILES)} 个文件")

if __name__ == '__main__':
    main()

