#!/usr/bin/env python3
"""
将reduce_v0的绘制修复应用到所有其他reduce文件（v1-v7）
"""

import os
import re
import shutil

REDUCE_FILES = [
    'reduce_v1_visualization.html',
    'reduce_v2_visualization.html',
    'reduce_v3_visualization.html',
    'reduce_v4_visualization.html',
    'reduce_v5_visualization.html',
    'reduce_v6_visualization.html',
    'reduce_v7_visualization.html',
]

def extract_section(content, start_pattern, end_pattern):
    """提取代码段"""
    start_match = re.search(start_pattern, content)
    end_match = re.search(end_pattern, content, re.MULTILINE)
    if start_match and end_match:
        return content[start_match.start():end_match.end()], start_match.start(), end_match.end()
    return None, None, None

def apply_fixes(filepath, v0_content):
    """应用修复到单个文件"""
    if not os.path.exists(filepath):
        print(f"⚠️  文件不存在: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # 1. 添加 layerStates 变量声明
    if 'let layerStates = []' not in content:
        pattern = r'(let readingIndices = new Set\(\);[\s\S]*?)(// Canvas)'
        replacement = r'\1let layerStates = []; // 存储每层的sharedMemory快照，用于绘制完整的归约过程\n\n        \2'
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append("添加layerStates变量")
    
    # 2. 修改 init 函数，添加 layerStates 初始化
    if 'layerStates = [[...sharedMemory]]' not in content:
        pattern = r'(activeThreads\.clear\(\);[\s\S]*?readingIndices\.clear\(\);[\s\S]*?)(updateDisplays\(\);)'
        replacement = r'\1layerStates = [[...sharedMemory]]; // 保存初始状态\n            \2'
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append("添加layerStates初始化")
    
    # 3. 替换 stepReduce 函数
    # 从v0提取stepReduce函数
    v0_stepReduce, v0_start, v0_end = extract_section(
        v0_content,
        r'// 执行一步归约\s*function stepReduce\(\) \{',
        r'^\s*\}\s*$'
    )
    
    if v0_stepReduce:
        # 查找当前文件的stepReduce函数
        pattern = r'// 执行一步归约\s*function stepReduce\(\) \{[\s\S]*?^\s*\}\s*$'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            # 替换为v0的版本
            content = content[:match.start()] + v0_stepReduce + content[match.end():]
            changes.append("替换stepReduce函数")
    
    # 4. 替换 drawTree 函数
    # 从v0提取drawTree函数
    v0_drawTree, v0_start, v0_end = extract_section(
        v0_content,
        r'// 绘制树形归约\s*function drawTree\(\) \{',
        r'^\s*\}\s*$'
    )
    
    if v0_drawTree:
        # 查找当前文件的drawTree函数
        pattern = r'// 绘制树形归约\s*function drawTree\(\) \{[\s\S]*?^\s*\}\s*$'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            # 替换为v0的版本
            content = content[:match.start()] + v0_drawTree + content[match.end():]
            changes.append("替换drawTree函数")
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {os.path.basename(filepath)}: {', '.join(changes)}")
        return True
    else:
        print(f"ℹ️  {os.path.basename(filepath)}: 无需更新")
        return False

def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 读取v0文件作为模板
    v0_file = 'reduce_v0_visualization.html'
    if not os.path.exists(v0_file):
        print(f"❌ 错误: 找不到模板文件 {v0_file}")
        return
    
    with open(v0_file, 'r', encoding='utf-8') as f:
        v0_content = f.read()
    
    print("🚀 开始应用绘制修复到所有reduce文件...\n")
    
    fixed_count = 0
    for filename in REDUCE_FILES:
        filepath = os.path.join(script_dir, filename)
        if apply_fixes(filepath, v0_content):
            fixed_count += 1
    
    print(f"\n✨ 完成！共修复 {fixed_count}/{len(REDUCE_FILES)} 个文件")
    print("\n💡 注意：如果某些修复未应用，可能需要手动检查")

if __name__ == '__main__':
    main()

