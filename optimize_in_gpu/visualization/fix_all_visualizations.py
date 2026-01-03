#!/usr/bin/env python3
"""
修复所有可视化文件（v2-v7），从 v1 复制完整的 drawTree 和 drawMemory 实现
"""

import re
import os

# 读取 v1 的完整 drawTree 和 drawMemory 函数
def read_v1_functions():
    """从 v1 读取完整的 drawTree 和 drawMemory 函数"""
    with open('reduce_v1_visualization.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取 drawTree 函数
    draw_tree_match = re.search(
        r'(// 绘制树形归约\s+function drawTree\(\) \{[\s\S]*?^\s+\})',
        content,
        re.MULTILINE
    )
    
    # 提取 drawMemory 函数
    draw_memory_match = re.search(
        r'(// 绘制共享内存状态\s+function drawMemory\(\) \{[\s\S]*?^\s+\})',
        content,
        re.MULTILINE
    )
    
    # 提取 draw 函数
    draw_match = re.search(
        r'(// 绘制所有内容\s+function draw\(\) \{[\s\S]*?^\s+\})',
        content,
        re.MULTILINE
    )
    
    return {
        'drawTree': draw_tree_match.group(1) if draw_tree_match else None,
        'drawMemory': draw_memory_match.group(1) if draw_memory_match else None,
        'draw': draw_match.group(1) if draw_match else None
    }

def fix_visualization_file(filename, functions):
    """修复单个可视化文件"""
    print(f"\n处理 {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 替换 drawTree 函数
    if functions['drawTree']:
        # 查找现有的 drawTree 函数
        draw_tree_pattern = r'// 绘制树形归约\s+function drawTree\(\) \{[\s\S]*?^\s+\}'
        if re.search(draw_tree_pattern, content, re.MULTILINE):
            content = re.sub(
                draw_tree_pattern,
                functions['drawTree'],
                content,
                flags=re.MULTILINE
            )
            print(f"  ✅ 已更新 drawTree")
        else:
            print(f"  ⚠️  未找到 drawTree 函数")
    
    # 替换 drawMemory 函数
    if functions['drawMemory']:
        # 查找现有的 drawMemory 函数
        draw_memory_pattern = r'// 绘制共享内存状态\s+function drawMemory\(\) \{[\s\S]*?^\s+\}'
        if re.search(draw_memory_pattern, content, re.MULTILINE):
            content = re.sub(
                draw_memory_pattern,
                functions['drawMemory'],
                content,
                flags=re.MULTILINE
            )
            print(f"  ✅ 已更新 drawMemory")
        else:
            print(f"  ⚠️  未找到 drawMemory 函数")
    
    # 替换 draw 函数
    if functions['draw']:
        # 查找现有的 draw 函数
        draw_pattern = r'// 绘制所有内容\s+function draw\(\) \{[\s\S]*?^\s+\}'
        if re.search(draw_pattern, content, re.MULTILINE):
            content = re.sub(
                draw_pattern,
                functions['draw'],
                content,
                flags=re.MULTILINE
            )
            print(f"  ✅ 已更新 draw")
        else:
            print(f"  ⚠️  未找到 draw 函数")
    
    # 检查是否有修改
    if content != original_content:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ {filename}: 已更新")
        return True
    else:
        print(f"  ℹ️  {filename}: 无需更新")
        return False

def main():
    # 读取 v1 的函数
    print("从 v1 读取完整的函数实现...")
    functions = read_v1_functions()
    
    if not functions['drawTree']:
        print("❌ 无法读取 drawTree 函数")
        return
    if not functions['drawMemory']:
        print("❌ 无法读取 drawMemory 函数")
        return
    if not functions['draw']:
        print("❌ 无法读取 draw 函数")
        return
    
    print("✅ 成功读取所有函数")
    
    # 修复所有 v2-v7 文件
    files = [f'reduce_v{i}_visualization.html' for i in range(2, 8)]
    fixed_count = 0
    
    for filename in files:
        if os.path.exists(filename):
            if fix_visualization_file(filename, functions):
                fixed_count += 1
        else:
            print(f"⚠️  {filename}: 文件不存在")
    
    print(f"\n✨ 完成！共修复 {fixed_count} 个文件")

if __name__ == '__main__':
    main()

