#!/usr/bin/env python3
"""
批量修复所有reduce可视化文件的绘制问题：
1. stride标签与线程图示交叉
2. 最后一步不显示
3. 多层显示问题
"""

import os
import re

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
    
    # 检查是否已经修复过（通过检查leftPadding = 90）
    if 'leftPadding = 90' in content or 'leftPadding: 90' in content:
        print(f"ℹ️  {os.path.basename(filepath)}: 可能已包含修复")
        return False
    
    # 1. 修复stepReduce函数，确保最后一步显示
    # 查找并替换终止条件检查
    pattern1 = r'(if \(CONFIG\.reduceLoop === \'backward\'\) \{[^}]*if \(config\.currentStride <= 0\) \{[^}]*return;[^}]*\}[^}]*\} else \{[^}]*if \(config\.currentStride >= config\.threadCount\) \{[^}]*return;[^}]*\}[^}]*\})'
    
    replacement1 = '''// 检查是否已完成（在显示最后一步之后才停止）
            let isFinished = false;
            if (CONFIG.reduceLoop === 'backward') {
                if (config.currentStride <= 0) {
                    isFinished = true;
                }
            } else {
                if (config.currentStride >= config.threadCount) {
                    isFinished = true;
                }
            }

            // 清除之前的状态
            activeThreads.clear();
            computingThreads.clear();
            readingIndices.clear();

            // 如果不是最后一步，确定哪些线程参与计算
            if (!isFinished) {'''
    
    # 由于模式匹配复杂，我们使用更简单的方法
    # 检查是否已经有isFinished变量
    if 'let isFinished = false' not in content:
        # 查找stepReduce函数开始
        pattern = r'(function stepReduce\(\) \{[\s\S]*?)(if \(CONFIG\.reduceLoop === \'backward\'\) \{[\s\S]*?if \(config\.currentStride <= 0\) \{[\s\S]*?return;[\s\S]*?\}[\s\S]*?\} else \{[\s\S]*?if \(config\.currentStride >= config\.threadCount\) \{[\s\S]*?return;[\s\S]*?\}[\s\S]*?\})'
        
        if re.search(pattern, content):
            # 替换为新的逻辑
            replacement = r'''function stepReduce() {
            // 检查是否已完成（在显示最后一步之后才停止）
            let isFinished = false;
            if (CONFIG.reduceLoop === 'backward') {
                if (config.currentStride <= 0) {
                    isFinished = true;
                }
            } else {
                if (config.currentStride >= config.threadCount) {
                    isFinished = true;
                }
            }

            // 清除之前的状态
            activeThreads.clear();
            computingThreads.clear();
            readingIndices.clear();

            // 如果不是最后一步，确定哪些线程参与计算
            if (!isFinished) {'''
            
            # 这里需要更精确的匹配，暂时跳过，手动修复v0作为模板
            pass
    
    print(f"ℹ️  {os.path.basename(filepath)}: 需要手动应用v0的修复")
    return False

def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("🔧 开始修复reduce可视化文件的绘制问题...\n")
    print("⚠️  注意：由于修复逻辑复杂，建议手动将reduce_v0_visualization.html的修复应用到其他文件")
    print("   或者使用reduce_v0作为模板复制修复\n")

if __name__ == '__main__':
    main()

