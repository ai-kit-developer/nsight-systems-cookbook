#!/usr/bin/env python3
"""
修复所有可视化文件的 resizeCanvases 函数
"""

import re
import os

def fix_resize_canvases(filename):
    """修复单个文件的 resizeCanvases 函数"""
    print(f"\n处理 {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 查找不完整的 resizeCanvases 函数
    incomplete_pattern = r'(function resizeCanvases\(\) \{[\s\S]*?// 保存当前播放状态[\s\S]*?if \(wasPlaying\) \{[\s\S]*?config\.isPlaying = false;[\s\S]*?\}[\s\S]*?\}\);[\s\S]*?\})'
    
    # 完整的 resizeCanvases 函数
    complete_function = '''function resizeCanvases() {
            const container = document.querySelector('.visualization-container');
            if (!container) return;
            
            const panels = container.querySelectorAll('.viz-panel');
            panels.forEach(panel => {
                const canvas = panel.querySelector('canvas');
                if (canvas) {
                    const rect = panel.getBoundingClientRect();
                    // 使用面板的实际宽度（减去padding）
                    const padding = 30; // 左右padding总和
                    const newWidth = Math.max(600, Math.floor(rect.width - padding));
                    
                    // 保存当前播放状态
                    const wasPlaying = config.isPlaying;
                    if (wasPlaying) {
                        config.isPlaying = false;
                    }
                    
                    // 设置新宽度
                    canvas.width = newWidth;
                    
                    // 对于treeCanvas，根据层数动态调整高度
                    if (canvas.id === 'treeCanvas') {
                        adjustTreeCanvasHeight();
                    } else {
                        // 对于memoryCanvas，使用面板高度（减去标题、padding、legend等）
                        const availableHeight = rect.height - 100; // 减去标题、padding、legend等
                        const newHeight = Math.max(400, Math.floor(availableHeight));
                        canvas.height = newHeight;
                    }
                    
                    // 重新绘制
                    if (wasPlaying) {
                        config.isPlaying = true;
                    }
                    draw();
                }
            });
        }'''
    
    # 检查是否有不完整的函数
    if re.search(r'function resizeCanvases\(\) \{[\s\S]*?\}\);[\s\S]*?\}\s+window\.addEventListener', content, re.MULTILINE):
        # 替换不完整的函数
        content = re.sub(
            r'function resizeCanvases\(\) \{[\s\S]*?\}\);[\s\S]*?\}',
            complete_function,
            content,
            flags=re.MULTILINE
        )
        
        if content != original_content:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ {filename}: 已修复 resizeCanvases")
            return True
        else:
            print(f"  ℹ️  {filename}: 无需修复")
            return False
    else:
        print(f"  ℹ️  {filename}: resizeCanvases 函数可能已完整或不存在")
        return False

# 修复所有文件
files = [f'reduce_v{i}_visualization.html' for i in range(0, 8)]
fixed_count = 0

for filename in files:
    if os.path.exists(filename):
        if fix_resize_canvases(filename):
            fixed_count += 1
    else:
        print(f"⚠️  {filename}: 文件不存在")

print(f"\n✨ 完成！共修复 {fixed_count} 个文件")

