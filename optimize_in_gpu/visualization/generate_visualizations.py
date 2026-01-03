#!/usr/bin/env python3
"""
生成所有 Reduce 版本的可视化页面
"""

import os

# 版本配置
VERSIONS = {
    'v0': {
        'title': 'Reduce v0 Baseline',
        'description': '基础版本的归约实现，使用树形归约算法',
        'optimization': 'Baseline',
        'problems': [
            '分支发散：if (thread_idx % (2*stride) == 0) 导致同一 warp 内的线程执行不同路径',
            'Bank conflict：访问模式可能导致共享内存 bank 冲突',
            '线程利用率低：每次迭代只有部分线程参与计算'
        ],
        'features': [],
        'reduce_loop': 'forward',  # forward: stride从1开始, backward: stride从blockDim.x/2开始
        'index_calc': 'mod',  # mod: 使用模运算, continuous: 使用连续索引
        'load_optimization': False,  # 是否在加载时进行加法
        'unroll_warp': False,  # 是否展开最后一个warp
        'unroll_complete': False,  # 是否完全展开循环
        'multi_element': False,  # 是否每个线程处理多个元素
        'use_shuffle': False  # 是否使用shuffle指令
    },
    'v1': {
        'title': 'Reduce v1 - 消除分支发散',
        'description': '使用连续索引计算代替模运算，消除分支发散问题',
        'optimization': '消除分支发散',
        'problems': [
            '仍然存在 bank conflict：访问模式仍可能导致冲突',
            '线程利用率仍然不高：每次迭代只有部分线程参与计算'
        ],
        'features': [
            '消除分支发散：使用 index = 2*stride*thread_idx',
            '前几轮迭代中，整个 warp 要么都执行，要么都不执行'
        ],
        'reduce_loop': 'forward',
        'index_calc': 'continuous',
        'load_optimization': False,
        'unroll_warp': False,
        'unroll_complete': False,
        'multi_element': False,
        'use_shuffle': False
    },
    'v2': {
        'title': 'Reduce v2 - 消除 Bank 冲突',
        'description': '使用反向循环，改变访问模式，消除 bank 冲突',
        'optimization': '消除 Bank 冲突',
        'problems': [
            '仍有分支发散：if (thread_idx < stride)',
            '线程利用率：每次迭代后一半线程空闲'
        ],
        'features': [
            '反向循环：从 stride = blockDim.x/2 开始',
            '消除 bank 冲突：相邻线程访问相邻内存位置',
            '提高共享内存带宽利用率'
        ],
        'reduce_loop': 'backward',
        'index_calc': 'continuous',
        'load_optimization': False,
        'unroll_warp': False,
        'unroll_complete': False,
        'multi_element': False,
        'use_shuffle': False
    },
    'v3': {
        'title': 'Reduce v3 - 加载时加法',
        'description': '每个线程加载两个元素并在加载时立即相加，提高线程利用率',
        'optimization': '加载时加法',
        'problems': [],
        'features': [
            '加载时加法：减少全局内存访问次数',
            '提高线程利用率：充分利用所有线程',
            '减少线程块数量：每个 block 处理更多数据'
        ],
        'reduce_loop': 'backward',
        'index_calc': 'continuous',
        'load_optimization': True,
        'unroll_warp': False,
        'unroll_complete': False,
        'multi_element': False,
        'use_shuffle': False
    },
    'v4': {
        'title': 'Reduce v4 - 展开最后一个 Warp',
        'description': '展开最后一个 warp 的归约操作，减少同步开销',
        'optimization': '展开最后一个 Warp',
        'problems': [],
        'features': [
            '展开最后一个 warp：消除循环开销',
            '减少同步操作：warp 内隐式同步',
            '使用 volatile 防止编译器优化'
        ],
        'reduce_loop': 'backward',
        'index_calc': 'continuous',
        'load_optimization': True,
        'unroll_warp': True,
        'unroll_complete': False,
        'multi_element': False,
        'use_shuffle': False
    },
    'v5': {
        'title': 'Reduce v5 - 完全展开循环',
        'description': '完全展开归约循环，使用模板参数在编译时优化',
        'optimization': '完全展开循环',
        'problems': [],
        'features': [
            '完全展开循环：消除循环开销',
            '条件编译：根据 block_size 生成代码',
            '模板化实现：编译时优化'
        ],
        'reduce_loop': 'backward',
        'index_calc': 'continuous',
        'load_optimization': True,
        'unroll_warp': True,
        'unroll_complete': True,
        'multi_element': False,
        'use_shuffle': False
    },
    'v6': {
        'title': 'Reduce v6 - 多元素处理',
        'description': '每个线程处理多个元素，提高 GPU 占用率',
        'optimization': '多元素处理',
        'problems': [],
        'features': [
            '每个线程处理多个元素',
            '循环展开：使用 #pragma unroll',
            '提高 GPU 占用率',
            '减少全局内存访问延迟影响'
        ],
        'reduce_loop': 'backward',
        'index_calc': 'continuous',
        'load_optimization': True,
        'unroll_warp': True,
        'unroll_complete': True,
        'multi_element': True,
        'use_shuffle': False
    },
    'v7': {
        'title': 'Reduce v7 - Shuffle 指令',
        'description': '使用 Shuffle 指令进行 warp 内归约，达到极致性能',
        'optimization': 'Shuffle 指令',
        'problems': [],
        'features': [
            '使用 __shfl_down_sync 指令',
            '寄存器间直接通信：延迟更低',
            '减少共享内存使用',
            '两阶段归约：warp 内 + warp 间'
        ],
        'reduce_loop': 'backward',
        'index_calc': 'continuous',
        'load_optimization': True,
        'unroll_warp': True,
        'unroll_complete': True,
        'multi_element': True,
        'use_shuffle': True
    }
}

def generate_html(version_key, config):
    """生成单个版本的可视化HTML页面"""
    
    html_template = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CUDA 归约算法可视化 - {config['title']}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .back-link {{
            display: inline-block;
            margin-top: 15px;
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            background: rgba(255,255,255,0.2);
            border-radius: 5px;
            transition: background 0.3s;
        }}

        .back-link:hover {{
            background: rgba(255,255,255,0.3);
        }}

        .content {{
            padding: 30px;
        }}

        .info-panel {{
            background: #e8f4f8;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }}

        .info-panel h3 {{
            color: #667eea;
            margin-bottom: 10px;
        }}

        .info-panel p {{
            line-height: 1.6;
            color: #555;
        }}

        .features-panel {{
            background: #e8f5e9;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 4px solid #4CAF50;
        }}

        .features-panel h3 {{
            color: #4CAF50;
            margin-bottom: 10px;
        }}

        .features-panel ul {{
            margin-left: 20px;
            color: #555;
        }}

        .features-panel li {{
            margin: 5px 0;
        }}

        .controls {{
            background: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .control-group label {{
            font-weight: 600;
            color: #555;
        }}

        .control-group input, .control-group select {{
            padding: 8px 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }}

        .control-group input[type="range"] {{
            width: 200px;
        }}

        button {{
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}

        button:active {{
            transform: translateY(0);
        }}

        button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }}

        .visualization-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}

        @media (max-width: 1200px) {{
            .visualization-container {{
                grid-template-columns: 1fr;
            }}
        }}

        .viz-panel {{
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
        }}

        .viz-panel h3 {{
            color: #667eea;
            margin-bottom: 15px;
            text-align: center;
        }}

        canvas {{
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: #fafafa;
        }}

        .legend {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 15px;
            flex-wrap: wrap;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            border: 1px solid #ccc;
        }}

        .problem-highlight {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }}

        .problem-highlight h4 {{
            color: #856404;
            margin-bottom: 8px;
        }}

        .problem-highlight ul {{
            margin-left: 20px;
            color: #856404;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}

        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}

        .stat-card .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 CUDA 归约算法可视化</h1>
            <p>{config['title']}</p>
            <a href="index.html" class="back-link">← 返回主页</a>
        </div>

        <div class="content">
            <div class="info-panel">
                <h3>📖 算法说明</h3>
                <p>{config['description']}</p>
            </div>

            {f'<div class="features-panel"><h3>✨ 优化特性</h3><ul>' + ''.join([f'<li>{f}</li>' for f in config['features']]) + '</ul></div>' if config['features'] else ''}

            {f'<div class="problem-highlight"><h4>⚠️ 已知问题</h4><ul>' + ''.join([f'<li>{p}</li>' for p in config['problems']]) + '</ul></div>' if config['problems'] else ''}

            <div class="controls">
                <div class="control-group">
                    <label>线程数:</label>
                    <input type="range" id="threadCount" min="8" max="256" step="8" value="32">
                    <span id="threadCountValue">32</span>
                </div>
                <div class="control-group">
                    <label>速度:</label>
                    <input type="range" id="speed" min="1" max="10" value="5">
                    <span id="speedValue">5</span>
                </div>
                <button id="playBtn">▶️ 开始动画</button>
                <button id="resetBtn">🔄 重置</button>
                <button id="stepBtn">⏭️ 单步执行</button>
            </div>

            <div class="visualization-container">
                <div class="viz-panel">
                    <h3>🌳 树形归约过程</h3>
                    <canvas id="treeCanvas" width="600" height="500"></canvas>
                    <div class="legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background: #4CAF50;"></div>
                            <span>活跃线程</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #9E9E9E;"></div>
                            <span>非活跃线程</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #FF9800;"></div>
                            <span>正在计算</span>
                        </div>
                    </div>
                </div>

                <div class="viz-panel">
                    <h3>💾 共享内存状态</h3>
                    <canvas id="memoryCanvas" width="600" height="500"></canvas>
                    <div class="legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background: #2196F3;"></div>
                            <span>已更新</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #FFC107;"></div>
                            <span>正在读取</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #E0E0E0;"></div>
                            <span>未使用</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="stats">
                <div class="stat-card">
                    <div class="label">当前步长 (Stride)</div>
                    <div class="value" id="currentStride">1</div>
                </div>
                <div class="stat-card">
                    <div class="label">迭代次数</div>
                    <div class="value" id="iterationCount">0</div>
                </div>
                <div class="stat-card">
                    <div class="label">活跃线程数</div>
                    <div class="value" id="activeThreads">0</div>
                </div>
                <div class="stat-card">
                    <div class="label">完成度</div>
                    <div class="value" id="completion">0%</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 配置
        const CONFIG = {{
            version: '{version_key}',
            reduceLoop: '{config['reduce_loop']}',
            indexCalc: '{config['index_calc']}',
            loadOptimization: {str(config['load_optimization']).lower()},
            unrollWarp: {str(config['unroll_warp']).lower()},
            unrollComplete: {str(config['unroll_complete']).lower()},
            multiElement: {str(config['multi_element']).lower()},
            useShuffle: {str(config['use_shuffle']).lower()}
        }};

        let config = {{
            threadCount: 32,
            speed: 5,
            isPlaying: false,
            currentStride: CONFIG.reduceLoop === 'backward' ? 16 : 1,
            iteration: 0,
            animationFrame: null
        }};

        // 状态
        let sharedMemory = [];
        let activeThreads = new Set();
        let computingThreads = new Set();
        let readingIndices = new Set();

        // Canvas 元素
        const treeCanvas = document.getElementById('treeCanvas');
        const memoryCanvas = document.getElementById('memoryCanvas');
        const treeCtx = treeCanvas.getContext('2d');
        const memoryCtx = memoryCanvas.getContext('2d');

        // 初始化
        function init() {{
            config.threadCount = parseInt(document.getElementById('threadCount').value);
            sharedMemory = new Array(config.threadCount).fill(0).map((_, i) => i + 1);
            config.currentStride = CONFIG.reduceLoop === 'backward' ? Math.floor(config.threadCount / 2) : 1;
            config.iteration = 0;
            activeThreads.clear();
            computingThreads.clear();
            readingIndices.clear();
            updateDisplays();
            draw();
        }}

        // 更新显示
        function updateDisplays() {{
            document.getElementById('threadCountValue').textContent = config.threadCount;
            document.getElementById('speedValue').textContent = config.speed;
            document.getElementById('currentStride').textContent = config.currentStride;
            document.getElementById('iterationCount').textContent = config.iteration;
            
            const activeCount = activeThreads.size;
            document.getElementById('activeThreads').textContent = activeCount;
            
            const maxIterations = CONFIG.reduceLoop === 'backward' 
                ? Math.ceil(Math.log2(config.threadCount))
                : Math.ceil(Math.log2(config.threadCount));
            const completion = Math.min(100, Math.round((config.iteration / maxIterations) * 100));
            document.getElementById('completion').textContent = completion + '%';
        }}

        // 执行一步归约
        function stepReduce() {{
            if (CONFIG.reduceLoop === 'backward') {{
                if (config.currentStride <= 0) {{
                    config.isPlaying = false;
                    document.getElementById('playBtn').textContent = '▶️ 开始动画';
                    return;
                }}
            }} else {{
                if (config.currentStride >= config.threadCount) {{
                    config.isPlaying = false;
                    document.getElementById('playBtn').textContent = '▶️ 开始动画';
                    return;
                }}
            }}

            // 清除之前的状态
            activeThreads.clear();
            computingThreads.clear();
            readingIndices.clear();

            // 确定哪些线程参与计算
            if (CONFIG.reduceLoop === 'backward') {{
                // 反向循环：只有前 stride 个线程参与
                for (let i = 0; i < config.currentStride; i++) {{
                    if (i + config.currentStride < config.threadCount) {{
                        activeThreads.add(i);
                        computingThreads.add(i);
                        readingIndices.add(i + config.currentStride);
                    }}
                }}
            }} else {{
                // 正向循环
                if (CONFIG.indexCalc === 'mod') {{
                    // v0: 使用模运算
                    for (let i = 0; i < config.threadCount; i++) {{
                        if (i % (2 * config.currentStride) === 0 && i + config.currentStride < config.threadCount) {{
                            activeThreads.add(i);
                            computingThreads.add(i);
                            readingIndices.add(i + config.currentStride);
                        }}
                    }}
                }} else {{
                    // v1+: 使用连续索引
                    for (let i = 0; i < config.threadCount; i++) {{
                        const index = 2 * config.currentStride * i;
                        if (index < config.threadCount && index + config.currentStride < config.threadCount) {{
                            activeThreads.add(index);
                            computingThreads.add(index);
                            readingIndices.add(index + config.currentStride);
                        }}
                    }}
                }}
            }}

            // 执行归约（模拟）
            for (let i of activeThreads) {{
                if (i + config.currentStride < config.threadCount) {{
                    sharedMemory[i] += sharedMemory[i + config.currentStride];
                }}
            }}

            config.iteration++;
            updateDisplays();
            draw();

            // 准备下一步
            setTimeout(() => {{
                if (config.isPlaying) {{
                    if (CONFIG.reduceLoop === 'backward') {{
                        config.currentStride = Math.floor(config.currentStride / 2);
                    }} else {{
                        config.currentStride *= 2;
                    }}
                    stepReduce();
                }}
            }}, 1000 / config.speed);
        }}

        // 绘制树形归约
        function drawTree() {{
            const ctx = treeCtx;
            const width = treeCanvas.width;
            const height = treeCanvas.height;
            const padding = 40;
            const cellWidth = (width - 2 * padding) / config.threadCount;
            const cellHeight = 30;
            const levels = Math.ceil(Math.log2(config.threadCount)) + 1;

            ctx.clearRect(0, 0, width, height);

            // 绘制每一层
            for (let level = 0; level < levels; level++) {{
                const stride = CONFIG.reduceLoop === 'backward' 
                    ? Math.floor(config.threadCount / Math.pow(2, level))
                    : Math.pow(2, level);
                const y = padding + level * (cellHeight + 40);
                const activeStride = level === config.iteration ? config.currentStride : 0;

                for (let i = 0; i < config.threadCount; i++) {{
                    const x = padding + i * cellWidth;
                    const isActive = level === config.iteration && activeThreads.has(i);
                    const isComputing = level === config.iteration && computingThreads.has(i);
                    const isReading = level === config.iteration && readingIndices.has(i);

                    // 绘制单元格
                    if (isComputing) {{
                        ctx.fillStyle = '#FF9800';
                    }} else if (isReading) {{
                        ctx.fillStyle = '#FFC107';
                    }} else if (isActive) {{
                        ctx.fillStyle = '#4CAF50';
                    }} else if (level < config.iteration || (level === config.iteration && !isActive)) {{
                        ctx.fillStyle = '#9E9E9E';
                    }} else {{
                        ctx.fillStyle = '#E0E0E0';
                    }}

                    ctx.fillRect(x, y, cellWidth - 2, cellHeight);
                    ctx.strokeStyle = '#333';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(x, y, cellWidth - 2, cellHeight);

                    // 绘制值
                    if (level <= config.iteration) {{
                        ctx.fillStyle = '#000';
                        ctx.font = '10px Arial';
                        ctx.textAlign = 'center';
                        ctx.fillText(
                            sharedMemory[i].toFixed(0),
                            x + cellWidth / 2,
                            y + cellHeight / 2 + 4
                        );
                    }}
                }}

                // 绘制层标签
                ctx.fillStyle = '#666';
                ctx.font = 'bold 12px Arial';
                ctx.textAlign = 'left';
                ctx.fillText('Stride ' + stride, 10, y + cellHeight / 2 + 4);
            }}
        }}

        // 绘制共享内存状态
        function drawMemory() {{
            const ctx = memoryCtx;
            const width = memoryCanvas.width;
            const height = memoryCanvas.height;
            const padding = 40;
            const barWidth = (width - 2 * padding) / config.threadCount;
            const maxValue = Math.max(...sharedMemory, 1);
            const barMaxHeight = height - 2 * padding - 60;

            ctx.clearRect(0, 0, width, height);

            // 绘制标题
            ctx.fillStyle = '#333';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('共享内存数组 (shared_data)', width / 2, 25);

            // 绘制每个内存位置
            for (let i = 0; i < config.threadCount; i++) {{
                const x = padding + i * barWidth;
                const value = sharedMemory[i];
                const barHeight = (value / maxValue) * barMaxHeight;
                const y = height - padding - barHeight - 30;

                // 确定颜色
                if (computingThreads.has(i)) {{
                    ctx.fillStyle = '#FF9800';
                }} else if (readingIndices.has(i)) {{
                    ctx.fillStyle = '#FFC107';
                }} else if (activeThreads.has(i)) {{
                    ctx.fillStyle = '#2196F3';
                }} else {{
                    ctx.fillStyle = '#E0E0E0';
                }}

                ctx.fillRect(x, y, barWidth - 2, barHeight);
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 1;
                ctx.strokeRect(x, y, barWidth - 2, barHeight);

                // 绘制值
                ctx.fillStyle = '#000';
                ctx.font = '9px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(
                    value.toFixed(0),
                    x + barWidth / 2,
                    y - 5
                );

                // 绘制索引
                ctx.fillStyle = '#666';
                ctx.font = '8px Arial';
                ctx.fillText('[' + i + ']', x + barWidth / 2, height - padding - 10);
            }}

            // 绘制当前操作说明
            if ((CONFIG.reduceLoop === 'backward' && config.currentStride > 0) ||
                (CONFIG.reduceLoop === 'forward' && config.currentStride < config.threadCount)) {{
                ctx.fillStyle = '#333';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                const activeList = Array.from(activeThreads).join(', ');
                const opText = '当前操作: shared_data[i] += shared_data[i + ' + config.currentStride + '] (i = ' + (activeList || '无') + ')';
                ctx.fillText(opText, width / 2, height - 15);
            }} else {{
                ctx.fillStyle = '#4CAF50';
                ctx.font = 'bold 14px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('✓ 归约完成！结果: ' + sharedMemory[0], width / 2, height - 15);
            }}
        }}

        // 绘制所有内容
        function draw() {{
            drawTree();
            drawMemory();
        }}

        // 事件监听
        document.getElementById('threadCount').addEventListener('input', (e) => {{
            config.threadCount = parseInt(e.target.value);
            document.getElementById('threadCountValue').textContent = config.threadCount;
            if (!config.isPlaying) {{
                init();
            }}
        }});

        document.getElementById('speed').addEventListener('input', (e) => {{
            config.speed = parseInt(e.target.value);
            document.getElementById('speedValue').textContent = config.speed;
        }});

        document.getElementById('playBtn').addEventListener('click', () => {{
            if (config.isPlaying) {{
                config.isPlaying = false;
                document.getElementById('playBtn').textContent = '▶️ 开始动画';
            }} else {{
                if ((CONFIG.reduceLoop === 'backward' && config.currentStride <= 0) ||
                    (CONFIG.reduceLoop === 'forward' && config.currentStride >= config.threadCount)) {{
                    init();
                }}
                config.isPlaying = true;
                document.getElementById('playBtn').textContent = '⏸️ 暂停';
                stepReduce();
            }}
        }});

        document.getElementById('resetBtn').addEventListener('click', () => {{
            config.isPlaying = false;
            document.getElementById('playBtn').textContent = '▶️ 开始动画';
            init();
        }});

        document.getElementById('stepBtn').addEventListener('click', () => {{
            if ((CONFIG.reduceLoop === 'backward' && config.currentStride <= 0) ||
                (CONFIG.reduceLoop === 'forward' && config.currentStride >= config.threadCount)) {{
                init();
                return;
            }}
            config.isPlaying = false;
            document.getElementById('playBtn').textContent = '▶️ 开始动画';
            
            // 清除之前的状态
            activeThreads.clear();
            computingThreads.clear();
            readingIndices.clear();

            // 确定哪些线程参与计算
            if (CONFIG.reduceLoop === 'backward') {{
                for (let i = 0; i < config.currentStride; i++) {{
                    if (i + config.currentStride < config.threadCount) {{
                        activeThreads.add(i);
                        computingThreads.add(i);
                        readingIndices.add(i + config.currentStride);
                    }}
                }}
            }} else {{
                if (CONFIG.indexCalc === 'mod') {{
                    for (let i = 0; i < config.threadCount; i++) {{
                        if (i % (2 * config.currentStride) === 0 && i + config.currentStride < config.threadCount) {{
                            activeThreads.add(i);
                            computingThreads.add(i);
                            readingIndices.add(i + config.currentStride);
                        }}
                    }}
                }} else {{
                    for (let i = 0; i < config.threadCount; i++) {{
                        const index = 2 * config.currentStride * i;
                        if (index < config.threadCount && index + config.currentStride < config.threadCount) {{
                            activeThreads.add(index);
                            computingThreads.add(index);
                            readingIndices.add(index + config.currentStride);
                        }}
                    }}
                }}
            }}

            // 执行归约
            for (let i of activeThreads) {{
                if (i + config.currentStride < config.threadCount) {{
                    sharedMemory[i] += sharedMemory[i + config.currentStride];
                }}
            }}

            config.iteration++;
            if (CONFIG.reduceLoop === 'backward') {{
                config.currentStride = Math.floor(config.currentStride / 2);
            }} else {{
                config.currentStride *= 2;
            }}
            updateDisplays();
            draw();
        }});

        // 响应式调整 Canvas 大小
        function resizeCanvases() {{
            const container = document.querySelector('.visualization-container');
            const panels = container.querySelectorAll('.viz-panel');
            panels.forEach(panel => {{
                const canvas = panel.querySelector('canvas');
                if (canvas) {{
                    const rect = panel.getBoundingClientRect();
                    canvas.width = rect.width - 40;
                    canvas.height = Math.min(500, (rect.width - 40) * 0.8);
                    draw();
                }}
            }});
        }}

        window.addEventListener('resize', resizeCanvases);

        // 初始化
        init();
        resizeCanvases();
    </script>
</body>
</html>'''
    
    return html_template

def main():
    """主函数：生成所有版本的可视化页面"""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    for version_key, config in VERSIONS.items():
        html_content = generate_html(version_key, config)
        output_file = os.path.join(output_dir, f"reduce_{version_key}_visualization.html")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ 已生成: {output_file}")

if __name__ == '__main__':
    main()
