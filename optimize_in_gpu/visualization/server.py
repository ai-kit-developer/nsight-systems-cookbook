#!/usr/bin/env python3
"""
简单的 HTTP 服务器，用于提供 CUDA GPU 性能优化算法可视化页面
支持路由处理和URL重写
"""

import http.server
import socketserver
import os
import sys
import urllib.parse

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """自定义请求处理器"""
    
    # 路由映射表
    ROUTES = {
        '/': 'index.html',
        '/index': 'index.html',
        '/reduce': 'reduce_index.html',
        '/reduce/': 'reduce_index.html',
        '/elementwise': 'elementwise.html',
        '/spmv': 'spmv.html',
        '/spmm': 'spmm.html',
        '/sgemm': 'sgemm.html',
        '/sgemv': 'sgemv.html',
    }
    
    def do_GET(self):
        """处理GET请求"""
        # 解析URL
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # 处理路由
        if path in self.ROUTES:
            self.path = '/' + self.ROUTES[path]
        # 处理根路径
        elif path == '/':
            self.path = '/index.html'
        # 处理其他路径（保持原有行为）
        elif not path.startswith('/') or os.path.exists(path.lstrip('/')):
            pass
        else:
            # 尝试添加.html扩展名
            if not path.endswith('.html') and not '.' in os.path.basename(path):
                test_path = path + '.html'
                if os.path.exists(test_path.lstrip('/')):
                    self.path = test_path
        
        # 调用父类方法处理文件
        return super().do_GET()
    
    def end_headers(self):
        # 添加 CORS 头，允许跨域访问
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        # 添加缓存控制
        if self.path.endswith('.html'):
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
        super().end_headers()
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        # 美化日志输出
        log_entry = format % args
        # 只记录重要请求
        if not any(skip in log_entry for skip in ['favicon.ico', '.ico', '.png', '.jpg', '.gif']):
            sys.stderr.write(f"📄 [{self.log_date_time_string()}] {log_entry}\n")

def main():
    """启动服务器"""
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    Handler = MyHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("=" * 70)
            print(f"🚀 CUDA GPU 性能优化算法可视化服务器已启动")
            print("=" * 70)
            print(f"📍 服务器地址: http://localhost:{PORT}")
            print(f"📄 主页面: http://localhost:{PORT}/ 或 http://localhost:{PORT}/index.html")
            print("=" * 70)
            print("\n📚 可用路由:")
            print("  / 或 /index          - 主页面（标签页导航）")
            print("  /reduce              - Reduce 归约索引页（8个优化版本）")
            print("  /elementwise         - Elementwise 逐元素操作")
            print("  /spmv                - SpMV 稀疏矩阵-向量乘法")
            print("  /spmm                - SpMM 稀疏矩阵-矩阵乘法")
            print("  /sgemm               - SGEMM 矩阵-矩阵乘法")
            print("  /sgemv               - SGEMV 矩阵-向量乘法")
            print("\n💡 提示:")
            print("  - 支持URL参数，如: /reduce?version=v0")
            print("  - 支持标签页路由，如: /?tab=reduce")
            print("=" * 70)
            print(f"\n按 Ctrl+C 停止服务器\n")
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
        sys.exit(0)
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ 错误: 端口 {PORT} 已被占用")
            print(f"   请使用其他端口或关闭占用该端口的程序")
        else:
            print(f"❌ 错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
