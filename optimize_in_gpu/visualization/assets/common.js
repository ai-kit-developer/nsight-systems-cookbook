/**
 * 通用JavaScript文件 - CUDA GPU 性能优化可视化系统
 * 提供路由、导航、消息传递等通用功能
 */

// ========== 路由管理 ==========
const Router = {
    // 路由配置
    routes: {
        'reduce': 'reduce_index.html',
        'elementwise': 'elementwise.html',
        'spmv': 'spmv.html',
        'spmm': 'spmm.html',
        'sgemm': 'sgemm.html',
        'sgemv': 'sgemv.html'
    },

    // 标签页名称映射
    tabNames: {
        'reduce': 'Reduce 归约',
        'elementwise': 'Elementwise 逐元素操作',
        'spmv': 'SpMV 稀疏矩阵-向量乘法',
        'spmm': 'SpMM 稀疏矩阵-矩阵乘法',
        'sgemm': 'SGEMM 矩阵-矩阵乘法',
        'sgemv': 'SGEMV 矩阵-向量乘法'
    },

    /**
     * 从URL获取当前标签页
     */
    getTabFromURL() {
        const hash = window.location.hash.slice(1);
        const params = new URLSearchParams(window.location.search);
        return hash || params.get('tab') || 'reduce';
    },

    /**
     * 更新URL
     */
    updateURL(tabName, params = {}) {
        const url = new URL(window.location);
        url.searchParams.set('tab', tabName);
        
        // 添加其他参数
        Object.keys(params).forEach(key => {
            if (params[key]) {
                url.searchParams.set(key, params[key]);
            }
        });
        
        window.history.pushState({ tab: tabName, ...params }, '', url);
    },

    /**
     * 获取URL参数
     */
    getURLParam(name) {
        const params = new URLSearchParams(window.location.search);
        return params.get(name);
    }
};

// ========== 导航管理 ==========
const Navigation = {
    /**
     * 检查是否在iframe中
     */
    isInIframe() {
        return window.parent !== window;
    },

    /**
     * 向父窗口发送导航消息
     */
    navigateToParent(url, data = {}) {
        if (this.isInIframe()) {
            window.parent.postMessage({
                type: 'navigate',
                url: url,
                ...data
            }, '*');
            return true;
        }
        return false;
    },

    /**
     * 导航到指定URL
     */
    navigate(url, params = {}) {
        // 如果在iframe中，通知父窗口
        if (this.navigateToParent(url, params)) {
            return;
        }

        // 否则直接跳转
        if (Object.keys(params).length > 0) {
            const urlObj = new URL(url, window.location);
            Object.keys(params).forEach(key => {
                urlObj.searchParams.set(key, params[key]);
            });
            window.location.href = urlObj.toString();
        } else {
            window.location.href = url;
        }
    }
};

// ========== 消息传递 ==========
const MessageBus = {
    /**
     * 发送消息到父窗口
     */
    send(type, data = {}) {
        if (window.parent !== window) {
            window.parent.postMessage({
                type: type,
                ...data
            }, '*');
        }
    },

    /**
     * 监听来自父窗口的消息
     */
    listen(callback) {
        window.addEventListener('message', (event) => {
            if (event.data && typeof event.data === 'object') {
                callback(event.data);
            }
        });
    }
};

// ========== UI工具函数 ==========
const UI = {
    /**
     * 显示加载状态
     */
    showLoading(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'block';
        }
    },

    /**
     * 隐藏加载状态
     */
    hideLoading(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'none';
        }
    },

    /**
     * 更新面包屑导航
     */
    updateBreadcrumb(items) {
        const breadcrumb = document.getElementById('breadcrumb');
        if (!breadcrumb) return;

        const html = items.map((item, index) => {
            if (index === items.length - 1) {
                return `<span>${item.text}</span>`;
            }
            return `<a href="${item.url || '#'}">${item.text}</a><span>/</span>`;
        }).join(' ');

        breadcrumb.innerHTML = html;
    },

    /**
     * 平滑滚动到元素
     */
    scrollTo(elementId, offset = 0) {
        const element = document.getElementById(elementId);
        if (element) {
            const elementPosition = element.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - offset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });
        }
    }
};

// ========== 工具函数 ==========
const Utils = {
    /**
     * 防抖函数
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * 节流函数
     */
    throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    /**
     * 格式化数字
     */
    formatNumber(num, decimals = 2) {
        return parseFloat(num).toFixed(decimals);
    },

    /**
     * 格式化文件大小
     */
    formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }
};

// ========== 导出到全局 ==========
if (typeof window !== 'undefined') {
    window.Router = Router;
    window.Navigation = Navigation;
    window.MessageBus = MessageBus;
    window.UI = UI;
    window.Utils = Utils;
}

