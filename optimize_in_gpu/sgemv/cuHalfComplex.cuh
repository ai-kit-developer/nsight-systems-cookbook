#if !defined(CU_HALFCOMPLEX_H_)
#define CU_HALFCOMPLEX_H_

#if !defined(__CUDACC__)
#include <math.h>       /* import fabsf, sqrt */
#endif /* !defined(__CUDACC__) */
#include <cuda_fp16.h>

/**
 * cuHalfComplex: 半精度复数结构体
 * 用于表示使用 half 精度（FP16）的复数
 * 
 * 成员变量：
 * - r: 实部（real part）
 * - i: 虚部（imaginary part）
 * 
 * 支持的操作：
 * - 复数乘法：实现 (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
 * - 复数加法：实现 (a+bi) + (c+di) = (a+c) + (b+d)i
 * 
 * 优化：
 * - 使用 __device__ __forceinline__ 确保在设备端内联
 * - 使用 half 精度减少内存占用和带宽需求
 */
struct cuHalfComplex {
    half r;  // 实部
    half i;  // 虚部
    
    // 构造函数：初始化实部和虚部
    __host__ __device__ __forceinline__ cuHalfComplex( half a, half b ) : r(a), i(b) {}

    // 复数乘法运算符重载
    // 计算 (r + i*i) * (a.r + a.i*i) = (r*a.r - i*a.i) + (i*a.r + r*a.i)*i
    __device__ __forceinline__ cuHalfComplex operator*(const cuHalfComplex& a) {
        return cuHalfComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }

    // 复数加法运算符重载
    // 计算 (r + i*i) + (a.r + a.i*i) = (r+a.r) + (i+a.i)*i
    __device__ __forceinline__ cuHalfComplex operator+(const cuHalfComplex& a) {
        return cuHalfComplex(r+a.r, i+a.i);
    }
};

#endif /* !defined(CU_HALFCOMPLEX_H_) */
