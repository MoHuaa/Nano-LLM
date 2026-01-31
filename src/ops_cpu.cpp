#include "ops.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace ops {

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    // 1. 检查所有 Tensor 是否在 CPU 上 (A.device() ...)
    
    // 2. 获取维度信息 (M, K, N) 并检查维度匹配
    // A: [M, K], B: [K, N], C: [M, N]
    
    // 3. 获取数据指针
    const float* ptrA = static_cast<const float*>(A.data());
    const float* ptrB = static_cast<const float*>(B.data());
    float* ptrC = static_cast<float*>(C.data());

    // 4. 实现矩阵乘法循环 (M, N, K)
    // for (m in M)
    //   for (n in N)
    //     sum = 0
    //     for (k in K) ...
    
}

void softmax(Tensor& input) {
    // 1. 检查是否在 CPU
    
    // 2. 将 Tensor 视为 [rows, cols] 的 2D 矩阵
    // rows = product(shape[:-1]), cols = shape[-1]
    
    // 3. 对每一行进行 Softmax
    //   a. 找到当前行的最大值 max_val
    //   b. 计算 sum = sum(exp(x - max_val))
    //   c. 更新每个元素 x = exp(x - max_val) / sum
    
}

} // namespace ops
