#include "ops.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#define REQUIRE_CPU(t) \
    if (t.device() != Device::CPU) { \
        fprintf(stderr, "Error: Tensor must be on CPU but is on CUDA at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }
namespace ops {
void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    // 1. 检查所有 Tensor 是否在 CPU 上 (A.device() ...)
    REQUIRE_CPU(A);
    REQUIRE_CPU(B);
    REQUIRE_CPU(C);
    // 2. 获取维度信息 (M, K, N) 并检查维度匹配
    // A: [M, K], B: [K, N], C: [M, N]
    const auto& shapeA = A.shape();
    const auto& shapeB = B.shape();
    const auto& shapeC = C.shape();

    // 检查是否都是二维矩阵
    if (shapeA.size() != 2 || shapeB.size() != 2 || shapeC.size() != 2) {
        throw std::runtime_error("matmul: Only supports 2D matrices");
    }

    size_t M = shapeA[0];
    size_t K = shapeA[1];
    size_t N = shapeB[1];

    // 检查矩阵乘法维度规则：Inner dimensions must match (A.cols == B.rows)
    if (shapeB[0] != K) {
        throw std::runtime_error("matmul: Dimension mismatch, A.cols != B.rows");
    }

    // 检查输出矩阵形状是否正确
    if (shapeC[0] != M || shapeC[1] != N) {
        throw std::runtime_error("matmul: Output shape mismatch");
    }
    // 3. 获取数据指针
    const float* ptrA = static_cast<const float*>(A.data());
    const float* ptrB = static_cast<const float*>(B.data());
    float* ptrC = static_cast<float*>(C.data());

    // 4. 实现矩阵乘法循环 (M, N, K)
    // Naive implementation O(M*N*K)
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                // Flattened index access: row * stride + col
                sum += ptrA[m * K + k] * ptrB[k * N + n];
            }
            ptrC[m * N + n] = sum;
        }
    }
}

void softmax(Tensor& input) {
    // 1. 检查是否在 CPU
    REQUIRE_CPU(input);

    // 2. 将 Tensor 视为 [rows, cols] 的 2D 矩阵
    const auto& shape = input.shape();
    if (shape.empty()) return;

    size_t cols = shape.back();
    size_t rows = input.numel() / cols; // 其余维度合并为 rows

    float* data = static_cast<float*>(input.data());

    // 3. 对每一行进行 Softmax
    for (size_t i = 0; i < rows; ++i) {
        float* row_ptr = data + i * cols; // 指向当前行起始位置
        
        // a. 找到当前行的最大值 (Max Trick for numerical stability)
        float max_val = row_ptr[0];
        for (size_t j = 1; j < cols; ++j) {
            if (row_ptr[j] > max_val) max_val = row_ptr[j];
        }

        // b. 计算 exp 并求和
        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            row_ptr[j] = std::exp(row_ptr[j] - max_val);
            sum += row_ptr[j];
        }

        // c. 归一化
        for (size_t j = 0; j < cols; ++j) {
            row_ptr[j] /= sum;
        }
    }
}

} // namespace ops
