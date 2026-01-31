#include "ops.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace ops {

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.device() != Device::CPU || B.device() != Device::CPU || C.device() != Device::CPU) {
        throw std::runtime_error("matmul_cpu: All tensors must be on CPU");
    }

    const auto& shapeA = A.shape();
    const auto& shapeB = B.shape();
    const auto& shapeC = C.shape();

    if (shapeA.size() != 2 || shapeB.size() != 2 || shapeC.size() != 2) {
        throw std::runtime_error("matmul_cpu: Only supports 2D matrices");
    }

    size_t M = shapeA[0];
    size_t K = shapeA[1];
    size_t N = shapeB[1];

    if (shapeB[0] != K || shapeC[0] != M || shapeC[1] != N) {
        throw std::runtime_error("matmul_cpu: Shape mismatch");
    }

    const float* ptrA = static_cast<const float*>(A.data());
    const float* ptrB = static_cast<const float*>(B.data());
    float* ptrC = static_cast<float*>(C.data());

    // Naive implementation O(M*N*K)
    // TODO: Optimize with Tiling / OpenMP / SIMD
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += ptrA[m * K + k] * ptrB[k * N + n];
            }
            ptrC[m * N + n] = sum;
        }
    }
}

void softmax(Tensor& input) {
    if (input.device() != Device::CPU) {
        throw std::runtime_error("softmax_cpu: Input must be on CPU");
    }

    const auto& shape = input.shape();
    size_t rows = 1;
    size_t cols = shape.back();
    
    // Treat as [N, cols] where N is product of all other dims
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        rows *= shape[i];
    }

    float* data = static_cast<float*>(input.data());

    for (size_t i = 0; i < rows; ++i) {
        float* row_ptr = data + i * cols;
        
        // 1. Find max for numerical stability
        float max_val = row_ptr[0];
        for (size_t j = 1; j < cols; ++j) {
            if (row_ptr[j] > max_val) max_val = row_ptr[j];
        }

        // 2. Exp and sum
        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            row_ptr[j] = std::exp(row_ptr[j] - max_val);
            sum += row_ptr[j];
        }

        // 3. Normalize
        for (size_t j = 0; j < cols; ++j) {
            row_ptr[j] /= sum;
        }
    }
}

} // namespace ops
