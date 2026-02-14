#include "ops.h"
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include <numeric>

#define REQUIRE_CPU(t) \
    if (t.device() != Device::CPU) { \
        fprintf(stderr, "Error: Tensor must be on CPU but is on CUDA at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

namespace ops {

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    REQUIRE_CPU(A);
    REQUIRE_CPU(B);
    REQUIRE_CPU(C);

    const auto& shapeA = A.shape();
    const auto& shapeB = B.shape();
    const auto& shapeC = C.shape();

    if (shapeA.size() != 2 || shapeB.size() != 2 || shapeC.size() != 2) {
        throw std::runtime_error("matmul: Only supports 2D matrices");
    }

    size_t M = shapeA[0];
    size_t K = shapeA[1];
    size_t N = shapeB[1];

    if (shapeB[0] != K) {
        throw std::runtime_error("matmul: Dimension mismatch, A.cols != B.rows");
    }

    if (shapeC[0] != M || shapeC[1] != N) {
        throw std::runtime_error("matmul: Output shape mismatch");
    }

    const float* ptrA = static_cast<const float*>(A.data());
    const float* ptrB = static_cast<const float*>(B.data());
    float* ptrC = static_cast<float*>(C.data());

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
    REQUIRE_CPU(input);

    const auto& shape = input.shape();
    if (shape.empty()) return;

    size_t cols = shape.back();
    size_t rows = input.numel() / cols;

    float* data = static_cast<float*>(input.data());

    for (size_t i = 0; i < rows; ++i) {
        float* row_ptr = data + i * cols;
        
        float max_val = row_ptr[0];
        for (size_t j = 1; j < cols; ++j) {
            if (row_ptr[j] > max_val) max_val = row_ptr[j];
        }

        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            row_ptr[j] = std::exp(row_ptr[j] - max_val);
            sum += row_ptr[j];
        }

        for (size_t j = 0; j < cols; ++j) {
            row_ptr[j] /= sum;
        }
    }
}

void rmsnorm(const Tensor& input, const Tensor& weight, Tensor& output, float epsilon) {
    REQUIRE_CPU(input);
    REQUIRE_CPU(weight);
    REQUIRE_CPU(output);

    const auto& shape = input.shape();
    size_t dim = shape.back();
    size_t rows = input.numel() / dim;

    if (weight.numel() != dim) {
        throw std::runtime_error("rmsnorm: Weight dimension mismatch");
    }

    const float* in_ptr = static_cast<const float*>(input.data());
    const float* w_ptr = static_cast<const float*>(weight.data());
    float* out_ptr = static_cast<float*>(output.data());

    for (size_t i = 0; i < rows; ++i) {
        const float* row_in = in_ptr + i * dim;
        float* row_out = out_ptr + i * dim;

        float ss = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            ss += row_in[j] * row_in[j];
        }

        float rms = 1.0f / std::sqrt(ss / dim + epsilon);

        for (size_t j = 0; j < dim; ++j) {
            row_out[j] = row_in[j] * rms * w_ptr[j];
        }
    }
}

void swiglu(const Tensor& gate, const Tensor& feat, Tensor& output) {
    REQUIRE_CPU(gate);
    REQUIRE_CPU(feat);
    REQUIRE_CPU(output);

    if (gate.numel() != feat.numel() || gate.numel() != output.numel()) {
        throw std::runtime_error("swiglu: Element size mismatch");
    }

    const float* gate_ptr = static_cast<const float*>(gate.data());
    const float* feat_ptr = static_cast<const float*>(feat.data());
    float* out_ptr = static_cast<float*>(output.data());
    
    size_t n = gate.numel();

    for(size_t i = 0; i < n; ++i) {
        float x = gate_ptr[i]; 
        float silu_x = x / (1.0f + std::exp(-x));
        out_ptr[i] = silu_x * feat_ptr[i];
    }
}

void rope(const Tensor& input, const Tensor& pos, Tensor& output) {
    // 1. 检查 CPU
    REQUIRE_CPU(input);
    REQUIRE_CPU(pos);
    REQUIRE_CPU(output);

    // 2. 假设 input 形状为 [batch, seq_len, head_dim]
    // 简化起见，我们将其视为 [rows, head_dim]，其中 rows = batch * seq_len
    const auto& shape = input.shape();
    size_t head_dim = shape.back();
    size_t total_rows = input.numel() / head_dim;

    // 3. 检查 pos 长度 (这里简化处理，假设 pos 包含了每一行的位置索引)
    // 实际 Llama 中 pos 是 [seq_len]，通过广播应用到所有 batch。
    // 为了简化测试，我们要求 pos 的长度等于 total_rows (即每一行都有一个对应的位置 ID)
    if (pos.numel() != total_rows) {
        throw std::runtime_error("rope: pos numel must match input rows (simplified)");
    }
    
    // RoPE 只能处理偶数维度的 head_dim (因为要两两配对)
    if (head_dim % 2 != 0) {
        throw std::runtime_error("rope: head_dim must be even");
    }

    const float* in_ptr = static_cast<const float*>(input.data());
    const float* pos_ptr = static_cast<const float*>(pos.data());
    float* out_ptr = static_cast<float*>(output.data());

    // 4. 实现 RoPE 循环
    // 对每一行 (对应一个 token):
    //   获取当前位置 p = pos_ptr[i]
    //   对每对元素 (x, y) = (in[2j], in[2j+1]):
    //     计算角度 theta = p / (10000 ^ (2j / head_dim))
    //     旋转:
    //       x' = x * cos(theta) - y * sin(theta)
    //       y' = x * sin(theta) + y * cos(theta)
    
    // 请在此处实现循环逻辑...
        for (size_t i = 0; i < total_rows; ++i) {
        // 当前行的起始指针
        const float* row_in = in_ptr + i * head_dim;
        float* row_out = out_ptr + i * head_dim;
        
        // 当前行的位置索引
        float p = pos_ptr[i];

        // 遍历每一对元素 (j 从 0 到 head_dim/2)
        for (size_t j = 0; j < head_dim / 2; ++j) {
            float x = row_in[2 * j];
            float y = row_in[2 * j + 1];

            // 计算频率和角度
            // pow(10000, -2j/dim) 等价于 1 / (10000^(2j/dim))
            float freq = std::pow(10000.0f, -2.0f * j / head_dim);
            float theta = p * freq;

            float cos_theta = std::cos(theta);
            float sin_theta = std::sin(theta);

            // 旋转
            row_out[2 * j]     = x * cos_theta - y * sin_theta;
            row_out[2 * j + 1] = x * sin_theta + y * cos_theta;
        }
    }
}

} // namespace ops
