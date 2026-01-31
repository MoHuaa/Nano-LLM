#pragma once
#include "tensor.h"

// 算子接口
namespace ops {

// C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

// Softmax along the last dimension
void softmax(Tensor& input);

} // namespace ops
