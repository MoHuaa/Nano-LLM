#pragma once
#include "tensor.h"

// 算子接口
namespace ops {

// C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

// Softmax along the last dimension
void softmax(Tensor& input);

// RMSNorm (Root Mean Square Layer Normalization)
// input: [rows, dim], weight: [dim], output: [rows, dim]
// Formula: output = input * weight / sqrt(mean(input^2) + epsilon)
void rmsnorm(const Tensor& input, const Tensor& weight, Tensor& output, float epsilon = 1e-5f);

// SwiGLU (SiLU + Gated Linear Unit)
// gate: [M, N], feat: [M, N], output: [M, N]
// Formula: output = SiLU(gate) * feat = (gate / (1 + exp(-gate))) * feat
void swiglu(const Tensor& gate, const Tensor& feat, Tensor& output);

// RoPE (Rotary Positional Embeddings)
// input: [batch, seq_len, head_dim] (simplified for now, usually [batch, heads, seq, dim])
// pos: [seq_len] (position indices, e.g., 0, 1, 2...)
// output: [batch, seq_len, head_dim]
// Formula: Rotate every pair of elements (x, y) by angle theta = pos / (10000^(2i/dim))
void rope(const Tensor& input, const Tensor& pos, Tensor& output);

} // namespace ops
