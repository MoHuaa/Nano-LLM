#include "ops.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ==========================================
// RMSNorm CUDA Kernel
// ==========================================
// 每个线程块 (Block) 处理一行数据 (Row)
// 利用 Shared Memory 进行 Block 内归约 (Reduction)
__global__ void rmsnorm_kernel(const float* input, const float* weight, float* output, 
                              int dim, float epsilon) {
    // 1. 获取当前行号 (blockIdx.x)
    int row = blockIdx.x;
    // 2. 获取当前线程号 (threadIdx.x)
    int tid = threadIdx.x;

    // 指向当前行数据的指针
    const float* row_in = input + row * dim;
    float* row_out = output + row * dim;

    // 声明共享内存 (大小在 launch 时指定)
    // 用于存储每个线程计算的局部平方和
    extern __shared__ float sdata[];

    // 3. 计算局部平方和 (Partial Sum)
    // 一个线程可能需要处理多个元素 (如果 dim > blockDim.x)
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        sum += row_in[i] * row_in[i];
    }
    sdata[tid] = sum;
    __syncthreads(); // 等待所有线程把局部结果写进去

    // 4. Block 内归约 (Reduction) 求出总平方和
    // 简单的树状归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 5. 此时 sdata[0] 就是总平方和
    float rms = 0.0f;
    if (tid == 0) {
        rms = rsqrtf(sdata[0] / dim + epsilon); // rsqrtf = 1 / sqrt
        sdata[0] = rms; // 把结果放回去广播
    }
    __syncthreads();
    
    rms = sdata[0]; // 所有线程读取 RMS

    // 6. 归一化并写回
    for (int i = tid; i < dim; i += blockDim.x) {
        row_out[i] = row_in[i] * rms * weight[i];
    }
}

// Host Wrapper for RMSNorm
void launch_rmsnorm(const Tensor& input, const Tensor& weight, Tensor& output, float epsilon) {
    int dim = input.shape().back();
    int rows = input.numel() / dim;
    
    // 设置 Block 维度为 2 的幂次方，以支持树状归约
    // 这里我们直接用 256，足够处理常见 dim
    int threads = 256;
    while (threads < dim && threads < 1024) threads *= 2;
    
    dim3 block(threads);
    dim3 grid(rows);
    
    size_t shared_mem_size = threads * sizeof(float);

    rmsnorm_kernel<<<grid, block, shared_mem_size>>>(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(weight.data()),
        static_cast<float*>(output.data()),
        dim, epsilon
    );
    CHECK_CUDA(cudaGetLastError());
}

// ==========================================
// SwiGLU CUDA Kernel
// ==========================================
// 简单的 Element-wise 操作，一一对应
__global__ void swiglu_kernel(const float* gate, const float* feat, float* output, int n) {
    // 1. 计算全局索引 idx = blockIdx.x * blockDim.x + threadIdx.x
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. 边界检查 if (idx < n)
    if (idx < n) {
        float x = gate[idx];
        // 3. 计算 SiLU(gate[idx]) * feat[idx]
        // CUDA 中的 exp 函数是 expf (针对 float 优化)
        float silu = x / (1.0f + expf(-x));
        output[idx] = silu * feat[idx];
    }
}

void launch_swiglu(const Tensor& gate, const Tensor& feat, Tensor& output) {
    int n = gate.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    swiglu_kernel<<<blocks, threads>>>(
        static_cast<const float*>(gate.data()),
        static_cast<const float*>(feat.data()),
        static_cast<float*>(output.data()),
        n
    );
    CHECK_CUDA(cudaGetLastError());
}

// ==========================================
// ==========================================
// RoPE CUDA Kernel
// ==========================================
__global__ void rope_kernel(const float* input, const float* pos, float* output, 
                           int head_dim, int total_rows) {
    // 1. 计算全局索引 idx
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = total_rows * head_dim / 2;

    if (idx < total_pairs) {
        // 2. 确定当前处理的是哪一行 (row_idx) 和哪一对元素 (j)
        // 一行有 head_dim/2 对元素
        int row_idx = idx / (head_dim / 2);
        int j = idx % (head_dim / 2);

        // 3. 读取位置 p = pos[row_idx]
        float p = pos[row_idx];

        // 4. 计算旋转角度
        // pow(10000, -2j/dim) -> __powf(10000.0f, ...)
        float freq = __powf(10000.0f, -2.0f * j / head_dim);
        float theta = p * freq;

        // 5. 读取输入数据 (x, y)
        // 这一对在原数组中的偏移量：行偏移 + 列偏移
        int offset = row_idx * head_dim + 2 * j;
        float x = input[offset];
        float y = input[offset + 1];

        // 6. 旋转并写回
        float cos_t, sin_t;
        __sincosf(theta, &sin_t, &cos_t); // CUDA 高速三角函数

        output[offset]     = x * cos_t - y * sin_t;
        output[offset + 1] = x * sin_t + y * cos_t;
    }
}

void launch_rope(const Tensor& input, const Tensor& pos, Tensor& output) {
    int head_dim = input.shape().back();
    int total_rows = input.numel() / head_dim;
    int total_pairs = input.numel() / 2; // RoPE 是两两配对处理
    
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;

    rope_kernel<<<blocks, threads>>>(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(pos.data()),
        static_cast<float*>(output.data()),
        head_dim, total_rows
    );
    CHECK_CUDA(cudaGetLastError());
}
