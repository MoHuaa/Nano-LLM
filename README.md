# Nano-LLM: Lightweight C++/CUDA Transformer Inference Engine

## 📖 项目简介 (Introduction)
**Nano-LLM** 是一个从零实现的、不依赖 PyTorch/TensorFlow 等重型框架的轻量级 Transformer 推理引擎。它专为理解大语言模型（LLM）的底层计算流与高性能计算（HPC）优化而设计。

本项目旨在在消费级显卡（如 RTX 5070 Ti Laptop）上实现 Llama 3 / TinyLlama 等模型的高效推理，展示 C++ 工程能力与 CUDA 异构编程技巧。

## 🎯 核心特性 (Features)
*   **纯 C++/CUDA 实现**：深入理解底层矩阵计算与内存管理。
*   **Llama 架构支持**：支持 RMSNorm, SwiGLU, RoPE (Rotary Positional Embeddings), GQA (Grouped Query Attention)。
*   **高性能优化**：
    *   **KV-Cache**：显著降低自回归生成的计算复杂度。
    *   **混合精度推理**：支持 FP16 / INT8 (Planned) 以减少显存占用与带宽压力。
    *   **Continuous Batching** (Planned): 提升服务吞吐量。
*   **零拷贝加载**：使用 `mmap` 直接加载 GGUF 或自定义格式的模型权重。

## 🛠️ 技术栈 (Tech Stack)
*   **Language**: C++17, CUDA
*   **Build System**: CMake
*   **Libraries**: cuBLAS (Matrix Multiplication), OpenMP (CPU Parallelism)
*   **Hardware**: NVIDIA GPU (Compute Capability 7.0+)

## 📅 开发计划 (Roadmap)

### Week 1: 基础架构搭建
- [ ] 项目环境搭建 (CMake, CUDA)
- [ ] 实现基础 `Tensor` 类 (CPU/GPU 内存管理, RAII)
- [ ] 实现 CPU 版本的基础算子 (MatMul, Softmax)
- [ ] 单元测试框架搭建

### Week 2: Llama 核心算子
- [ ] RMSNorm (Root Mean Square Layer Normalization)
- [ ] RoPE (Rotary Positional Embeddings)
- [ ] Multi-Head Attention / GQA 逻辑
- [ ] FeedForward Network (SwiGLU)

### Week 3: CUDA 加速与模型加载
- [ ] 接入 cuBLAS 加速矩阵乘法 (GEMM)
- [ ] 编写 CUDA Kernels (Element-wise ops: Add, Silu, RMSNorm)
- [ ] 实现 Model Loader (解析权重文件)
- [ ] 跑通 Forward Pass

### Week 4: 推理优化与服务化
- [ ] 实现 KV-Cache 管理
- [ ] 实现采样策略 (Greedy, Top-k, Top-p)
- [ ] 性能 Benchmark (vs PyTorch)
- [ ] 整理文档与 Demo

## 🚀 快速开始 (Quick Start)

```bash
mkdir build && cd build
cmake ..
make -j
./nano_llm_test
```
