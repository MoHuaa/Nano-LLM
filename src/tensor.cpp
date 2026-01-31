#include "tensor.h"
#include <numeric>
#include <cstring>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>

// 辅助宏：检查 CUDA 错误
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

Tensor::Tensor(const std::vector<size_t>& shape, Device device, DType dtype)
    : shape_(shape), device_(device), dtype_(dtype) {
    numel_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
    allocate();
}

Tensor::~Tensor() {
    free();
}

Tensor::Tensor(Tensor&& other) noexcept 
    : data_(other.data_), shape_(std::move(other.shape_)), numel_(other.numel_), device_(other.device_), dtype_(other.dtype_) {
    other.data_ = nullptr;
    other.numel_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        numel_ = other.numel_;
        device_ = other.device_;
        dtype_ = other.dtype_;
        
        other.data_ = nullptr;
        other.numel_ = 0;
    }
    return *this;
}

size_t Tensor::size_bytes() const {
    size_t element_size = (dtype_ == DType::Float32) ? 4 : 2;
    return numel_ * element_size;
}

void Tensor::allocate() {
    if (numel_ == 0) return;

    if (device_ == Device::CPU) {
        data_ = malloc(size_bytes());
        if (!data_) throw std::runtime_error("CPU memory allocation failed");
    } else {
        CHECK_CUDA(cudaMalloc(&data_, size_bytes()));
    }
}

void Tensor::free() {
    if (data_) {
        if (device_ == Device::CPU) {
            std::free(data_);
        } else {
            CHECK_CUDA(cudaFree(data_));
        }
        data_ = nullptr;
    }
}

void Tensor::to(Device target_device) {
    if (device_ == target_device) return;

    void* new_data = nullptr;
    size_t bytes = size_bytes();

    // 分配新内存
    if (target_device == Device::CPU) {
        new_data = malloc(bytes);
        if (!new_data) throw std::runtime_error("CPU allocation failed during transfer");
        // Device -> Host
        CHECK_CUDA(cudaMemcpy(new_data, data_, bytes, cudaMemcpyDeviceToHost));
    } else {
        CHECK_CUDA(cudaMalloc(&new_data, bytes));
        // Host -> Device
        CHECK_CUDA(cudaMemcpy(new_data, data_, bytes, cudaMemcpyHostToDevice));
    }

    // 释放旧内存并更新
    free();
    data_ = new_data;
    device_ = target_device;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape, Device device) {
    Tensor t(shape, device);
    if (device == Device::CPU) {
        std::memset(t.data_, 0, t.size_bytes());
    } else {
        CHECK_CUDA(cudaMemset(t.data_, 0, t.size_bytes()));
    }
    return t;
}

Tensor Tensor::rand(const std::vector<size_t>& shape, Device device) {
    // 先在 CPU 生成，如果需要 GPU 再转过去 (简单实现)
    Tensor t(shape, Device::CPU);
    float* ptr = static_cast<float*>(t.data_);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < t.numel_; ++i) {
        ptr[i] = static_cast<float>(dis(gen));
    }

    if (device == Device::CUDA) {
        t.to(Device::CUDA);
    }
    return t;
}

void Tensor::print(size_t max_elements) const {
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i] << (i < shape_.size() - 1 ? ", " : "");
    }
    std::cout << "], device=" << (device_ == Device::CPU ? "CPU" : "CUDA") << ")\nData: ";

    // 如果是 CUDA，先拷贝回 CPU 打印
    if (device_ == Device::CUDA) {
        // 创建一个临时 CPU tensor
        // 这里为了简单，手动拷贝前几个元素
        size_t count = std::min(numel_, max_elements);
        std::vector<float> temp(count);
        CHECK_CUDA(cudaMemcpy(temp.data(), data_, count * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (size_t i = 0; i < count; ++i) {
            std::cout << std::fixed << std::setprecision(4) << temp[i] << " ";
        }
    } else {
        const float* ptr = static_cast<const float*>(data_);
        for (size_t i = 0; i < std::min(numel_, max_elements); ++i) {
            std::cout << std::fixed << std::setprecision(4) << ptr[i] << " ";
        }
    }
    
    if (numel_ > max_elements) std::cout << "...";
    std::cout << std::endl;
}
