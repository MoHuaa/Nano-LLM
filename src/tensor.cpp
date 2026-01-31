#include "tensor.h"
#include <numeric>   // for std::accumulate
#include <iostream>
#include <iomanip>   // for std::setprecision
#include <cstring>   // for std::memcpy
#include <stdexcept> // for std::runtime_error
#include <cuda_runtime.h> // CUDA Runtime API

// 修正 1: 宏定义改为接受两个参数 (call, msg)
// 或者在宏内部忽略 msg，但为了保留错误信息，建议如下写法：
#define CHECK_CUDA(call, msg) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (%s) at %s:%d\n", msg, cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

Tensor::Tensor(std::vector<size_t> shape, Device device) {
    shape_ = shape;
    numel_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
    device_ = device;
    allocate(); 
}

Tensor::~Tensor() {
    free();
}

Tensor::Tensor(Tensor&& other) noexcept {
    shape_ = std::move(other.shape_);
    numel_ = other.numel_;
    device_ = other.device_;
    data_ = other.data_; 
    
    other.data_ = nullptr;
    other.numel_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free(); 
        
        shape_ = std::move(other.shape_);
        numel_ = other.numel_;
        device_ = other.device_;
        data_ = other.data_;
        
        other.data_ = nullptr;
        other.numel_ = 0;
    }
    return *this; 
}

// 修正 2: 变量名拼写 date -> data_, device -> device_, Cuda -> CUDA
void Tensor::to(Device target_device){
    if(target_device == device_){ // device_
        return;
    }

    float* new_data = nullptr;
    size_t bytes = numel_ * sizeof(float); // numel_

    // 1. Allocate on target
    if(target_device == Device::CPU){
        new_data = new float[numel_];
    } else {
        CHECK_CUDA(cudaMalloc((void**)&new_data, bytes), "cudaMalloc failed");
    }

    // 2. Copy
    // 如果现在是在 GPU (device_ == CUDA)，且目标是 CPU
    if(device_ == Device::CUDA && target_device == Device::CPU){
        CHECK_CUDA(cudaMemcpy(new_data, data_, bytes, cudaMemcpyDeviceToHost), "D2H failed");
    } 
    // 如果现在是在 CPU，且目标是 GPU
    else if (device_ == Device::CPU && target_device == Device::CUDA) {
        CHECK_CUDA(cudaMemcpy(new_data, data_, bytes, cudaMemcpyHostToDevice), "H2D failed");
    }
    // (可选) GPU 到 GPU 的拷贝通常用 cudaMemcpyDeviceToDevice，暂时不处理

    // 3. Free old & Update
    free();
    data_ = new_data;
    device_ = target_device;
}

void Tensor::allocate(){
    if(device_ == Device::CPU){
        data_ = new float[numel_]; // numel_
    } else {
        size_t bytes = numel_ * sizeof(float);
        CHECK_CUDA(cudaMalloc((void**)&data_, bytes), "cudaMalloc failed");
    }
}

void Tensor::free(){
    // 修正 3: 安全性检查
    // 如果 data_ 已经是空，直接返回，防止重复释放
    if (data_ == nullptr) return;

    if(device_ == Device::CPU){
        delete[] data_; // date -> data_
    } else {
        CHECK_CUDA(cudaFree(data_), "cudaFree failed");
    }
    
    // 修正 4: 必须置空悬空指针！
    data_ = nullptr; 
}

void Tensor::print() const {
    // 建议改进：如果是 GPU，自动拷回来打印，而不是报错
    // 这里保留你的逻辑，但加上了必要的头文件引用
    if (device_ != Device::CPU) {
        // 如果想省事，可以在这里调用 const_cast<Tensor*>(this)->to(Device::CPU); 
        // 但这样会改变原数据位置，不太好。
        // 现在的报错也可以：
        std::cerr << "Warning: Cannot print GPU tensor directly. Move to CPU first." << std::endl;
        return;
    }
    
    std::cout << "Tensor: [";
    for (size_t i = 0; i < std::min(numel_, (size_t)10); ++i) { // 只打前10个防止刷屏
        std::cout << data_[i] << " ";
    }
    if (numel_ > 10) std::cout << "...";
    std::cout << "]" << std::endl;
}
