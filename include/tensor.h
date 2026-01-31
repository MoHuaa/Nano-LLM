#pragma once

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>

enum class Device {
    CPU,
    CUDA
};

enum class DType {
    Float32,
    Float16 // Planned
};

class Tensor {
public:
    Tensor(const std::vector<size_t>& shape, Device device = Device::CPU, DType dtype = DType::Float32);
    ~Tensor();

    // 禁止拷贝，只能移动 (为了管理 device 内存)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // 数据访问
    void* data() { return data_; }
    const void* data() const { return data_; }
    
    // 形状信息
    const std::vector<size_t>& shape() const { return shape_; }
    size_t numel() const { return numel_; }
    size_t size_bytes() const;
    Device device() const { return device_; }

    // 常用操作
    void to(Device target_device);
    static Tensor zeros(const std::vector<size_t>& shape, Device device = Device::CPU);
    static Tensor rand(const std::vector<size_t>& shape, Device device = Device::CPU); // 0-1 uniform

    // 打印部分数据用于调试
    void print(size_t max_elements = 10) const;

private:
    void allocate();
    void free();

    void* data_ = nullptr;
    std::vector<size_t> shape_;
    size_t numel_ = 0;
    Device device_;
    DType dtype_;
};
