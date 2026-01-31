#pragma once

#include <vector>
#include <cstddef> // for size_t
#include <string>

// 设备类型枚举
enum class Device {
    CPU,
    CUDA
};

class Tensor {
public:
    // 构造函数
    // shape: 张量的形状，例如 {2, 3} 表示 2行3列
    // device: 初始存储设备，默认为 CPU
    Tensor(std::vector<size_t> shape, Device device = Device::CPU);

    // 析构函数：负责释放内存 (RAII)
    ~Tensor();

    // === 禁止拷贝 (Rule of 5) ===
    // 深度学习中张量通常很大，隐式拷贝极易导致性能问题和显存爆炸
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // === 允许移动 (Move Semantics) ===
    // 允许所有权转移，例如：Tensor a = create_tensor();
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // === 核心功能 ===
    
    // 数据迁移：将 Tensor 移动到指定设备 (CPU <-> CUDA)
    // 如果已经在该设备上，则什么都不做
    void to(Device target_device);

    // 获取原始数据指针
    void* data() { return data_; }
    const void* data() const { return data_; }

    // 获取形状信息
    const std::vector<size_t>& shape() const { return shape_; }
    
    // 获取元素总数
    size_t numel() const { return numel_; }
    
    // 获取当前设备
    Device device() const { return device_; }

    // 打印张量内容 (用于调试)
    // 如果数据在 GPU 上，会自动拷贝回 CPU 进行打印，但不改变原 Tensor 状态
    void print() const;

private:
    // 内部使用的内存分配函数
    void allocate();
    
    // 内部使用的内存释放函数
    void free();

    float* data_ = nullptr;        // 指向数据的原始指针
    std::vector<size_t> shape_;   // 形状
    size_t numel_ = 0;            // 元素总数 (缓存起来避免重复计算)
    Device device_;               // 当前所在的设备
};
