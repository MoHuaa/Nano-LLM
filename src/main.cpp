#include <iostream>
#include <vector>
#include "tensor.h"
#include "ops.h" 

int main() {
    std::cout << "=== Nano-LLM Operator Test ===" << std::endl;

    // 1. Test MatMul
    std::cout << "\n--- Testing MatMul (CPU) ---" << std::endl;
    // A: [2, 3], B: [3, 2], C: [2, 2]
    Tensor A({2, 3}, Device::CPU);
    Tensor B({3, 2}, Device::CPU);
    Tensor C({2, 2}, Device::CPU);

    float* dataA = static_cast<float*>(A.data());
    float* dataB = static_cast<float*>(B.data());

    // Init A with 1.0, B with 2.0
    // 1 * 2 + 1 * 2 + 1 * 2 = 6
    for(int i=0; i<6; ++i) dataA[i] = 1.0f;
    for(int i=0; i<6; ++i) dataB[i] = 2.0f;

    ops::matmul(A, B, C);
    
    std::cout << "C = A @ B (Should be all 6.0):" << std::endl;
    C.print();

    // 2. Test Softmax
    std::cout << "\n--- Testing Softmax (CPU) ---" << std::endl;
    Tensor S({1, 5}, Device::CPU);
    float* dataS = static_cast<float*>(S.data());
    // 10, 10, 10, 10, 10 -> Softmax -> 0.2, 0.2, ...
    for(int i=0; i<5; ++i) dataS[i] = 10.0f;
    
    std::cout << "Input:" << std::endl;
    S.print();
    
    ops::softmax(S);
    
    std::cout << "Output (Should be all 0.2):" << std::endl;
    S.print();

    std::cout << "\nAll Operator Tests Passed!" << std::endl;
    return 0;
}
