#include <iostream>
#include "tensor.h"
#include "ops.h"

int main() {
    try {
        std::cout << "=== Nano-LLM: Tensor & Ops Test ===" << std::endl;

        // 1. MatMul Test
        std::cout << "\n[1] Testing CPU MatMul..." << std::endl;
        // A: [2, 3]
        auto A = Tensor::zeros({2, 3}, Device::CPU);
        float* pA = static_cast<float*>(A.data());
        pA[0]=1; pA[1]=2; pA[2]=3;
        pA[3]=4; pA[4]=5; pA[5]=6;
        
        // B: [3, 2]
        auto B = Tensor::zeros({3, 2}, Device::CPU);
        float* pB = static_cast<float*>(B.data());
        pB[0]=1; pB[1]=2;
        pB[2]=3; pB[3]=4;
        pB[4]=5; pB[5]=6;

        // C: [2, 2]
        auto C = Tensor::zeros({2, 2}, Device::CPU);
        
        ops::matmul(A, B, C);
        
        std::cout << "A:\n"; A.print();
        std::cout << "B:\n"; B.print();
        std::cout << "C (Result):\n"; C.print();

        // Expected C:
        // [1*1+2*3+3*5, 1*2+2*4+3*6] = [1+6+15, 2+8+18] = [22, 28]
        // [4*1+5*3+6*5, 4*2+5*4+6*6] = [4+15+30, 8+20+36] = [49, 64]

        // 2. Softmax Test
        std::cout << "\n[2] Testing CPU Softmax..." << std::endl;
        auto S = Tensor::zeros({1, 3}, Device::CPU);
        float* pS = static_cast<float*>(S.data());
        pS[0] = 1.0f; pS[1] = 2.0f; pS[2] = 3.0f;
        
        std::cout << "Input:\n"; S.print();
        ops::softmax(S);
        std::cout << "Softmax:\n"; S.print();
        // Expected: exp(1)/sum, exp(2)/sum, exp(3)/sum
        // sum = e^1 + e^2 + e^3 ≈ 2.718 + 7.389 + 20.085 ≈ 30.192
        // e^1/sum ≈ 0.090
        // e^2/sum ≈ 0.244
        // e^3/sum ≈ 0.665

        std::cout << "\nTest Passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
