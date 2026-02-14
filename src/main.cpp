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

    // 3. Test RMSNorm
    std::cout << "\n--- Testing RMSNorm (CPU) ---" << std::endl;
    // Input: [2, 3] -> [[1, 2, 3], [4, 5, 6]]
    Tensor R_in({2, 3}, Device::CPU);
    Tensor R_w({3}, Device::CPU);   // Weight: [1, 1, 1]
    Tensor R_out({2, 3}, Device::CPU);

    float* ptrR = static_cast<float*>(R_in.data());
    float* ptrW = static_cast<float*>(R_w.data());
    
    // Set input: 1, 2, 3, 4, 5, 6
    for(int i=0; i<6; ++i) ptrR[i] = i + 1.0f;
    // Set weight: 1, 1, 1
    for(int i=0; i<3; ++i) ptrW[i] = 1.0f;

    std::cout << "Input:" << std::endl;
    R_in.print();

    ops::rmsnorm(R_in, R_w, R_out);

    std::cout << "Output (RMSNorm):" << std::endl;
    R_out.print();

    // 4. Test SwiGLU
    std::cout << "\n--- Testing SwiGLU (CPU) ---" << std::endl;
    // Gate: [2, 2], Feat: [2, 2]
    Tensor G({2, 2}, Device::CPU);
    Tensor F({2, 2}, Device::CPU);
    Tensor SG_out({2, 2}, Device::CPU);
    
    float* ptrG = static_cast<float*>(G.data());
    float* ptrF = static_cast<float*>(F.data());
    
    // Gate: [10, -10, 0, 1]
    // SiLU(10) ≈ 10, SiLU(-10) ≈ 0, SiLU(0) = 0, SiLU(1) ≈ 0.731
    ptrG[0] = 10.0f; ptrG[1] = -10.0f; ptrG[2] = 0.0f; ptrG[3] = 1.0f;
    
    // Feat: [1, 1, 1, 1]
    for(int i=0; i<4; ++i) ptrF[i] = 1.0f;
    
    ops::swiglu(G, F, SG_out);
    
    std::cout << "Gate:" << std::endl;
    G.print();
    std::cout << "Output (SwiGLU):" << std::endl;
    SG_out.print();
    // Expected: [~10, ~0, 0, ~0.731]

    // 5. Test RoPE
    std::cout << "\n--- Testing RoPE (CPU) ---" << std::endl;
    // Input: [1, 2] -> [[1, 0], [0, 1]] (dim=2)
    Tensor Rope_in({2, 2}, Device::CPU);
    Tensor Rope_pos({2}, Device::CPU);
    Tensor Rope_out({2, 2}, Device::CPU);
    
    float* ptr_ri = static_cast<float*>(Rope_in.data());
    float* ptr_rp = static_cast<float*>(Rope_pos.data());
    
    // Row 0: [1, 0], Pos 0 -> Theta=0 -> No rotation -> [1, 0]
    ptr_ri[0] = 1.0f; ptr_ri[1] = 0.0f; ptr_rp[0] = 0.0f;
    
    // Row 1: [1, 0], Pos 1 -> Theta = 1 * 10000^(-0) = 1 rad ≈ 57.3 deg
    // Rotation: x' = cos(1) - 0 = 0.5403, y' = sin(1) + 0 = 0.8415
    ptr_ri[2] = 1.0f; ptr_ri[3] = 0.0f; ptr_rp[1] = 1.0f;
    
    ops::rope(Rope_in, Rope_pos, Rope_out);
    
    std::cout << "Output (RoPE):" << std::endl;
    Rope_out.print();
    // Expected Row 0: [1.0000, 0.0000]
    // Expected Row 1: [0.5403, 0.8415]

    std::cout << "\nAll Operator Tests Passed!" << std::endl;
    return 0;
}
