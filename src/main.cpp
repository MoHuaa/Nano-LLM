#include <iostream>
#include <vector>
#include "tensor.h"
#include "ops.h" 

int main() {
    std::cout << "=== Nano-LLM Operator Test (CPU & GPU) ===" << std::endl;

    // ==========================================
    // 1. Test MatMul (CPU Only for now)
    // ==========================================
    std::cout << "\n--- Testing MatMul (CPU) ---" << std::endl;
    Tensor A({2, 3}, Device::CPU);
    Tensor B({3, 2}, Device::CPU);
    Tensor C({2, 2}, Device::CPU);

    float* dataA = static_cast<float*>(A.data());
    float* dataB = static_cast<float*>(B.data());

    for(int i=0; i<6; ++i) dataA[i] = 1.0f;
    for(int i=0; i<6; ++i) dataB[i] = 2.0f;

    ops::matmul(A, B, C);
    C.print(); // Should be all 6.0

    // ==========================================
    // 2. Test Softmax (CPU Only for now)
    // ==========================================
    std::cout << "\n--- Testing Softmax (CPU) ---" << std::endl;
    Tensor S({1, 5}, Device::CPU);
    float* dataS = static_cast<float*>(S.data());
    for(int i=0; i<5; ++i) dataS[i] = 10.0f;
    
    ops::softmax(S);
    S.print(); // Should be all 0.2

    // ==========================================
    // 3. Test RMSNorm (CPU vs GPU)
    // ==========================================
    std::cout << "\n--- Testing RMSNorm ---" << std::endl;
    Tensor R_in({2, 3}, Device::CPU);
    Tensor R_w({3}, Device::CPU);
    Tensor R_out({2, 3}, Device::CPU);

    float* ptrR = static_cast<float*>(R_in.data());
    float* ptrW = static_cast<float*>(R_w.data());
    for(int i=0; i<6; ++i) ptrR[i] = i + 1.0f;
    for(int i=0; i<3; ++i) ptrW[i] = 1.0f;

    // CPU Run
    ops::rmsnorm(R_in, R_w, R_out);
    std::cout << "[CPU] "; R_out.print();

    // GPU Run
    R_in.to(Device::CUDA);
    R_w.to(Device::CUDA);
    R_out.to(Device::CUDA);
    ops::rmsnorm(R_in, R_w, R_out);
    std::cout << "[GPU] "; R_out.print();

    // ==========================================
    // 4. Test SwiGLU (CPU vs GPU)
    // ==========================================
    std::cout << "\n--- Testing SwiGLU ---" << std::endl;
    Tensor G({2, 2}, Device::CPU);
    Tensor F({2, 2}, Device::CPU);
    Tensor SG_out({2, 2}, Device::CPU);
    
    float* ptrG = static_cast<float*>(G.data());
    float* ptrF = static_cast<float*>(F.data());
    
    ptrG[0] = 10.0f; ptrG[1] = -10.0f; ptrG[2] = 0.0f; ptrG[3] = 1.0f;
    for(int i=0; i<4; ++i) ptrF[i] = 1.0f;
    
    // CPU Run
    ops::swiglu(G, F, SG_out);
    std::cout << "[CPU] "; SG_out.print();

    // GPU Run
    G.to(Device::CUDA);
    F.to(Device::CUDA);
    SG_out.to(Device::CUDA);
    ops::swiglu(G, F, SG_out);
    std::cout << "[GPU] "; SG_out.print();

    // ==========================================
    // 5. Test RoPE (CPU vs GPU)
    // ==========================================
    std::cout << "\n--- Testing RoPE ---" << std::endl;
    Tensor Rope_in({2, 2}, Device::CPU);
    Tensor Rope_pos({2}, Device::CPU);
    Tensor Rope_out({2, 2}, Device::CPU);
    
    float* ptr_ri = static_cast<float*>(Rope_in.data());
    float* ptr_rp = static_cast<float*>(Rope_pos.data());
    
    ptr_ri[0] = 1.0f; ptr_ri[1] = 0.0f; ptr_rp[0] = 0.0f;
    ptr_ri[2] = 1.0f; ptr_ri[3] = 0.0f; ptr_rp[1] = 1.0f;
    
    // CPU Run
    ops::rope(Rope_in, Rope_pos, Rope_out);
    std::cout << "[CPU] "; Rope_out.print();

    // GPU Run
    Rope_in.to(Device::CUDA);
    Rope_pos.to(Device::CUDA);
    Rope_out.to(Device::CUDA);
    ops::rope(Rope_in, Rope_pos, Rope_out);
    std::cout << "[GPU] "; Rope_out.print();

    std::cout << "\nAll Operator Tests Passed!" << std::endl;
    return 0;
}
