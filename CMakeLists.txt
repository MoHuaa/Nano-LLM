cmake_minimum_required(VERSION 3.18)
project(Nano-LLM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置 CUDA 架构 (根据你的 5070 Ti，它是 Blackwell，通常兼容 Ampere/Ada，这里先设为 auto 或 86/89)
# RTX 40 series is arch=89, 50 series might be higher, but let's stick to native or common archs.
set(CMAKE_CUDA_ARCHITECTURES native)

# 开启编译器警告
if(MSVC)
    add_compile_options(/W3)
else()
    add_compile_options(-Wall -Wextra)
endif()

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 查找 CUDA 库
find_package(CUDAToolkit REQUIRED)

# 源文件
file(GLOB_RECURSE SOURCES 
    "src/*.cpp" 
    "src/*.cu"
)

# 如果没有源文件，先创建一个空的 main.cpp 避免 cmake 报错
if(NOT SOURCES)
    file(WRITE ${CMAKE_SOURCE_DIR}/src/main.cpp "#include <iostream>\nint main() { std::cout << \"Hello Nano-LLM\" << std::endl; return 0; }")
    set(SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")
endif()

# 创建可执行文件
add_executable(nano_llm_main ${SOURCES})

# 链接 CUDA 库
target_link_libraries(nano_llm_main PRIVATE CUDA::cudart CUDA::cublas)

# 设置 CUDA 编译选项
target_compile_options(nano_llm_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# 测试相关 (简单起见，暂时手动写 test main，后续可以集成 GTest)
enable_testing()
