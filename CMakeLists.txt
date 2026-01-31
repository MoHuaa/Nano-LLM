cmake_minimum_required(VERSION 3.18)
project(Nano-LLM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES 
    "src/*.cpp" 
    "src/*.cu"
)

add_executable(nano_llm_main ${SOURCES})

target_link_libraries(nano_llm_main PRIVATE 
    CUDA::cudart
    CUDA::cublas
)

if(MSVC)
    target_compile_options(nano_llm_main PRIVATE /W3)
else()
    target_compile_options(nano_llm_main PRIVATE -Wall -Wextra)
    target_compile_options(nano_llm_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()
