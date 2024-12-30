cmake_minimum_required(VERSION 3.18)
project(flash_attn_cxx LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(FLASH_ATTENTION_FORCE_BUILD OFF CACHE BOOL "Force build the flash attention library")
set(FLASH_ATTENTION_SKIP_CUDA_BUILD OFF CACHE BOOL "Skip building the CUDA part of the flash attention library")
set(FLASH_ATTENTION_FORCE_CXX11_ABI OFF CACHE BOOL "Force the C++11 ABI")
set(FLASH_ATTENTION_TRITON_AMD_ENABLE OFF CACHE BOOL "Enable the AMD GPU support for the Triton backend")

set(CAFFE2_USE_CUDNN ON)
set(CUDA_ARCH "sm_80;sm_90")

set(Python_ROOT_DIR "${CMAKE_SOURCE_DIR}/.venv" CACHE PATH "Path to the Python installation")

find_package(Python COMPONENTS Interpreter Development)
message(STATUS "Python interpreter: ${Python_EXECUTABLE}")
message(STATUS ${Python_INCLUDE_DIRS})

include_directories(${Python_INCLUDE_DIRS})
set(Torch_DIR ${Python_ROOT_DIR}/Lib/site-packages/torch/share/cmake/Torch CACHE PATH "Path to the Torch installation")

find_package(Torch REQUIRED)
message(STATUS ${CUDA_nvrtc_LIBRARY})

include_directories(.)
include_directories(csrc/flash_attn csrc/flash_attn/src csrc/cutlass/include)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 4")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code=arch=compute_75,code=[compute_75,sm_75]")
add_compile_definitions("__CUDA_NO_HALF_OPERATORS__")
add_compile_definitions("__CUDA_NO_HALF_CONVERSIONS__")
add_compile_definitions("__CUDA_NO_HALF2_OPERATORS__")
add_compile_definitions("__CUDA_NO_BFLOAT16_CONVERSIONS__")


add_library(flash_attn_2_cuda SHARED
csrc/flash_attn/flash_api.cpp
csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu
csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim32_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim32_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim64_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim96_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim96_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim160_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim160_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim192_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim192_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim256_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_hdim256_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim160_fp16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim160_bf16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim32_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim32_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim64_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim64_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim96_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim96_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim128_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim128_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim160_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim160_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim192_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim192_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim256_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_bwd_hdim256_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim160_fp16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim160_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim160_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim160_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_causal_sm80.cu 
csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu 
)

target_link_libraries(flash_attn_2_cuda ${TORCH_LIBRARIES})
link_directories(${Python_LIBRARY_DIRS})
target_link_libraries(flash_attn_2_cuda ${Python_LIBRARIES})

# 设置 CUDA 架构
set_target_properties(flash_attn_2_cuda PROPERTIES
    CUDA_ARCHITECTURES "80;90"
)
