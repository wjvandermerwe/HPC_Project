cmake_minimum_required(VERSION 3.31)
project(CudaRayTracer LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 26)
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SRC_C "${CMAKE_SOURCE_DIR}/sources/*.c")

add_executable(CudaRayTracer
        cuRaytracer-base.cu
        ${SRC_C})

set_target_properties(CudaRayTracer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
