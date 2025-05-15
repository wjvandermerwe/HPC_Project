cmake_minimum_required(VERSION 3.31)
project(CudaRayTracer LANGUAGES CXX CUDA)


set(CMAKE_CUDA_STANDARD 26)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SRC_CUDA   "${CMAKE_SOURCE_DIR}/sources/device_wrappers/*.cu")
#file(GLOB HOST_SOURCES "${CMAKE_SOURCE_DIR}/sources/*.c")

add_executable(CudaRayTracer
        cuRaytracer-base.cu
        ${SRC_CUDA}
        ${HOST_SOURCES})
#set_source_files_properties(${SRC_CUDA} PROPERTIES LANGUAGE CUDA)
#set_source_files_properties(${HOST_SOURCES} PROPERTIES LANGUAGE C)
set_target_properties(CudaRayTracer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
