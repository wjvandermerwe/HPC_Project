cmake_minimum_required(VERSION 3.18)
project(cuRaytracer-base LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# adjust this if you have headers in a different spot
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(cuRaytracer-base cuRaytracer-base.cu)

target_link_libraries(cuRaytracer-base PRIVATE CUDA::cudart)
