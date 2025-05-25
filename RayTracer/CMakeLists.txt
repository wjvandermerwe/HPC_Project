cmake_minimum_required(VERSION 3.31)
project(CudaRayTracer LANGUAGES C CXX CUDA)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 26)
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SRC_C "${CMAKE_SOURCE_DIR}/sources/*.c")
file(GLOB SRC_CUDA "${CMAKE_SOURCE_DIR}/sources/*.cu")

function(add_raytracer exe main_cu)
    add_executable(${exe} ${main_cu} ${SRC_C} ${SRC_CUDA})
    set_target_properties(${exe} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
    target_link_libraries(${exe} PRIVATE CUDA::cudart_static)
endfunction()

# ---- three variants ---------------------------------------------------
add_raytracer(raytrace_base    cuRaytracer-base.cu)
add_raytracer(raytrace_shared  cuRaytracer-shared.cu)
add_raytracer(raytrace_improved cuRaytracer-improved.cu)
