#ifndef UTIL_H
#define UTIL_H

#include "hypatiaINC.h"
#include "types.h"

#include <stdint.h>
#include <stdbool.h>

#if defined(__CUDACC__)
  // when compiling with NVCC
  #define CUDA_FN __host__ __device__
#else
  // plain host compiler
  #define CUDA_FN
#endif
CUDA_FN extern CFLOAT util_floatClamp(CFLOAT c, CFLOAT lower, CFLOAT upper);
CUDA_FN extern uint8_t util_uint8Clamp(uint8_t c, uint8_t lower, uint8_t upper);

// not in use
CUDA_FN extern uint32_t util_randomRange(uint32_t lower, uint32_t upper);
CUDA_FN extern vec3 util_randomUnitSphere();
CUDA_FN extern vec3 util_randomUnitVector();
CUDA_FN extern CFLOAT util_randomFloat(CFLOAT lower, CFLOAT upper);
CUDA_FN extern vec3 util_vec3Reflect(vec3 v,vec3 n);
CUDA_FN extern bool util_isVec3Zero(vec3 v);
CUDA_FN extern vec3 util_randomUnitDisk();

#endif

