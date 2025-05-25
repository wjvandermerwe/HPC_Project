//
// Created by johan on 25/05/2025.
//
#ifndef DEVICE_HELPERS_CUH
#define DEVICE_HELPERS_CUH
#pragma once
#include <cuda_runtime.h>
#include <cuda.h>

#if defined(__cplusplus) || defined(__CUDACC__)
#  ifndef restrict
#    define restrict __restrict__
#  endif
#endif
extern "C" {
#include "camera.h"
#include "helpers.h"
#include "sphere.h"
#include "types.h"
#include "util.h"
}

__host__ __device__ inline vec3  v_add (vec3 a, vec3 b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; }
__host__ __device__ inline vec3  v_sub (vec3 a, vec3 b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__host__ __device__ inline vec3  v_mul (vec3 a, float k){ return {a.x*k, a.y*k, a.z*k};   }
__host__ __device__ inline float v_dot (vec3 a, vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline float v_len2(vec3 a){ return v_dot(a,a); }
__host__ __device__ inline vec3  v_unit(vec3 a){ return v_mul(a, rsqrtf(v_len2(a))); }

struct SphereGPU {
    vec3       center;
    float      radius;
    int        matType;

    RGBColorF  albedo;
    float      fuzz;
    float      ir;
};

/* ---------- device helpers (declared; bodies live in .cu) ---------- */
static __device__ inline float rng_next(uint32_t &state)
{
    state = state * 1664525u + 1013904223u;
    return (state & 0x00FFFFFF) / 16777216.0f;
}
static __device__ __forceinline__
vec3 reflect(vec3 v, vec3 n)
{
    return v_sub(v, v_mul(n, 2.0f * v_dot(v,n)));
}

static __device__ __forceinline__
bool refract(vec3 v, vec3 n, float eta, vec3& refrOut)
{
    float cos_theta = fminf(-v_dot(v,n), 1.0f);
    vec3  r_out_perp  = v_mul(v_add(v, v_mul(n, cos_theta)), eta);
    float k = 1.0f - v_len2(r_out_perp);
    if (k < 0.0f) return false;                 // total internal reflection
    vec3  r_out_parallel = v_mul(n, -sqrtf(k));
    refrOut = v_add(r_out_perp, r_out_parallel);
    return true;
}
static __device__ __forceinline__
float schlick(float cosine, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}
__device__ vec3       random_vec3   (uint32_t &state);
__device__ vec3       random_in_unit_disk(uint32_t &state);
__device__ Ray        cam_getRay_gpu(const Camera*, float u, float v,
                                     uint32_t &state);
__device__ RGBColorF  ray_color     (Ray r, const SphereGPU *spheres,
                                     int ns, uint32_t &rng, int maxDepth);


static __host__ __device__ inline RGBColorU8 coloru8_createf_gpu(CFLOAT r, CFLOAT g, CFLOAT b) {
    return RGBColorU8{
        (uint8_t)fminf(r * 256.0f, 255.0f),
        (uint8_t)fminf(g * 256.0f, 255.0f),
        (uint8_t)fminf(b * 256.0f, 255.0f)
    };
}
/* ---------- kernels ------------------------------------------------ */
#define MAX_SPHERES 64
/* classic launch-per-pixel kernel */
__global__ void render_kernel(RGBColorU8 *out, int W,int H,
                              Camera cam,
                              const SphereGPU *d_spheres,int ns,
                              int spp,int maxDepth);
/* persistent-threads variant */
__device__ extern unsigned int d_nextPixel;      /* global pixel counter */
__global__ void render_persistent(RGBColorU8 *out,int W,int H,Camera cam,
                                  const SphereGPU *d_spheres,int ns,
                                  int spp,int maxDepth,
                                  cudaTextureObject_t envTex);

/* ---------- host-side context & helpers --------------------------- */
struct RenderCtx {
    SphereGPU  *d_spheres = nullptr;
    RGBColorU8 *d_out     = nullptr;
    int         ns        = 0;
    int         W = 0, H = 0;
};

/* allocate device buffers + upload scene */
RenderCtx render_init_cuda(const std::vector<Sphere> &cpuSpheres,
                           int W, int H);
struct SceneContext
{
    /* immutable render parameters */
    int        width   = 0;
    int        height  = 0;
    int        spp     = 0;      /* samples per pixel */
    int        maxDepth= 0;

    /* camera and geometry */
    Camera                 cam;
    std::vector<Sphere>    spheres;

    /* CPU framebuffer (malloc’ed inside, freed by caller) */
    RGBColorU8* framebuffer = nullptr;

    /* raw handles in case the app needs them later */
    ObjectLL*            world         = nullptr;
    DynamicStackAlloc*   dsaMaterials  = nullptr;
    DynamicStackAlloc*   dsaTextures   = nullptr;
    LinearAllocFC*       hitRecAlloc   = nullptr;
};

SceneContext prepare_world();
/* launch any kernel; example uses pixel-per-thread kernel */
void      render_launch_cuda(const RenderCtx &ctx,const Camera &cam,
                             int samplesPerPixel,int maxDepth);

void      render_finish_cuda(RenderCtx &ctx,
                             RGBColorU8 *hostPixels,
                             const char *ppmName);

void      uploadScene   (const SphereGPU *host,int n);               /* to const mem */
cudaTextureObject_t uploadEnvMap(int w,int h,const float4 *hostRGBA);
void      resetPixelCounter(int W,int H);                            /* for persistent */

#endif /* RENDER_CUDA_CUH */
