/* envmap.cu / .cpp ----------------------------------------------------- */
#include <cuda_runtime.h>

static cudaTextureObject_t g_envTex = 0;        // handle queried by kernel

static __device__ inline RGBColorF sample_env(cudaTextureObject_t tex, vec3 dir)
{
    /* normalise once */
    dir = vec3_normalize(dir);

    float u = 0.5f + atan2f(dir.z, dir.x) * 0.159154943f;   // 1 / (2π)
    float v = acosf(dir.y)          * 0.318309886f;         // 1 / π

    float4 t = tex2D<float4>(tex, u, v);
    return { t.x, t.y, t.z };
}

void uploadScene(const SphereGPU* h, int n)
{
    cudaMemcpyToSymbol(d_spheres, h, n*sizeof(SphereGPU));
}

cudaTextureObject_t uploadEnvMap(int w, int h, const float4* hostRGBA)
{
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float4>();

    cudaArray_t cuArr;
    cudaMallocArray(&cuArr, &fmt, w, h);
    cudaMemcpy2DToArray(cuArr, 0, 0,
                        hostRGBA, w*sizeof(float4),
                        w*sizeof(float4), h,
                        cudaMemcpyHostToDevice);

    cudaResourceDesc res = {};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = cuArr;

    cudaTextureDesc tex = {};
    tex.addressMode[0] = tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode  = cudaFilterModeLinear;
    tex.readMode    = cudaReadModeElementType;
    tex.normalizedCoords = 1;          // (u,v) ∈ [0,1]

    cudaTextureObject_t texObj;
    cudaCreateTextureObject(&texObj, &res, &tex, nullptr);

    g_envTex = texObj;                 // stash for launch parameters
    return texObj;
}

__device__ RGBColorF ray_color(const Ray& r,
                               const SphereGPU* s, int ns,
                               uint32_t& rng, int depth,
                               cudaTextureObject_t envTex)       // +1
{
    if (depth <= 0) return {0,0,0};

    Hit h;
    if (hit_scene(r, s, ns, 0.001f, 1e20f, h))
    {
        Ray scattered;  RGBColorF attenuation;
        if (scatter(r, h, rng, attenuation, scattered))
        {
            RGBColorF c = ray_color(scattered, s, ns, rng, depth-1, envTex);
            return { attenuation.r * c.r,
                     attenuation.g * c.g,
                     attenuation.b * c.b };
        }
        return {0,0,0};
    }

    /* -------------- SKY ---------------- */
    return sample_env(envTex, r.dir);          // <-- new line
}


#define MAX_SPHERES 64

__constant__ SphereGPU d_spheres[MAX_SPHERES];

__global__ void render_kernel(RGBColorU8 *out,
                              int w, int h,
                              Camera cam,
                              int ns,
                              int spp, int maxDepth)
{
    extern __shared__ SphereGPU s_sph[];

    /* first warp only pulls constant → shared */
    for(int i = threadIdx.x; i < ns; i += blockDim.x)
        s_sph[i] = d_spheres[i];
    __syncthreads();                                 // ensure cache ready

    /* pixel coordinate */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    /* per-thread RNG state in registers */
    uint32_t rng = 1234u ^ (y*w + x);
    RGBColorF col = {0,0,0};

    /* main sampling loop */
    #pragma unroll
    for (int s = 0; s < spp; ++s)
    {
        float u = (x + rng_next(rng)) / (w - 1.f);
        float v = (y + rng_next(rng)) / (h - 1.f);

        Ray r = cam_getRay_gpu(&cam, u, v, rng);
        RGBColorF c = ray_color(r, s_sph, ns, rng, maxDepth);  // shared cache
        col.r += c.r;  col.g += c.g;  col.b += c.b;
    }

    /* write back */
    float scale = rsqrtf((float)spp);                // faster than sqrtf+divide
    col.r = sqrtf(col.r) * scale;
    col.g = sqrtf(col.g) * scale;
    col.b = sqrtf(col.b) * scale;

    out[(h-1-y)*w + x] = coloru8_createf_gpu(col.r, col.g, col.b);
}



