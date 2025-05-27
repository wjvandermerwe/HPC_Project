/* envmap.cu / .cpp ----------------------------------------------------- */

#include "deviceHelpers.cuh"
#include "stb_image.h"

static cudaTextureObject_t g_envTex = 0;        // handle queried by kernel

__device__ __forceinline__
vector3* vec3_normalize(vector3* self)
{
    float mag = sqrtf(self->x * self->x +
                      self->y * self->y +
                      self->z * self->z);

    if (mag == 0.0f)        // avoid div-by-zero
        return self;

    float inv = 1.0f / mag;
    self->x *= inv;
    self->y *= inv;
    self->z *= inv;
    return self;
}

static __device__ inline RGBColorF sample_env(cudaTextureObject_t tex, vec3 dir)
{
    /* normalise once */
    dir = v_unit(dir);

    float u = 0.5f + atan2f(dir.z, dir.x) * 0.159154943f;   // 1 / (2π)
    float v = acosf(dir.y)          * 0.318309886f;         // 1 / π

    float4 t = tex2D<float4>(tex, u, v);
    return { t.x, t.y, t.z };
}

__constant__ SphereGPU d_spheres[MAX_SPHERES];

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

struct Hit {
    HitRecord      rec;
    const SphereGPU* surf;   // pointer to the sphere we hit
};

__device__ bool hit_scene(const Ray& r,
                          const SphereGPU* sph, int ns,
                          float tmin, float tmax,
                          Hit& out)
{
    bool  hitAnything = false;
    float closest     = tmax;

    for (int i = 0; i < ns; ++i)
    {
        HitRecord rec;
        if (hit_sphere(sph[i], r, tmin, closest, rec))
        {
            hitAnything = true;
            closest     = rec.distanceFromOrigin;
            out.rec     = rec;
            out.surf    = &sph[i];
        }
    }
    return hitAnything;
}


#define MAX_SPHERES 64
#define GROUND_Y 0.0f

__constant__ RGBColorF c_bg_A = { 0.95f, 0.95f, 0.98f };   // light square
__constant__ RGBColorF c_bg_B = { 0.15f, 0.18f, 0.22f };   // dark square

__device__ RGBColorF shared_ray_color(Ray r,
                               const SphereGPU *spheres, int ns,
                               uint32_t &rngState, int maxDepth,
                               cudaTextureObject_t envTex)          // <-- extra arg
{
    RGBColorF acc   = {0,0,0};
    RGBColorF atten = {1,1,1};


    constexpr float kEps     = 1e-4f;  // shadow-bias / self-hit guard

    for (int depth = 0; depth < maxDepth; ++depth)
    {
        /* -------------------------------- 1.  intersection test ------ */
        HitRecord rec;  rec.valid = false;
        const SphereGPU *hitSphere = nullptr;
        float nearest = 1e30f;

        /* ground-plane candidate */
        bool  planeHit = false;
        float tPlane   = 1e30f;
        if (r.direction.y < -kEps) {
            tPlane = (GROUND_Y - r.origin.y) / r.direction.y;
            if (tPlane > kEps) {             // plane is in front of camera
                nearest  = tPlane;
                planeHit = true;
            }
        }

        /* sphere candidates */
        for (int i = 0; i < ns; ++i)
            if (hit_sphere(spheres[i], r, kEps, nearest, rec)) {
                nearest   = rec.distanceFromOrigin;
                hitSphere = &spheres[i];
                planeHit  = false;           // sphere wins if nearer
            }

        /* ------------------------------ 2.  shading  ---------------- */
        if (planeHit) {                      // ✔ ground checkerboard
            vec3 p = v_add(r.origin, v_mul(r.direction, tPlane));
            bool dark       = ( (int)floorf(p.x) + (int)floorf(p.z) ) & 1;
            RGBColorF tile  = dark ? c_bg_B : c_bg_A;

            acc.r += atten.r * tile.r;
            acc.g += atten.g * tile.g;
            acc.b += atten.b * tile.b;
            break;                           // path ends on the ground
        }

        if (!rec.valid) {                    // ✔ miss → sky
            RGBColorF sky;
            if (envTex)                      // HDR skybox path
                sky = sample_env(envTex, r.direction);
            else {                           // simple gradient sky
                vec3 u  = v_unit(r.direction);
                float t = 0.5f * (u.y + 1.f);
                sky = { (1-t) + 0.5f*t,
                        (1-t) + 0.7f*t,
                        (1-t) + 1.0f*t };
            }
            acc.r += atten.r * sky.r;
            acc.g += atten.g * sky.g;
            acc.b += atten.b * sky.b;
            break;
        }

        /* ✔ we hit a sphere */
        const SphereGPU *s = hitSphere;

        if (s->matType == LAMBERTIAN) {
            vec3 dir = v_add(rec.normal, random_vec3(rngState));
            r        = { rec.point, v_unit(dir) };
            atten.r *= s->albedo.r;
            atten.g *= s->albedo.g;
            atten.b *= s->albedo.b;
            continue;
        }

        if (s->matType == METAL) {
            vec3 refl = reflect(v_unit(r.direction), rec.normal);
            refl      = v_add(refl, v_mul(random_vec3(rngState), s->fuzz));
            if (v_dot(refl, rec.normal) <= 0.f) break;  // absorbed
            r        = { rec.point, v_unit(refl) };
            atten.r *= s->albedo.r;
            atten.g *= s->albedo.g;
            atten.b *= s->albedo.b;
            continue;
        }

        if (s->matType == DIELECTRIC) {
            float eta   = rec.frontFace ? (1.f / s->ir) : s->ir;
            float cosIn = fminf(-v_dot(v_unit(r.direction), rec.normal), 1.f);
            float reflP = schlick(cosIn, eta);

            vec3 dir;
            if (rng_next(rngState) < reflP ||
                !refract(v_unit(r.direction), rec.normal, eta, dir))
                dir = reflect(v_unit(r.direction), rec.normal);

            r = { rec.point, v_unit(dir) };
            continue;
        }

        break;        // any other material → absorb
    }
    return acc;
}
__global__ void render_kernel(RGBColorU8 *out,
                              int w, int h,
                              Camera cam,
                              int ns,
                              int spp, int maxDepth,
                              cudaTextureObject_t texture_object)
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
        RGBColorF c = shared_ray_color(r, s_sph, ns, rng, maxDepth, texture_object);  // shared cache
        col.r += c.r;  col.g += c.g;  col.b += c.b;
    }

    /* write back */
    float scale = rsqrtf((float)spp);                // faster than sqrtf+divide
    col.r = sqrtf(col.r) * scale;
    col.g = sqrtf(col.g) * scale;
    col.b = sqrtf(col.b) * scale;

    out[(h-1-y)*w + x] = coloru8_createf_gpu(col.r, col.g, col.b);
}

#include <iostream>

int main(int argc,char** argv)
{
    /* ------------- scene on host ---------------- */
    SceneContext   host = prepare_world();
    RenderCtx      dev  = render_init_cuda(host.spheres,
                                           host.width, host.height);

    /* ----------- optional HDR skybox ------------ */
    cudaTextureObject_t envTex = 0;
    if (argc == 3) {                       // “hdr.exr” or “probe.hdr”
        int w,h,comp; float* img;
        float *hdrData = nullptr;
        if (load_hdr(argv[2], &hdrData, &w, &h, &comp)) {
            std::vector<float4> img(w*h);
            for (int i = 0; i < w*h; ++i) {
                float *p = hdrData + comp*i;
                img[i] = make_float4(p[0], p[1], p[2], comp > 3 ? p[3] : 1.0f);
            }
            uploadScene(dev.d_spheres, dev.ns);      // copy to __constant__
            envTex = uploadEnvMap(w, h, img.data());
            stbi_image_free(hdrData);
        }
    } else
        uploadScene(dev.d_spheres, dev.ns);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    /* ------------- kernel launch --------------- */
    dim3 blk(8,8);
    dim3 grd( (host.width  + blk.x-1)/blk.x,
              (host.height + blk.y-1)/blk.y );

    size_t shmem = dev.ns * sizeof(SphereGPU);
    render_kernel<<<grd, blk, shmem>>>(
        dev.d_out,
        host.width, host.height,
        host.cam,
        dev.ns,
        host.spp,
        host.maxDepth,
        envTex);                                   // <- extra arg

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float msFinish;
    cudaEventElapsedTime(&msFinish, start, stop);
    std::cout << "[gpu_render] took " << msFinish << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    render_finish_cuda(dev, host.framebuffer, "out.ppm");
    return 0;
}