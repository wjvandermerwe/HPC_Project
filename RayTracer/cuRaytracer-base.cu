/***********************************************************************
 *  raytracer_cuda.cu  – GPU back-end for the existing CPU path tracer  *
 *                                                                    *
 *  – 1 thread   = 1 pixel × 1 sample                                  *
 *  – global-mem buffers only (no shared mem / textures)               *
 *  – stubbed device versions of the handful of “CPU-only” helpers     *
 **********************************************************************/
#if defined(__cplusplus) || defined(__CUDACC__)
#  ifndef restrict
#    define restrict __restrict__
#  endif
#endif
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <vector>
#include "camera.h"
#include "sphere.h"
#include "types.h"
#include "util.h"

/* ---------- 1. Tiny device-side math helpers ----------------------- */

__device__ inline vec3  v_add   (vec3 a, vec3 b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; }
__device__ inline vec3  v_sub   (vec3 a, vec3 b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__device__ inline vec3  v_mul   (vec3 a, float k){ return {a.x*k, a.y*k, a.z*k};   }
__device__ inline float v_dot   (vec3 a, vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ inline float v_len2  (vec3 a){ return v_dot(a,a); }
__device__ inline vec3  v_unit  (vec3 a){ float il = rsqrtf(v_len2(a)); return v_mul(a,il); }
__device__ inline float rng_next(uint32_t &state)
{
    state = state * 1664525u + 1013904223u;
    return (state & 0x00FFFFFF) / 16777216.0f;
}

__device__ inline vec3 random_vec3(uint32_t &rngState) {
    return {
        rng_next(rngState),
        rng_next(rngState),
        rng_next(rngState)
    };
}
/* ---------- 2. Very small LCG – one RNG state per thread ----------- */
// Device‐side random point in unit disk (Box–Muller variant)
__device__ vec3 random_in_unit_disk(uint32_t &state) {
    vec3 p;
    float x, y;
    do {
        x = 2.0f * rng_next(state) - 1.0f;
        y = 2.0f * rng_next(state) - 1.0f;
        p = { x, y, 0.0f };
    } while (x*x + y*y >= 1.0f);
    return p;
}

// GPU version of cam_getRay; pass in your per‐thread rngState
__device__ Ray cam_getRay_gpu(const Camera * __restrict__ cam,
                              float u, float v,
                              uint32_t &rngState)
{
    // sample aperture
    vec3 rd = random_in_unit_disk(rngState);
    rd.x *= cam->lensRadius;
    rd.y *= cam->lensRadius;

    // offset = u_vec * rd.x + v_vec * rd.y
    vec3 offset = {
        cam->u.x * rd.x + cam->v.x * rd.y,
        cam->u.y * rd.x + cam->v.y * rd.y,
        cam->u.z * rd.x + cam->v.z * rd.y
    };

    // origin with aperture offset
    vec3 orig = {
        cam->origin.x + offset.x,
        cam->origin.y + offset.y,
        cam->origin.z + offset.z
    };

    // compute target on viewport
    vec3 horiz = cam->horizontal;
    horiz.x *= u; horiz.y *= u; horiz.z *= u;
    vec3 vert  = cam->vertical;
    vert.x  *= v; vert.y  *= v; vert.z  *= v;

    vec3 dir = {
        cam->lowerLeftCorner.x + horiz.x + vert.x - cam->origin.x - offset.x,
        cam->lowerLeftCorner.y + horiz.y + vert.y - cam->origin.y - offset.y,
        cam->lowerLeftCorner.z + horiz.z + vert.z - cam->origin.z - offset.z
    };

    return Ray{ orig, v_unit(dir) };
}

/* ---------- 3. Device-side material & hit routines (trimmed) ------- */
/* NOTE: only diffuse + metal implemented to keep snippet short.       */
/* Extend exactly like the CPU versions if you need the rest.          */

struct SphereGPU
{
    vec3  center;
    float radius;
    int   matType;      // 0=LAMBERT, 1=METAL, 2=DIELECTRIC
    RGBColorF albedo;   // lambert & metal
    float fuzz;         // metal
    float ir;           // dielectric
};



__device__ bool hit_sphere(const SphereGPU &s, const Ray &r,
                           float tmin, float tmax, HitRecord &rec)
{
    vec3 oc = v_sub(r.origin, s.center);
    float a  = v_len2(r.direction);
    float half_b = v_dot(oc, r.direction);
    float c  = v_len2(oc) - s.radius*s.radius;
    float d  = half_b*half_b - a*c;
    if (d < 0) return false;
    float sqrt_d = sqrtf(d);

    float root = (-half_b - sqrt_d) / a;
    if (root < tmin || root > tmax)
    {
        root = (-half_b + sqrt_d) / a;
        if (root < tmin || root > tmax) return false;
    }
    rec.distanceFromOrigin = root;
    rec.point  = v_add(r.origin, v_mul(r.direction, root));
    rec.normal = v_mul(v_sub(rec.point, s.center), 1.0f/s.radius);
    rec.frontFace = v_dot(r.direction, rec.normal) < 0.0f;
    if(!rec.frontFace) rec.normal = v_mul(rec.normal,-1.0f);
    rec.valid = true;
    // rec.matId = 0;            // not used (we inline scatter below)
    // rec.uMeta = &s;           // point to sphere to know material
    return true;
}

/* ---------- 4. Recursive colour (iterative version to avoid stack) --*/

__device__ RGBColorF ray_color(Ray r,
                               const SphereGPU *spheres, int ns,
                               uint32_t &rngState, int maxDepth)
{
    RGBColorF acc = {0,0,0}, atten = {1,1,1};
    for(int depth=0; depth<maxDepth; ++depth)
    {
        HitRecord rec; rec.valid = false;
        float nearest = 1e30f;
        /* find nearest hit */
        for(int i=0;i<ns;++i)
            if(hit_sphere(spheres[i], r, 0.0001f, nearest, rec))
                nearest = rec.distanceFromOrigin;

        if(!rec.valid)
        {   /* background: simple gradient      */
            vec3 unit = v_unit(r.direction);
            float t = 0.5f*(unit.y+1.f);
            RGBColorF sky = { (1.f-t)+0.5f*t, (1.f-t)+0.7f*t, (1.f-t)+1.0f*t };
            acc = { acc.r + atten.r*sky.r,
                    acc.g + atten.g*sky.g,
                    acc.b + atten.b*sky.b };
            break;
        }

        const SphereGPU *s = static_cast<const SphereGPU*>(rec.uMeta);

        if(s->matType == 0) /* diffuse */
        {
            vec3 dir = v_add(rec.normal,random_vec3(rngState));
            r = {rec.point, v_unit(dir)};
            atten = { atten.r*s->albedo.r,
                      atten.g*s->albedo.g,
                      atten.b*s->albedo.b };
            continue;
        }
        else if(s->matType == 1) /* metal */
        {
            vec3 refl = v_sub(r.direction,
                              v_mul(rec.normal, 2*v_dot(r.direction,rec.normal)));
            refl = v_add(refl,
                         v_mul(random_vec3(rngState), s->fuzz));
            if(v_dot(refl,rec.normal)<=0) break;      // absorbed
            r = {rec.point, v_unit(refl)};
            atten = { atten.r*s->albedo.r,
                      atten.g*s->albedo.g,
                      atten.b*s->albedo.b };
            continue;
        }
        /* dielectric, emissive, etc. left as exercise */
        break; /* absorb */
    }
    return acc;
}

/* ---------- 5. Kernel ------------------------------------------------*/
__host__ __device__ RGBColorU8 coloru8_createf(CFLOAT r, CFLOAT g, CFLOAT b) {
    return RGBColorU8{
        (uint8_t)fminf(r * 256.0f, 255.0f),
        (uint8_t)fminf(g * 256.0f, 255.0f),
        (uint8_t)fminf(b * 256.0f, 255.0f)
    };
}

__global__ void render_kernel(RGBColorU8 *out,
                              int w, int h,
                              Camera cam,
                              SphereGPU *spheres, int ns,
                              int spp, int maxDepth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x>=w || y>=h) return;

    uint32_t rng = 1234u ^ (y*w + x);   // simple per-thread seed
    RGBColorF col = {0,0,0};

    for(int s=0; s<spp; ++s)
    {
        float u = (x + rng_next(rng)) / (w-1.f);
        float v = (y + rng_next(rng)) / (h-1.f);
        Ray r = cam_getRay_gpu(&cam,u,v, rng);          // **device overload**
        RGBColorF c = ray_color(r, spheres, ns, rng, maxDepth);
        col.r += c.r; col.g += c.g; col.b += c.b;
    }

    float scale = 1.0f / spp;
    col.r = sqrtf(col.r*scale);
    col.g = sqrtf(col.g*scale);
    col.b = sqrtf(col.b*scale);

    int idx = (h-1-y)*w + x;                    // flip Y like CPU code
    out[idx] = coloru8_createf(col.r, col.g, col.b);
}

/* ---------- 6. Host wrapper (called from your main) ----------------- */

void render_cuda(const Camera &cam,
                 const std::vector<Sphere> &cpuSpheres,
                 RGBColorU8 *hostPixels,
                 int W, int H,
                 int samplesPerPixel, int maxDepth)
{
    /* ---- a. flatten & copy spheres --------------------------------- */
    std::vector<SphereGPU> gpuSpheres;
    gpuSpheres.reserve(cpuSpheres.size());
    for(const auto &s: cpuSpheres)
        gpuSpheres.push_back({s.center, s.radius,
                              s.sphMat.matType,
                              s.sphMat.albedo, s.sphMat.fuzz, s.sphMat.ir});

    SphereGPU *d_spheres;
    cudaMalloc(&d_spheres, gpuSpheres.size()*sizeof(SphereGPU));
    cudaMemcpy(d_spheres, gpuSpheres.data(),
               gpuSpheres.size()*sizeof(SphereGPU),
               cudaMemcpyHostToDevice);

    /* ---- b. output buffer ------------------------------------------ */
    RGBColorU8 *d_out;
    cudaMalloc(&d_out, W*H*sizeof(RGBColorU8));

    /* ---- c. timing -------------------------------------------------- */
    cudaEvent_t t0,t1;  cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    /* ---- d. kernel launch ------------------------------------------ */
    dim3 block(8,8);
    dim3 grid ((W+block.x-1)/block.x,
               (H+block.y-1)/block.y);

    render_kernel<<<grid,block>>>(d_out, W, H, cam,
                                  d_spheres, (int)gpuSpheres.size(),
                                  samplesPerPixel, maxDepth);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    /* ---- e. copy back & report ------------------------------------- */
    cudaMemcpy(hostPixels, d_out, W*H*sizeof(RGBColorU8),
               cudaMemcpyDeviceToHost);

    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("[CUDA] Rendered %dx%d @ %d spp in %.2f ms (%.1f Mray/s)\n",
           W,H,samplesPerPixel,ms,
           (double)(W*H*samplesPerPixel)/(ms*1000.0));

    /* ---- f. tidy up ------------------------------------------------- */
    cudaFree(d_out);
    cudaFree(d_spheres);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

#define randomFloat() util_randomFloat(0.0, 1.0)

CFLOAT lcg(int *n) {

    static int seed;
    const int m = 2147483647;
    const int a = 1103515245;
    const int c = 12345;

    if (n != NULL) {
        seed = *n;
    }

    seed = (seed * a + c) % m;
    *n = seed;

    return fabs((CFLOAT)seed / m);
}

void randomSpheres2(ObjectLL *world, DynamicStackAlloc *dsa, int n,
                    Image* imgs, int *seed) {

    LambertianMat *materialGround = (LambertianMat*)alloc_dynamicStackAllocAllocate(
        dsa, sizeof(LambertianMat), alignof(LambertianMat));
    SolidColor *sc1 = (SolidColor*)alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                                      alignof(SolidColor));

    SolidColor *sc = (SolidColor*)alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                                     alignof(SolidColor));

    Checker *c =
        (Checker*)alloc_dynamicStackAllocAllocate(dsa, sizeof(Checker), alignof(Checker));

    sc1->color = { 0.0,  0.0, 0.0};
    sc->color = { 0.4,  0.4, 0.4};

    c->even.tex = sc1;
    c->even.texType = SOLID_COLOR;
    c->odd.tex = sc;
    c->odd.texType = SOLID_COLOR;

    materialGround->lambTexture.tex = c;
    materialGround->lambTexture.texType = CHECKER;

    /*materialGround->albedo.r = 0.5;
    materialGround->albedo.g = 0.5;
    materialGround->albedo.b = 0.5;*/

    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = 0, .y = -1000, .z = 0},
                                .radius = 1000,
                                .sphMat = MAT_CREATE_LAMB_IP(materialGround)});

    for (int a = -2; a < 9; a++) {
        for (int b = -9; b < 9; b++) {
            CFLOAT chooseMat = lcg(seed);
            vec3 center = {
                .x = a + 0.9 * lcg(seed), .y = 0.2, .z = b + 0.9 * lcg(seed)};

            if (chooseMat < 0.8) {
                // diffuse
                RGBColorF albedo = {
                    .r = lcg(seed) * lcg(seed),
                    .g = lcg(seed) * lcg(seed),
                    .b = lcg(seed) * lcg(seed),

                };

                LambertianMat *lambMat = (LambertianMat*)alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(LambertianMat), alignof(LambertianMat));

                SolidColor *sc = (SolidColor*)alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(SolidColor), alignof(SolidColor));

                sc->color = albedo;

                lambMat->lambTexture.tex = sc;
                lambMat->lambTexture.texType = SOLID_COLOR;

                obj_objLLAddSphere(
                    world, { center,MAT_CREATE_LAMB_IP(lambMat),0.2,});

            } else if (chooseMat < 0.95) {
                // metal
                RGBColorF albedo = {.r = lcg(seed) / 2 + 0.5,
                                    .g = lcg(seed) / 2 + 0.5,
                                    .b = lcg(seed) / 2 + 0.5};
                CFLOAT fuzz = lcg(seed) / 2 + 0.5;

                MetalMat *metalMat = (MetalMat*)alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(MetalMat), alignof(MetalMat));

                metalMat->albedo = albedo;
                metalMat->fuzz = fuzz;

                obj_objLLAddSphere(
                    world, (Sphere){.center = center,
                                    .radius = 0.2,
                                    .sphMat = MAT_CREATE_METAL_IP(metalMat)});

            } else {
                DielectricMat *dMat = (DielectricMat*)alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(DielectricMat), alignof(DielectricMat));
                dMat->ir = 1.5;
                obj_objLLAddSphere(
                    world, (Sphere){.center = center,
                                    .radius = 0.2,
                                    .sphMat = MAT_CREATE_DIELECTRIC_IP(dMat)});
            }
        }
    }

    LambertianMat *material2 = (LambertianMat*)alloc_dynamicStackAllocAllocate(
        dsa, sizeof(LambertianMat), alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[0];
    material2->lambTexture.texType = IMAGE;
    /*material2->albedo.r = 0.4;
    material2->albedo.g = 0.2;
    material2->albedo.b = 0.1;
    */

    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = -4, .y = 1, .z = 0},
                                .radius = 1.0,
                                .sphMat = MAT_CREATE_LAMB_IP(material2)});

    material2 = (LambertianMat*)alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
                                                alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[1];
    material2->lambTexture.texType = IMAGE;
    /*material2->albedo.r = 0.4;
    material2->albedo.g = 0.2;
    material2->albedo.b = 0.1;
    */

    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = -4, .y = 1, .z = -2.2},
                                .radius = 1.0,
                                .sphMat = MAT_CREATE_LAMB_IP(material2)});

    material2 = (LambertianMat*)alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
                                                alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[2];
    material2->lambTexture.texType = IMAGE;
    /*material2->albedo.r = 0.4;
    material2->albedo.g = 0.2;
    material2->albedo.b = 0.1;
    */

    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = -4, .y = 1, .z = +2.2},
                                .radius = 1.0,
                                .sphMat = MAT_CREATE_LAMB_IP(material2)});

    material2 = (LambertianMat*)alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
                                                alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[3];
    material2->lambTexture.texType = IMAGE;
    /*material2->albedo.r = 0.4;
    material2->albedo.g = 0.2;
    material2->albedo.b = 0.1;
    */

    obj_objLLAddSphere(world,
                       (Sphere){.center = {.x = -4, .y = 1, .z = -4.2},
                                .radius = 1.0,
                                .sphMat = MAT_CREATE_LAMB_IP(material2)});
}
#undef randomFloat


int main() {
    const CFLOAT aspect_ratio = 16.0 / 9.0;
    const CFLOAT aperture = 0.1;
    const CFLOAT distToFocus = 10.0;
    const int MAX_DEPTH = 50;
    int seed = 100;
    const int SAMPLES_PER_PIXEL = 100;
    const int WIDTH = 640;
    const int HEIGHT = 640;

    RGBColorU8 *image =
            (RGBColorU8 *)malloc(sizeof(RGBColorF) * HEIGHT * WIDTH);

    vec3 lookFrom = {.x = 13.0, .y = 2.0, .z = 3.0};
    vec3 lookAt = {.x = 0.0, .y = 0.0, .z = 0.0};
    vec3 up = {.x = 0.0, .y = 1.0, .z = 0.0};

    DynamicStackAlloc *dsa = alloc_createDynamicStackAllocD(1024, 100);
    DynamicStackAlloc *dsa0 = alloc_createDynamicStackAllocD(1024, 10);
    ObjectLL *world = obj_createObjectLL(dsa0, dsa);
    Image img[4];

    tex_loadImage(&img[0], "./test_textures/kitchen_probe.jpg");
    tex_loadImage(&img[1], "./test_textures/campus_probe.jpg");
    tex_loadImage(&img[2], "./test_textures/building_probe.jpg");
    tex_loadImage(&img[3], "./test_textures/kitchen_probe.jpg");
    randomSpheres2(world, dsa, 4, img, &seed);
    LinearAllocFC *lafc =
            alloc_createLinearAllocFC(MAX_DEPTH * world->numObjects,
                                      sizeof(HitRecord), alignof(HitRecord));

    world->hrAlloc = lafc;

    Camera c;
    cam_setLookAtCamera(&c, lookFrom, lookAt, up, 20, aspect_ratio,
                        aperture, distToFocus);


    std::vector<Sphere> sphereVec = obj_toVector(world);
    render_cuda(c, sphereVec, image, WIDTH, HEIGHT,
                SAMPLES_PER_PIXEL, MAX_DEPTH);


    return 0;
}