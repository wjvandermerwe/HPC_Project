#include "deviceHelpers.cuh"

#include <vector>

#if defined(__cplusplus) || defined(__CUDACC__)
#  ifndef restrict
#    define restrict __restrict__
#  endif
#endif
extern "C" {
    #include "camera.h"
    #include "color.h"
    #include "helpers.h"
    #include "hitRecord.h"
    #include "sphere.h"
    #include "material.h"
    #include "texture.h"
}

__device__ inline vec3 random_vec3(uint32_t &rngState) {
    return {
        rng_next(rngState),
        rng_next(rngState),
        rng_next(rngState)
    };
}

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
    return true;
}

__constant__ RGBColorF c_bgA = { 0.95f, 0.95f, 0.98f };   // light square
__constant__ RGBColorF c_bgB = { 0.15f, 0.18f, 0.22f };   // dark square

__device__ RGBColorF ray_color(Ray r,
                               const SphereGPU *spheres, int ns,
                               uint32_t &rngState, int maxDepth)
{
    RGBColorF acc = {0,0,0}, atten = {1,1,1};
    for(int depth=0; depth<maxDepth; ++depth)
    {
        HitRecord rec; rec.valid = false;
        const SphereGPU *hit = nullptr;
        float nearest = 1e30f;
        /* find nearest hit */
        for(int i=0;i<ns;++i)
            if(hit_sphere(spheres[i], r, 0.0001f, nearest, rec)) {
                nearest = rec.distanceFromOrigin;
                hit = &spheres[i];
            }

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

        const SphereGPU *s = hit;
        if(s->matType == LAMBERTIAN)
        {
            vec3 dir = v_add(rec.normal, random_vec3(rngState));
            r = { rec.point, v_unit(dir) };
            atten.r *= s->albedo.r;
            atten.g *= s->albedo.g;
            atten.b *= s->albedo.b;
            continue;
        }

        if(s->matType == METAL)
        {
            vec3 reflected = reflect(v_unit(r.direction), rec.normal);
            reflected = v_add(reflected, v_mul(random_vec3(rngState), s->fuzz));
            if(v_dot(reflected, rec.normal) <= 0.0f) break; // absorbed
            r = { rec.point, v_unit(reflected) };
            atten.r *= s->albedo.r;
            atten.g *= s->albedo.g;
            atten.b *= s->albedo.b;
            continue;
        }

        if(s->matType == DIELECTRIC)
        {
            float eta   = rec.frontFace ? (1.0f / s->ir) : s->ir;
            float cosIn = fminf(-v_dot(v_unit(r.direction), rec.normal), 1.0f);
            float reflProb = schlick(cosIn, eta);

            vec3 dir;
            if(rng_next(rngState) < reflProb ||
               !refract(v_unit(r.direction), rec.normal, eta, dir))
            {
                dir = reflect(v_unit(r.direction), rec.normal);
            }
            r = { rec.point, v_unit(dir) };
            /* dielectric is colourless ⇒ no albedo attenuation */
            continue;
        }

        break; /* absorb */
    }
    return acc;
}



SceneContext prepare_world() {
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

    vec3 lookFrom = { 13.0,  2.0,  3.0};
    vec3 lookAt = { 0.0,  0.0,  0.0};
    vec3 up = { 0.0,  1.0,  0.0};

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


    return SceneContext {
    .width = WIDTH,
    .height = HEIGHT,
    .spp = SAMPLES_PER_PIXEL,
    .maxDepth = MAX_DEPTH,
    .cam = c,
    .spheres = sphereVec,
    .framebuffer = image,
    .world = world,
    .dsaMaterials = dsa,
    .dsaTextures = dsa0,
    .hitRecAlloc = lafc
};
}

RenderCtx render_init_cuda(const std::vector<Sphere> &cpuSpheres, int W, int H)
{
    /* --- build SphereGPU array on host ---------------------------- */
    std::vector<SphereGPU> gpuSpheres;
    gpuSpheres.reserve(cpuSpheres.size());
    for (const auto &s : cpuSpheres) {
        SphereGPU g{};
        g.center = s.center;
        g.radius = s.radius;
        switch (s.sphMat.matType) {
            case LAMBERTIAN:
                g.matType = 0;

                g.albedo  = ((LambertianMat*)s.sphMat.mat)->albedo;
                break;
            case METAL:
                g.matType = 1;
                g.albedo  = ((MetalMat*)s.sphMat.mat)->albedo;
                g.fuzz    = ((MetalMat*)s.sphMat.mat)->fuzz;
                break;
            case DIELECTRIC:
                g.matType = 2;
                g.ir      = ((DielectricMat*)s.sphMat.mat)->ir;
                break;
        }
        gpuSpheres.push_back(g);
    }

    /* --- device allocations -------------------------------------- */
    RenderCtx ctx;
    ctx.ns = static_cast<int>(gpuSpheres.size());
    ctx.W  = W;
    ctx.H  = H;

    cudaMalloc(&ctx.d_spheres, ctx.ns * sizeof(SphereGPU));
    cudaMemcpy(ctx.d_spheres, gpuSpheres.data(),
               ctx.ns * sizeof(SphereGPU), cudaMemcpyHostToDevice);

    cudaMalloc(&ctx.d_out, W * H * sizeof(RGBColorU8));

    return ctx;          /* hand the opaque context back to caller */
}

void render_finish_cuda(RenderCtx &ctx,
                        RGBColorU8 *hostPixels,
                        const char *ppmName)
{
    cudaMemcpy(hostPixels, ctx.d_out,
               ctx.W * ctx.H * sizeof(RGBColorU8),
               cudaMemcpyDeviceToHost);

    writeToPPM(ppmName, ctx.W, ctx.H, hostPixels);

    cudaFree(ctx.d_out);
    cudaFree(ctx.d_spheres);

    ctx = {};
}