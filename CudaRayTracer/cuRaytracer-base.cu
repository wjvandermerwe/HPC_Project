// cuRaytracer-base.cu – CUDA replacement for the OpenMP path‑tracer
// Build: nvcc -O3 -rdc=true -arch=sm_70 cuRaytracer-base.cu -o cuRaytracer-base
// This *base* version keeps exactly the same scene/physics code you already have
// (HitRecord, obj_objLLHit, mat_scatter, etc.) and merely runs the per‑pixel
// loop on the GPU.  No shared-memory tricks, no texture/buffer objects – all
// data sit in global memory so it’s easy to follow.
// ─────────────────────────────────────────────────────────────────────────────
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <vector>
#ifdef __cplusplus
#  define restrict __restrict__
#endif
#include "allocator.h"
#include "texture.h"
#include "helpers.h"
#include "outfile.h"
#include "types.h"      // CFLOAT, vec3, RGBColorF, RGBColorU8, …
#include "ray.h"        // Ray struct + cam_getRay
#include "camera.h"     // cam_setLookAtCamera
#include "sphere.h"   // ObjectLL + obj_objLLHit
#include "material.h"   // mat_scatter
#include "util.h"       // writeToPPM, COLOR_U8CREATE, etc.

// ————————————————————————————————————————————————————————————————
//  Device helpers
// ————————————————————————————————————————————————————————————————
__device__ inline CFLOAT gpu_rand(curandState* s){ return curand_uniform(s); }

// Recursive colour function – identical to CPU but RNG passed in
__device__ RGBColorF ray_c_device(Ray r, const ObjectLL* world, int depth,
                                  curandState* st){
    if(depth<=0) return makeColorF();

    HitRecord rec; rec.valid=false;
    if(obj_objLLHit(world, r, 1e-5f, FLT_MAX, &rec)){
        Ray scattered={0}; RGBColorF attenuation={0};
        if(mat_scatter(&r,&rec,&attenuation,&scattered)){
            RGBColorF col = ray_c_device(scattered, world, depth-1, st);
            return colorf_multiply(col, attenuation);
        }
        return makeColorF();
    }
    vec3 d=r.direction; vector3_normalize(&d);
    CFLOAT t = 0.5f*(d.y+1.f);
    vec3 a,b; vector3_setf3(&a,1.f-t,1.f-t,1.f-t);
    vector3_setf3(&b,0.5f*t,0.7f*t,1.f*t); vector3_add(&b,&a);
    return makeColorF(b.x,b.y,b.z);
}

// Kernel: each thread shades one pixel, accumulates SPP samples
__global__ void render_kernel(RGBColorU8* fb, int W, int H, int spp, int maxD,
                              Camera cam, const ObjectLL* dWorld){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;
    int idx = y*W + x;

    curandState rng; curand_init(1234ULL+idx,0,0,&rng);
    CFLOAT rAcc=0,gAcc=0,bAcc=0;
    for(int s=0;s<spp;++s){
        CFLOAT u = (x + gpu_rand(&rng)) / (W-1.f);
        CFLOAT v = (y + gpu_rand(&rng)) / (H-1.f);
        Ray r = cam_getRay(&cam,u,v);
        RGBColorF c = ray_c_device(r,dWorld,maxD,&rng);
        rAcc += c.r; gAcc += c.g; bAcc += c.b;
    }
    CFLOAT scale=1.f/spp;
    rAcc=sqrtf(rAcc*scale); gAcc=sqrtf(gAcc*scale); bAcc=sqrtf(bAcc*scale);
    fb[idx]=COLOR_U8CREATE(rAcc,gAcc,bAcc);
}

// ————————————————————————————————————————————————————————————————
// Host: build scene exactly like CPU, copy linked‑list arena to GPU once.
// ————————————————————————————————————————————————————————————————
static ObjectLL* build_world(){
    DynamicStackAlloc* dsa = alloc_createDynamicStackAllocD(1024,100);
    DynamicStackAlloc* dsa0= alloc_createDynamicStackAllocD(1024,10);
    ObjectLL* w = obj_createObjectLL(dsa0,dsa);
    // randomSpheres2(w,dsa); // or randomSpheres2(...)
    return w;
}

/* ───────────────────────── helpers ──────────────────────────────────── */
static StackAlloc*  copy_stackalloc_to_device(const StackAlloc* hSA,
                                              StackAlloc**      dHdrOut)
{
    /* 1. copy the raw buffer */
    uint8_t* dBuf;
    cudaMalloc(&dBuf, hSA->totalSize);
    cudaMemcpy(dBuf, hSA->buffptr,
               hSA->totalSize, cudaMemcpyHostToDevice);

    /* 2. clone & patch the header */
    StackAlloc saHdr = *hSA;
    saHdr.buffptr    = dBuf;

    StackAlloc* dSA;
    cudaMalloc(&dSA, sizeof(StackAlloc));
    cudaMemcpy(dSA, &saHdr, sizeof(StackAlloc), cudaMemcpyHostToDevice);

    *dHdrOut = dSA;                 /* return header pointer               */
    return (StackAlloc*)dBuf;       /* return buffer start for offset math */
}

static DynamicStackAlloc* copy_dsa_to_device(const DynamicStackAlloc* hDsa)
{
    /* copy every live StackAlloc header+buffer and collect device headers */
    std::vector<StackAlloc*> dSAhdrs(hDsa->ps.curOffset);
    std::vector<void*>       dSAdata(hDsa->ps.curOffset);

    for (size_t i = 0; i < hDsa->ps.curOffset; ++i) {
        StackAlloc* hSA = (StackAlloc*)hDsa->ps.bufptr[i];
        StackAlloc* dHdr;
        uint8_t*    dBuf = (uint8_t*)copy_stackalloc_to_device(hSA, &dHdr);
        dSAhdrs[i] = dHdr;
        dSAdata[i] = dBuf;          /* keep for head-pointer patch later   */
    }

    /* copy PtrStack.bufptr array */
    void** dPtrArray;
    cudaMalloc(&dPtrArray,
               hDsa->ps.maxPointers * sizeof(void*));
    cudaMemcpy(dPtrArray, dSAhdrs.data(),
               hDsa->ps.maxPointers * sizeof(void*),
               cudaMemcpyHostToDevice);

    /* clone & patch PtrStack header */
    PtrStack psHdr  = hDsa->ps;
    psHdr.bufptr    = dPtrArray;

    PtrStack* dPs;
    cudaMalloc(&dPs, sizeof(PtrStack));
    cudaMemcpy(dPs, &psHdr, sizeof(PtrStack), cudaMemcpyHostToDevice);

    /* clone & patch DynamicStackAlloc header */
    DynamicStackAlloc dsaHdr = *hDsa;
    dsaHdr.ps = psHdr;                 /* struct copy is fine (holds ptr)   */

    DynamicStackAlloc* dDsa;
    cudaMalloc(&dDsa, sizeof(DynamicStackAlloc));
    cudaMemcpy(dDsa, &dsaHdr, sizeof(DynamicStackAlloc),
               cudaMemcpyHostToDevice);

    return dDsa;   /* device pointer to fully patched DynamicStackAlloc     */
}

/* ─────────────────── deep-copy LinearAllocFC (HitRecord pool) ─────────── */
static LinearAllocFC* copy_lafc_to_device(const LinearAllocFC* hLafc)
{
    size_t bufBytes = hLafc->totalSize;
    uint8_t* dBuf;
    cudaMalloc(&dBuf, bufBytes);
    cudaMemcpy(dBuf, hLafc->bufptr,
               bufBytes, cudaMemcpyHostToDevice);

    LinearAllocFC lafHdr = *hLafc;
    lafHdr.bufptr        = dBuf;

    LinearAllocFC* dLaf;
    cudaMalloc(&dLaf, sizeof(LinearAllocFC));
    cudaMemcpy(dLaf, &lafHdr, sizeof(LinearAllocFC),
               cudaMemcpyHostToDevice);
    return dLaf;
}


int main(int argc,char** argv){ if(argc<2){printf("usage: %s out.ppm\n",argv[0]); return 0;} srand(time(NULL));
    const int W=640, H=360, SPP=100, MAXD=50;

    // 1️⃣ Host world
    ObjectLL* hWorld = build_world();

    // 2️⃣ Copy arenas to device (single contiguous memcpy)
    DynamicStackAlloc* dDsa = copy_dsa_to_device(hWorld->dsa);
    LinearAllocFC*     dLaf = copy_lafc_to_device(hWorld->hrAlloc);

    /* patch linked-list head pointer: same offset inside first StackAlloc */
    uintptr_t hostBase = (uintptr_t)((StackAlloc*)hWorld->dsa->ps.bufptr[0])->buffptr;
    uintptr_t devBase  = (uintptr_t)((StackAlloc*)dDsa->ps.bufptr[0])->buffptr;
    ObjectLLNode* dHead =
        (ObjectLLNode*)(devBase + ((uintptr_t)hWorld->head - hostBase));

    /* final ObjectLL copy */
    ObjectLL hPatched = *hWorld;
    hPatched.dsa     = dDsa;
    hPatched.hrAlloc = dLaf;
    hPatched.head    = dHead;

    ObjectLL* dWorld;
    cudaMalloc(&dWorld, sizeof(ObjectLL));
    cudaMemcpy(dWorld, &hPatched, sizeof(ObjectLL),
               cudaMemcpyHostToDevice);

    // 3️⃣ Camera
    Camera cam; vec3 lookFrom={13,2,3},lookAt={0,0,0},up={0,1,0};
    cam_setLookAtCamera(&cam,lookFrom,lookAt,up,20,(CFLOAT)W/H,0.1f,10.f);

    // 4️⃣ Framebuffer
    RGBColorU8* dFB; cudaMalloc(&dFB,W*H*sizeof(RGBColorU8));

    // 5️⃣ Launch kernel (all global mem)
    dim3 blk(8,8), grd((W+7)/8,(H+7)/8);
    render_kernel<<<grd,blk>>>(dFB,W,H,SPP,MAXD,cam,dWorld);
    cudaDeviceSynchronize();

    // 6️⃣ Save image
    std::vector<RGBColorU8> hFB(W*H);
    cudaMemcpy(hFB.data(),dFB,W*H*sizeof(RGBColorU8),cudaMemcpyDeviceToHost);
    writeToPPM(argv[1],W,H,hFB.data());

    // 7️⃣ Cleanup
    cudaFree(dFB);
    cudaFree(dWorld);
    // cudaFree(dArenaMem);
    return 0;
}
