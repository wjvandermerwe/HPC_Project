


__device__ unsigned int d_nextPixel;

void resetPixelCounter(int w, int h)
{
    unsigned int zero = 0;
    cudaMemcpyToSymbol(d_nextPixel, &zero, sizeof(unsigned int));
}

__global__ void render_persistent(RGBColorU8* out, int w, int h,
                                  Camera cam, int ns,
                                  int spp, int maxDepth,
                                  cudaTextureObject_t envTex)
{
    extern __shared__ SphereGPU s_sph[];
    for (int i = threadIdx.x; i < ns; i += blockDim.x)
        s_sph[i] = d_spheres[i];
    __syncthreads();

    /* one infinite loop per thread */
    while (true)
    {
        /* atomically grab the next pixel index */
        unsigned int idx = atomicAdd(&d_nextPixel, 1u);
        if (idx >= (unsigned)w*h) break;          // all done

        int x = idx %  w;
        int y = idx /  w;

        uint32_t rng = 1234u ^ idx;
        RGBColorF col = {0,0,0};

#pragma unroll
        for (int s = 0; s < spp; ++s) {
            float u = (x + rng_next(rng)) / (w - 1.f);
            float v = (y + rng_next(rng)) / (h - 1.f);

            Ray r = cam_getRay_gpu(&cam, u, v, rng);
            RGBColorF c = ray_color(r, s_sph, ns, rng, maxDepth, envTex);
            col.r += c.r; col.g += c.g; col.b += c.b;
        }

        float scale = rsqrtf((float)spp);
        col.r = sqrtf(col.r) * scale;
        col.g = sqrtf(col.g) * scale;
        col.b = sqrtf(col.b) * scale;

        out[(h-1-y)*w + x] = coloru8_createf_gpu(col.r, col.g, col.b);
    }
}


int main(int argc, char** argv) {



    resetPixelCounter(w,h);

    dim3 block(128);
    int  smCount;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
    dim3 grid(smCount * 4);             // ~4 blocks per SM is a good start

    size_t shmem = ns * sizeof(SphereGPU);
    render_persistent<<<grid, block, shmem>>>(d_out,w,h,
                                              cam, ns, spp, maxDepth,
                                              envTex);



}