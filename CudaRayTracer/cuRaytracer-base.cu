#include "deviceHelpers.cuh"

__global__ void render_kernel(RGBColorU8 *out,
                              int w, int h,
                              Camera cam,
                              SphereGPU *spheres, int ns,
                              int spp, int maxDepth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x>=w || y>=h) return;

    uint32_t rng = 1234u ^ (y*w + x);
    RGBColorF col = {0,0,0};

    for(int s=0; s<spp; ++s)
    {
        float u = (x + rng_next(rng)) / (w-1.f);
        float v = (y + rng_next(rng)) / (h-1.f);
        Ray r = cam_getRay_gpu(&cam,u,v, rng);
        RGBColorF c = ray_color(r, spheres, ns, rng, maxDepth);
        col.r += c.r; col.g += c.g; col.b += c.b;
    }

    float scale = 1.0f / spp;
    col.r = sqrtf(col.r*scale);
    col.g = sqrtf(col.g*scale);
    col.b = sqrtf(col.b*scale);

    int idx = (h-1-y)*w + x;
    out[idx] = coloru8_createf_gpu(col.r, col.g, col.b);
}

int main() {

    SceneContext params = prepare_world();
    RenderCtx render_ctx = render_init_cuda(params.spheres, params.width, params.height);

    dim3 block(8,8);
    dim3 grid((params.width + block.x - 1)/block.x,
              (params.height + block.y - 1)/block.y);

    render_kernel<<<grid, block>>>(render_ctx.d_out,
                                   params.width, params.height,
                                   params.cam,
                                   render_ctx.d_spheres, render_ctx.ns,
                                   params.spp, params.maxDepth);

    render_finish_cuda(render_ctx, params.framebuffer, "out");

    return 0;
}