//
// Created by johan on 14/05/2025.
//


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
                    Image imgs[n], int *seed) {

    LambertianMat *materialGround = alloc_dynamicStackAllocAllocate(
        dsa, sizeof(LambertianMat), alignof(LambertianMat));
    SolidColor *sc1 = alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                                      alignof(SolidColor));

    SolidColor *sc = alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                                     alignof(SolidColor));

    Checker *c =
        alloc_dynamicStackAllocAllocate(dsa, sizeof(Checker), alignof(Checker));

    sc1->color = (RGBColorF){.r = 0.0, .b = 0.0, .g = 0.0};
    sc->color = (RGBColorF){.r = 0.4, .b = 0.4, .g = 0.4};

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

                LambertianMat *lambMat = alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(LambertianMat), alignof(LambertianMat));

                SolidColor *sc = alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(SolidColor), alignof(SolidColor));

                sc->color = albedo;

                lambMat->lambTexture.tex = sc;
                lambMat->lambTexture.texType = SOLID_COLOR;

                obj_objLLAddSphere(
                    world, (Sphere){.center = center,
                                    .radius = 0.2,
                                    .sphMat = MAT_CREATE_LAMB_IP(lambMat)});

            } else if (chooseMat < 0.95) {
                // metal
                RGBColorF albedo = {.r = lcg(seed) / 2 + 0.5,
                                    .g = lcg(seed) / 2 + 0.5,
                                    .b = lcg(seed) / 2 + 0.5};
                CFLOAT fuzz = lcg(seed) / 2 + 0.5;

                MetalMat *metalMat = alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(MetalMat), alignof(MetalMat));

                metalMat->albedo = albedo;
                metalMat->fuzz = fuzz;

                obj_objLLAddSphere(
                    world, (Sphere){.center = center,
                                    .radius = 0.2,
                                    .sphMat = MAT_CREATE_METAL_IP(metalMat)});

            } else {
                DielectricMat *dMat = alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(DielectricMat), alignof(DielectricMat));
                dMat->ir = 1.5;
                obj_objLLAddSphere(
                    world, (Sphere){.center = center,
                                    .radius = 0.2,
                                    .sphMat = MAT_CREATE_DIELECTRIC_IP(dMat)});
            }
        }
    }

    LambertianMat *material2 = alloc_dynamicStackAllocAllocate(
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

    material2 = alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
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

    material2 = alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
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

    material2 = alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
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

__device__ RGBColorF ray_c_device(Ray r, const ObjectLL* world, int depth){
    if(depth<=0)
        return (RGBColorF){0};

    HitRecord rec;
    rec.valid=false;
    if(obj_objLLHit(world, r, 1e-5f, FLT_MAX, &rec)){
        Ray scattered={0};
        RGBColorF attenuation={0};
        if(mat_scatter(&r,&rec,&attenuation,&scattered)){
            RGBColorF col = ray_c_device(scattered, world, depth-1);
            return colorf_multiply(col, attenuation);
        }
        return (RGBColorF){0};
    }
    vec3 d=r.direction; vector3_normalize(&d);
    CFLOAT t = 0.5f*(d.y+1.f);
    vec3 a,b;
    vector3_setf3(&a,1.f-t,1.f-t,1.f-t);
    vector3_setf3(&b,0.5f*t,0.7f*t,1.f*t);
    vector3_add(&b,&a);
    return (RGBColorF){b.x,b.y,b.z};
}