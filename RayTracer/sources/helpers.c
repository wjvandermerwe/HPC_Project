//
// Created by johan on 25/05/2025.
//
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include "allocator.h"
#include "color.h"
#include "sphere.h"

void writeToPPM(const char * filename, int width, int height,
                const RGBColorU8* arr){

    FILE *fptr = fopen(filename, "w");

    if(fptr == NULL){
        printf("ERROR: File not found.\n");
        exit(1);
    }

    fprintf(fptr, "P3\n");
    fprintf(fptr,"%d %d\n", width, height);
    fprintf(fptr,"255\n");

    for(int i = 0; i < width*height; i++){
        fprintf(fptr, "%hu %hu %hu\n", arr[i].r, arr[i].g, arr[i].b);
    }

    fclose(fptr);
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

    sc1->color = (RGBColorF){ 0.0,  0.0, 0.0};
    sc->color = (RGBColorF){ 0.4,  0.4, 0.4};

    c->even.tex = sc1;
    c->even.texType = SOLID_COLOR;
    c->odd.tex = sc;
    c->odd.texType = SOLID_COLOR;

    materialGround->lambTexture.tex = c;
    materialGround->lambTexture.texType = CHECKER;

    Sphere sph ={ { 0,  -1000, 0},MAT_CREATE_LAMB_IP(materialGround),1000};
    obj_objLLAddSphere(world, sph);

    for (int a = -2; a < 9; a++) {
        for (int b = -9; b < 9; b++) {
            CFLOAT chooseMat = lcg(seed);
            vec3 center = {
                 a + 0.9 * lcg(seed),  0.2, b + 0.9 * lcg(seed)};

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

                obj_objLLAddSphere(world, (Sphere){ center,MAT_CREATE_LAMB_IP(lambMat),0.2,});

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

                Sphere sph = { center,MAT_CREATE_METAL_IP(metalMat), 0.2};

                obj_objLLAddSphere(world, sph);

            } else {
                DielectricMat *dMat = (DielectricMat*)alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(DielectricMat), alignof(DielectricMat));
                dMat->ir = 1.5;
                Sphere sph = { center,MAT_CREATE_DIELECTRIC_IP(dMat), 0.2};
                obj_objLLAddSphere(
                    world, sph);
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

    sph = (Sphere){ { -4, 1,  0},MAT_CREATE_LAMB_IP(material2),1.0};
    obj_objLLAddSphere(world, sph);

    material2 = (LambertianMat*)alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
                                                alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[1];
    material2->lambTexture.texType = IMAGE;
    /*material2->albedo.r = 0.4;
    material2->albedo.g = 0.2;
    material2->albedo.b = 0.1;
    */
    sph = (Sphere){ { -4, 1,  -2.2},MAT_CREATE_LAMB_IP(material2),1.0};
    obj_objLLAddSphere(world, sph);

    material2 = (LambertianMat*)alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
                                                alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[2];
    material2->lambTexture.texType = IMAGE;
    /*material2->albedo.r = 0.4;
    material2->albedo.g = 0.2;
    material2->albedo.b = 0.1;
    */
    sph = (Sphere){ { -4, 1,  +2.2},MAT_CREATE_LAMB_IP(material2),1.0};
    obj_objLLAddSphere(world, sph);

    material2 = (LambertianMat*)alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
                                                alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[3];
    material2->lambTexture.texType = IMAGE;
    /*material2->albedo.r = 0.4;
    material2->albedo.g = 0.2;
    material2->albedo.b = 0.1;
    */
    sph = (Sphere){ { -4,  1,  -4.2},MAT_CREATE_LAMB_IP(material2), 1.0};
    obj_objLLAddSphere(world, sph);
}
#undef randomFloat