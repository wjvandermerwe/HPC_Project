//
// Created by johan on 14/05/2025.
//

#ifndef HELPERS_H
#define HELPERS_H
#include "color.h"
#include "sphere.h"

void randomSpheres2(ObjectLL *world, DynamicStackAlloc *dsa, int n,
                    Image* imgs, int *seed);

__device__ RGBColorF ray_c_device(Ray r, const ObjectLL* world, int depth);


#endif //HELPERS_H
