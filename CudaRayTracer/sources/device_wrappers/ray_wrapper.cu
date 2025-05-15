#include "ray.h"
//
// Created by johan on 15/05/2025.
//
__device__ Ray ray_create(vec3 origin, vec3 direction){
    vector3_normalize(&direction);

    return {
         origin,
         direction
    };
}