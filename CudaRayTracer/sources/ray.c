#include "ray.h"

__device__ Ray ray_create(vec3 origin, vec3 direction){
    vector3_normalize(&direction);
    
    return (Ray){
        .origin = origin,
        .direction = direction
    };
}
