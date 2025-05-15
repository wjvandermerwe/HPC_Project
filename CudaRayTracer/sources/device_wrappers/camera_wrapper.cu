//
// Created by johan on 15/05/2025.
//

#include "hypatia.h"
#include "camera.h"
#include "ray.h"
#include "util.h"
#ifdef __cplusplus
#  define restrict __restrict__
#endif

__device__ Ray cam_getRay(const Camera * restrict cam, CFLOAT u, CFLOAT v){

    // randOnDist = lensRadius * util_randomUnitDisk()
    vec3 randOnDist = util_randomUnitDisk();
    vector3_multiplyf(&randOnDist, cam->lensRadius);

    CFLOAT x = randOnDist.x;
    CFLOAT y = randOnDist.y;

    // offset = randOnDist.x * c->u + randOnDist.y * c->v
    vec3 offset = {
        x * cam->u.x + y * cam->v.x,
        x * cam->u.y + y * cam->v.y,
        x * cam->u.z + y * cam->v.z,
    };

    // outOrigin = cam->origin + offset
    vec3 outOrigin = cam->origin;
    vector3_add(&outOrigin, &offset);

    // direction = lowerLeftCorner + u*horizontal + v*vertical - origin - offset
    vec3 outDirection = cam->lowerLeftCorner;
    vec3 uHori = cam->horizontal;
    vector3_multiplyf(&uHori, u);
    vec3 vVeri = cam->vertical;
    vector3_multiplyf(&vVeri, v);
    vector3_add(&outDirection, &uHori);
    vector3_add(&outDirection, &vVeri);
    vector3_subtract(&outDirection, &cam->origin);
    vector3_subtract(&outDirection, &offset);

    return ray_create(outOrigin, outDirection);
}
