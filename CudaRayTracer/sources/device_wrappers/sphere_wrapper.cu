#include "sphere.h"
//
// Created by johan on 15/05/2025.
//
__device__ void hr_setRecordi(CFLOAT distanceFromOrigin, vec3 point, vec3 normal, vec3 direction, HitRecord* restrict outRecord, const Material * restrict hitObjMat){
    // if temp < 0 then the ray has intersected the object at the front face
    // otherwise it has intersected the object at the back face
    CFLOAT temp = vector3_dot_product(&direction, &normal);
    bool frontFace = (temp < 0) ? true : false;

    // Adjusting the normal so it always points away in the opposite direction
    // of the ray
    if(!frontFace) {
        vector3_negate(&normal);
    }

    outRecord->point = point;
    outRecord->distanceFromOrigin = distanceFromOrigin;
    outRecord->normal = normal;
    outRecord->valid = true;
    outRecord->frontFace = frontFace;
    outRecord->hitObjMat = hitObjMat;
}

__device__ void obj_sphereTexCoords(vec3 pointOnSphere,
                         CFLOAT * outU, CFLOAT * outV){
    CFLOAT theta = acos(-pointOnSphere.y);
    CFLOAT phi = atan2(-pointOnSphere.z, pointOnSphere.x) + CF_PI;

    *outU = phi / (2 * CF_PI);
    *outV = theta / CF_PI;

}

__device__ bool obj_sphereHit(const Sphere* restrict s, Ray r, CFLOAT t_min, CFLOAT t_max, HitRecord * outRecord){
    vec3 oc = r.origin;
    vec3 direction = r.direction;

    /*
     * center - center of the sphere
     * radius - radius of the sphere
     * direction - direction of the ray
     */

    // oc = origin - center
    vector3_subtract(&oc, &s->center);

    // a = dot(direction, direction)
    CFLOAT a = vector3_dot_product(&direction, &direction);

    // half_b = dot((origin - center), direction)
    CFLOAT half_b = vector3_dot_product(&oc, &direction);

    // c = dot(origin, origin) - radius^2
    CFLOAT c = vector3_dot_product(&oc, &oc) - s->radius * s->radius;

    // discri = half_b^2 - a * c
    CFLOAT discri = half_b * half_b - a*c;

    // If the discriminant is less than 0 then no intersection
    if(discri < 0){
        // outRecord->valid = false;
        return false;
    }

    // sqrtd = sqrt(discri)
    CFLOAT sqrtd = sqrt(discri);

    // root1 = (-half_b - sqrtd) / a
    CFLOAT root = (-half_b - sqrtd) / a;

    // If the intersection point corresponding to this root is
    // not in the intersection range then check the other point
    if(root < t_min || t_max < root){
        root = (-half_b + sqrtd) / a;

        // If neither roots correspond to an intersection point in
        // the intersection range then return invalid
        if(root < t_min || t_max < root){
            // outRecord->valid = false;
            return false;
        }
    }

    // t = root
    CFLOAT t = root;

    // inter1 = direction
    vec3 inter1 = direction;

    // inter1 = root * direction
    vector3_multiplyf(&inter1, root);

    // p = inter1 + origin
    vec3 p = *(vector3_add(&inter1, &r.origin));
    // n = p
    vec3 n = p;

    // n = p - center
    vector3_subtract(&n, &s->center);

    // n = (p - center)/radius
    vector3_multiplyf(&n, 1/s->radius);

    hr_setRecordi(t, p, n, direction, outRecord, &s->sphMat);
    obj_sphereTexCoords(outRecord->normal, &outRecord->u, &outRecord->v);

    return true;
}

__device__ static bool hit(
        const Object * restrict obj,
        Ray r,
        CFLOAT t_min,
        CFLOAT t_max, HitRecord * outRecord )
{
    if(obj->objType == SPHERE){
        return obj_sphereHit((const Sphere *)obj->object, r, t_min, t_max, outRecord);
    }else if(obj->objType == OBJLL){
        return obj_objLLHit((const ObjectLL *)obj->object, r, t_min, t_max, outRecord);
    }


    return false;
}

__device__ bool obj_objLLHit (const ObjectLL* restrict objll,
                   Ray r,
                   CFLOAT t_min,
                   CFLOAT t_max,
                   HitRecord * out){

    if(!objll || !objll->valid){
        return false; //NULL;
    }

    HitRecord * hr = (HitRecord *) alloc_linearAllocFCAllocate(objll->hrAlloc);
    HitRecord * h = NULL;

    ObjectLLNode * cur = objll->head;
    while(cur != NULL){
        hit(&cur->obj, r, t_min, t_max, hr);

        if(hr->valid){
            if(h == NULL ){
                h = hr;
                hr = (HitRecord *) alloc_linearAllocFCAllocate(objll->hrAlloc);
            }else if(hr->distanceFromOrigin < h->distanceFromOrigin){
                h = hr;
                hr = (HitRecord *) alloc_linearAllocFCAllocate(objll->hrAlloc);
            }

        }

        cur = cur->next;
    }

    if(h != NULL){
        *out = *h;
        return true;
    }

    return false;
    //return h;
}