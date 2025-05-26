//
// Created by johan on 19/05/2025.
//

#ifndef HELPERS_H
#define HELPERS_H
#include <vector>

#include "sphere.h"
#ifdef __cplusplus
extern "C" {
#endif
inline void add_spheres_from_object(const Object& obj,
                                    std::vector<Sphere>& out);

inline void gather_spheres(const ObjectLL& list,
                           std::vector<Sphere>& out)
{
    const ObjectLLNode* node = list.head;
    while (node)
    {
        add_spheres_from_object(node->obj, out);
        node = node->next;
    }
}

/* ---- dispatch on object kind -------------------------------- */
inline void add_spheres_from_object(const Object& obj,
                                    std::vector<Sphere>& out)
{
    switch (obj.objType)
    {
        case SPHERE:
            out.push_back(*static_cast<const Sphere*>(obj.object));
            break;

        case OBJLL:   // nested linked list
            gather_spheres(*static_cast<const ObjectLL*>(obj.object),
                                   out);
            break;

        case OBJBVH:  // TODO: walk BVH if/when you implement it
        default:
            /* ignored for CUDA flattening */
            break;
    }
}

inline std::vector<Sphere> obj_toVector(const ObjectLL* world)
{
    std::vector<Sphere> spheres;
    if (!world || !world->valid) return spheres;

    spheres.reserve(world->numObjects);
    gather_spheres(*world, spheres);
    return spheres;
}

void writeToPPM(const char *filename,
                int         width,
                int         height,
                const RGBColorU8 *pixels);

CFLOAT lcg(int *seed);

void randomSpheres2(ObjectLL        *world,
                    DynamicStackAlloc *dsa,
                    int               n,
                    Image            *imgs,
                    int              *seed);


#ifdef __cplusplus
}
#endif
#endif //HELPERS_H
