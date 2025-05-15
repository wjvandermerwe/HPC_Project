#include <assert.h>
#include <tgmath.h>

#include "material.h"
#include "util.h"
#include "ray.h"
#include "types.h"

static CFLOAT reflectance(CFLOAT cosine, CFLOAT ref_idx){
    //using schlik's approximation
    CFLOAT r0 = (1-ref_idx)/(1+ref_idx);
    r0 *= r0;

    return r0 + (1-r0) * pow((1 - cosine), 5);
}





