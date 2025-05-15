#include <stdlib.h>
#include <stdbool.h>
#include <tgmath.h>
#include <assert.h>

#include "util.h"
#if defined(__CUDACC__)
  // when compiling with NVCC
  #define CUDA_FN __host__ __device__
#else
  // plain host compiler
  #define CUDA_FN
#endif
CUDA_FN static bool nearly_zero(CFLOAT a, CFLOAT epsilon, CFLOAT abs_th){
    assert(F_EPSILON <= epsilon);
    assert(epsilon < 1.0);

    if (a == 0.0) return true;

    CFLOAT diff = fabs(a);
    CFLOAT norm = fmin(diff, F_MAX);
    return diff < fmax(abs_th, epsilon * norm);
}


CUDA_FN CFLOAT util_floatClamp(CFLOAT c, CFLOAT lower, CFLOAT upper){
    if(c <= lower){
        return lower;
    }

    if(c >= upper){
        return upper;
    }

    return c;
}

CUDA_FN uint8_t util_uint8Clamp(uint8_t c, uint8_t lower, uint8_t upper){
    if(c <= lower){
        return lower;
    }

    if(c >= upper){
        return upper;
    }

    return c;
}


CUDA_FN uint32_t util_randomRange(uint32_t lower, uint32_t upper){
    return (rand() % (upper - lower + 1)) + lower;
}


CUDA_FN vec3 util_randomUnitSphere(){
    CFLOAT x, y, z;
    while(true){
        x = util_randomFloat(0.0, 1.0);
        y = util_randomFloat(0.0, 1.0);
        z = util_randomFloat(0.0, 1.0);

        if(x*x + y*y + z*z >= 1){
            continue;
        }else{
            break;
        }
    }

    return {
         x,
         y,
         z
    };
}

CUDA_FN CFLOAT util_randomFloat(CFLOAT lower, CFLOAT upper){
    CFLOAT scale = rand() / (CFLOAT) RAND_MAX;
    return scale * (upper - lower) + lower;
}

CUDA_FN vec3 util_randomUnitDisk(){
    while(true){
        vec3 p = {
             util_randomFloat(-1.0, 1.0),
             util_randomFloat(-1.0, 1.0),
             0
        };

        if(p.x*p.x + p.y*p.y >= 1){
            continue;
        }

        return p;
    }
}

CUDA_FN vec3 util_randomUnitVector(){
    
    CFLOAT x, y, z;
    while(true){
        x = util_randomFloat(0.0, 1.0);
        y = util_randomFloat(0.0, 1.0);
        z = util_randomFloat(0.0, 1.0);

        if(x*x + y*y + z*z > 1){
            continue;
        }else{
            break;
        }
    }

    CFLOAT len = sqrt(x*x + y*y + z*z);        

    return{
        x/len,
        y/len,
        z/len
    };
}

CUDA_FN vec3 util_vec3Reflect(vec3 v,vec3 n){
    CFLOAT two_dot_product = 2*vector3_dot_product(&v, &n);
    vector3_multiplyf(&n, two_dot_product);
    vector3_subtract(&v,&n);
    return v;
}

#define float_zero(a) nearly_zero(a, 128 * F_EPSILON, F_MIN)

CUDA_FN bool util_isVec3Zero(vec3 v){
    return (float_zero(v.x)) && (float_zero(v.y)) && (float_zero(v.z));
}

#undef float_zero
