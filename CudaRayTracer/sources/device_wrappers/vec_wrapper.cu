//
// Created by johan on 15/05/2025.
//

#include "hypatia.h"


/** @brief A function that returns the minimum of \a a and \a b */
static HYP_INLINE HYP_FLOAT HYP_MIN(HYP_FLOAT a, HYP_FLOAT b)
{
    return (a < b) ? a : b;
}

/** @brief A macro that returns the maximum of \a a and \a b */
static HYP_INLINE HYP_FLOAT HYP_MAX(HYP_FLOAT a, HYP_FLOAT b)
{
    return (a > b) ? b : a;
}
/**
 * @ingroup vector3
 * @brief computes the dot product of two vectors
 */
HYPAPI HYP_FLOAT vector3_dot_product(const struct vector3 *self, const struct vector3 *vT)
{
    return (self->x * vT->x) + (self->y * vT->y) + (self->z * vT->z);
}
/** @brief A macro that swaps \a a and \a b */
static HYP_INLINE void HYP_SWAP(HYP_FLOAT *a, HYP_FLOAT *b)
{
    HYP_FLOAT f = *a; *a = *b; *b = f;
}

HYPAPI struct vector3 *vector3_normalize(struct vector3 *self)
{
    HYP_FLOAT mag;

    mag = vector3_magnitude(self);

    if (scalar_equalsf(mag, 0.0f)) {
        /* can't normalize a zero
         * avoid divide by zero
         */
        return self;
    }

    self->x = self->x / mag;
    self->y = self->y / mag;
    self->z = self->z / mag;

    return self;
}

/**
 * @ingroup vector3
 * @brief calculates the magnitude of the vector
 */
HYPAPI HYP_FLOAT vector3_magnitude(const struct vector3 *self)
{
    return HYP_SQRT((self->x * self->x) + (self->y * self->y) + (self->z * self->z));
}

HYPAPI struct vector3 *vector3_setf3(struct vector3 *self, HYP_FLOAT xT, HYP_FLOAT yT, HYP_FLOAT zT)
{
    self->x = xT;
    self->y = yT;
    self->z = zT;
    return self;
}
/**
 * @ingroup vector3
 * @brief subtract each vector's component by a scalar
 */
HYPAPI struct vector3 *vector3_subtractf(struct vector3 *self, HYP_FLOAT f)
{
    self->v[0] -= f;
    self->v[1] -= f;
    self->v[2] -= f;
    return self;
}

/**
 * @ingroup vector3
 * @brief subtract two vectors using component-wise subtraction
 */
HYPAPI struct vector3 *vector3_subtract(struct vector3 *self, const struct vector3 *vT)
{
    self->v[0] -= vT->v[0];
    self->v[1] -= vT->v[1];
    self->v[2] -= vT->v[2];
    return self;
}


/**
 * @brief This checks for mathematical equality within HYP_EPSILON.
 *
 */
HYPAPI short scalar_equalsf(const HYP_FLOAT f1, const HYP_FLOAT f2)
{
    return scalar_equals_epsilonf(f1, f2, HYP_EPSILON);
}

/**
 * @brief This checks for mathematical equality within a custom epsilon.
 *
 */
HYPAPI short scalar_equals_epsilonf(const HYP_FLOAT f1, const HYP_FLOAT f2, const HYP_FLOAT epsilon)
{
    if ((HYP_ABS(f1 - f2) < epsilon) == 0) {
        return 0;
    }

    return 1;
}


/**
 * @ingroup vector3
 * @brief multiplies each component of the vector by a scalar
 */
HYPAPI struct vector3 *vector3_multiplyf(struct vector3 *self, HYP_FLOAT f)
{
    self->v[0] *= f;
    self->v[1] *= f;
    self->v[2] *= f;
    return self;
}


/**
 * @ingroup vector3
 * @brief switches the sign on each component of the vector
 */
HYPAPI struct vector3 *vector3_negate(struct vector3 *self)
{
    self->v[0] = -self->v[0];
    self->v[1] = -self->v[1];
    self->v[2] = -self->v[2];
    return self;
}

/**
 * @ingroup vector3
 * @brief adds vectors using component-wise addition
 */
HYPAPI struct vector3 *vector3_add(struct vector3 *self, const struct vector3 *vT)
{
    self->v[0] += vT->v[0];
    self->v[1] += vT->v[1];
    self->v[2] += vT->v[2];
    return self;
}


/**
 * @ingroup vector3
 * @brief add to each component of the vector using a scalar
 */
HYPAPI struct vector3 *vector3_addf(struct vector3 *self, HYP_FLOAT f)
{
    self->v[0] += f;
    self->v[1] += f;
    self->v[2] += f;
    return self;
}