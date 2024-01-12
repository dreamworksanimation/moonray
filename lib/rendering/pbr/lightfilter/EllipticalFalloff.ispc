// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "EllipticalFalloff.isph"

#include <scene_rdl2/common/platform/IspcUtil.isph>


//----------------------------------------------------------------------------

ISPC_UTIL_EXPORT_UNIFORM_STRUCT_TO_HEADER(EllipticalFalloff);


export uniform uint32_t
EllipticalFalloff_hudValidation(uniform bool verbose)
{
    ELLIPTICAL_FALLOFF_VALIDATION;
}


//----------------------------------------------------------------------------

varying float
evalSuperEllipse(varying float x, varying float y,
                 uniform float a, uniform float b,
                 uniform float A, uniform float B,
                 uniform float roundness)
{
    float t;

    x = abs(x);
    y = abs(y);

    if (roundness < 0.00001f) {
        /* Simpler case of a square */

        /* 1 - smoothstep(A, a, x) */
        if (x > A) {
            t = 0;
        } else if (x < a) {
            t = 1;
        } else if (A == a) { /* (A == a == x) */
            /* Degenerate case, don't want a divide by zero */
            t = 0;
        } else {
            t = 1 - (x - a) / (A - a);
        }

        /* 1 - smoothstep(B, b, y) */
        if (y > B) {
            t = 0;
        } else if (y >= b) {
            if (B == b) {
                /* Degenerate case, don't want a divide by zero */
                t = 0;
            } else {
                t *= 1 - (y - b) / (B - b);
            }
        }

        return t;

    } else {
        /* Harder, rounded corner case */
        uniform float re = 2.0f / roundness;   /* roundness exponent */
        float q  = a * b * pow(pow(b * x, re) + pow(a * y, re), -1.0f / re);
        float r  = A * B * pow(pow(B * x, re) + pow(A * y, re), -1.0f / re);

        /* smoothstep(r, q, 1) */
        if (1 > r) {
            t = 1;
        } else if (1 < q) {
            t = 0;
        } else if (r == q) {
            /* Degenerate case, don't want a divide by zero */
            /* By setting t to 1, we return 0 */
            t = 1;
        } else {
            t = (1 - q) / (r - q);
        }

        return 1 - t;
    }
}


//----------------------------------------------------------------------------


