// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#include <moonray/rendering/pbr/core/Distribution.isph>

//----------------------------------------------------------------------------

// Samples the given distribution, and checks that the results are equal to the
// given samples. Returns true if all samples are the same, and false otherwise.
// In any case sets the u and v samples array with the values computed herein.
export uniform bool
sampleDistribution2D(const uniform Distribution2D * uniform dist, uniform int size,
        const uniform float * uniform r1, const uniform float * uniform r2,
        uniform float * uniform u, uniform float * uniform v)
{
    bool equal = true;

    foreach (i = 0 ... size) {
        if (i < size) {
            Vec2f uv;
            float pdf;
            Distribution2D_sampleNearest(dist, r1[i], r2[i], &uv, &pdf);

            equal &= isEqual(u[i], uv.x);
            equal &= isEqual(v[i], uv.y);

            u[i] = uv.x;
            v[i] = uv.y;
        }
    }

    return all(equal);
}


export uniform bool
sampleImageDistribution(const uniform ImageDistribution * uniform dist, uniform int size,
        const uniform float * uniform r1, const uniform float * uniform r2,
        uniform float * uniform u, uniform float * uniform v)
{
    bool equal = true;

    foreach (i = 0 ... size) {
        if (i < size) {
            Vec2f uv;
            float pdf;
            ImageDistribution_sample(dist, r1[i], r2[i], 0.0f, &uv, &pdf, TEXTURE_FILTER_NEAREST);

            equal &= isEqual(u[i], uv.x);
            equal &= isEqual(v[i], uv.y);

            u[i] = uv.x;
            v[i] = uv.y;
        }
    }

    return all(equal);
}


//----------------------------------------------------------------------------

