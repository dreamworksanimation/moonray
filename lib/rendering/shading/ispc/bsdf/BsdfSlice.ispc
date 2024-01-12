// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfSlice.ispc

#include "BsdfSlice.isph"

#include <scene_rdl2/common/math/ispc/Constants.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>

varying float
BsdfSlice_computeShadowTerminatorFix(const varying BsdfSlice * uniform slice,
                                     const varying Vec3f &N,
                                     const varying Vec3f &wi)
{
    float G;

    switch (slice->mShadowTerminatorFix) {
    case SHADOW_TERMINATOR_FIX_OFF:
        G = 1.0f;
        break;
    case SHADOW_TERMINATOR_FIX_CUSTOM:
    {
        // Custom, targeted shadow terminator that only softens the
        // hard shadow edge and does not affect any other highlights
        // Thresholding cosine in a smoothstep range [0, sin(0.15)] for cosY->0
        // and [0, sin(0.1)] for cosY->1. If this is still not soft enough, we can
        // potentially expose this threshold as a user-option or widen it.
        // sin(0.15) = 0.14944f;
        // sin(0.02)  = 0.01999f;
        const float cosX = clamp(dot(N, slice->mNg), 0.0f, 1.0f);
        const float d = 0.14944f*(1.0f - cosX) + 0.01999f*cosX;
        const float t = clamp(dot(slice->mNg, wi)/d, 0.0f, 1.0f);
        // Cubic Hermite Interpolation
        G = t*t*(3.0f - 2.0f*t);
    }
    break;
    case SHADOW_TERMINATOR_FIX_SINE_COMPENSATION:
        // Same as Chiang's method, but replaces cosine(angle(Ng, N)) with sine(angle(Ng, N)).
        // The basic idea is to penalize small deviations between the geometric
        // and shadow normals less than in Chiang.
        G = clamp(min(1.0f, dot(slice->mNg, wi) /
            (max(sEpsilon, dot(N, wi)) *
             max(sEpsilon, length(cross(slice->mNg, N))))), 0.0, 1.0);
        G = -(G * G * G) + G * G + G;
        break;
    case SHADOW_TERMINATOR_FIX_GGX:
        // "A Microfacet-Based Shadowing Function to Solve The Bump Terminator Problem"
        // Conty Estevez, Lecocq, Stein Ray-Tracing Gems 2019
        {
            const float cosD = min(abs(dot(slice->mNg, N)), 1.0f);
            const float tan2D = (1.0f - cosD * cosD) / (cosD * cosD);
            const float alpha2 = clamp(0.125f * tan2D, 0.0, 1.0);

            const float cosI = max(abs(dot(slice->mNg, wi)), sEpsilon);
            const float tan2I = (1.0f - cosI * cosI) / (cosI * cosI);
            G = 2.0f / (1 + sqrt(1.0 + alpha2 * tan2I));
        }
        break;
    case SHADOW_TERMINATOR_FIX_COSINE_COMPENSATION:
        // "Taming the Shadow Terminator" Chiang, Li, Burley  SIGGRAPH 2019
        G = clamp(min(1.0f, dot(slice->mNg, wi) /
                (max(sEpsilon, dot(N, wi)) *
                 max(sEpsilon, dot(slice->mNg, N)))), 0.0, 1.0);
        G = -(G * G * G) + G * G + G;
        break;
    case SHADOW_TERMINATOR_FIX_NUM_MODES:
    default:
        MNRY_ASSERT(0);
        G = 1.0f;
        break;
    }

    return G;
}


#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(BsdfSlice);
ISPC_UTIL_EXPORT_ENUM_TO_HEADER(ShadowTerminatorFix);

