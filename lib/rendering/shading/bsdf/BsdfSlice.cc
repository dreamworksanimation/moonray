// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfSlice.cc
/// $Id$
///


#include "BsdfSlice.h"

#include <scene_rdl2/common/math/Constants.h>
#include <scene_rdl2/scene/rdl2/SceneVariables.h>

namespace moonray {
namespace shading {

// make sure rdl and shading enums stay in sync
MNRY_STATIC_ASSERT(int(scene_rdl2::rdl2::ShadowTerminatorFix::OFF) ==
                  int(ispc::SHADOW_TERMINATOR_FIX_OFF));
MNRY_STATIC_ASSERT(int(scene_rdl2::rdl2::ShadowTerminatorFix::CUSTOM) ==
                  int(ispc::SHADOW_TERMINATOR_FIX_CUSTOM));
MNRY_STATIC_ASSERT(int(scene_rdl2::rdl2::ShadowTerminatorFix::SINE_COMPENSATION) ==
                  int(ispc::SHADOW_TERMINATOR_FIX_SINE_COMPENSATION));
MNRY_STATIC_ASSERT(int(scene_rdl2::rdl2::ShadowTerminatorFix::GGX) ==
                  int(ispc::SHADOW_TERMINATOR_FIX_GGX));
MNRY_STATIC_ASSERT(int(scene_rdl2::rdl2::ShadowTerminatorFix::COSINE_COMPENSATION) ==
                  int(ispc::SHADOW_TERMINATOR_FIX_COSINE_COMPENSATION));

float
BsdfSlice::computeShadowTerminatorFix(const scene_rdl2::math::Vec3f &N, const scene_rdl2::math::Vec3f &wi) const
{

    float G;

    switch (mShadowTerminatorFix) {
    case ispc::SHADOW_TERMINATOR_FIX_OFF:
        G = 1.0f;
        break;
    case ispc::SHADOW_TERMINATOR_FIX_CUSTOM:
    {
        // Custom, targeted shadow terminator that only softens the
        // hard shadow edge and does not affect any other highlights
        // Thresholding cosine in a smoothstep range [0, sin(0.15)] for cosX->0
        // and [0, sin(0.1)] for cosY->1. If this is still not soft enough, we can
        // potentially expose this threshold as a user-option or widen it.
        // sin(0.15) = 0.14944f;
        // sin(0.02)  = 0.01999f;
        const float cosX = scene_rdl2::math::clamp(scene_rdl2::math::dot(N, mNg));
        const float d = scene_rdl2::math::lerp(0.14922f, 0.01999f, cosX);
        const float t = scene_rdl2::math::clamp(scene_rdl2::math::dot(mNg, wi)/d);
        // Cubic Hermite Interpolation
        G = t*t*(3.0f - 2.0f*t);
    }
    break;
    case ispc::SHADOW_TERMINATOR_FIX_SINE_COMPENSATION:
        // Same as Chiang's method, but replaces cosine(angle(Ng, N)) with sine(angle(Ng, N)).
        // The basic idea is to penalize small deviations between the geometric
        // and shading normals less than in Chiang.
        G = scene_rdl2::math::clamp(scene_rdl2::math::min(1.0f, scene_rdl2::math::dot(mNg, wi) /
            (std::max(scene_rdl2::math::sEpsilon, scene_rdl2::math::dot(N, wi)) *
             std::max(scene_rdl2::math::sEpsilon, scene_rdl2::math::length(scene_rdl2::math::cross(mNg, N))))));
        G = -(G * G * G) + G * G + G;
        break;
    case ispc::SHADOW_TERMINATOR_FIX_GGX:
        // "A Microfacet-Based Shadowing Function to Solve The Bump Terminator Problem"
        // Conty Estevez, Lecocq, Stein Ray-Tracing Gems 2019
        {
            const float cosD = scene_rdl2::math::min(scene_rdl2::math::abs(scene_rdl2::math::dot(mNg, N)), 1.0f);
            const float tan2D = (1.0f - cosD * cosD) / (cosD * cosD);
            const float alpha2 = scene_rdl2::math::clamp(0.125f * tan2D);

            const float cosI = scene_rdl2::math::max(scene_rdl2::math::abs(scene_rdl2::math::dot(mNg, wi)), scene_rdl2::math::sEpsilon);
            const float tan2I = (1.0f - cosI * cosI) / (cosI * cosI);
            G = 2.0f / (1 + scene_rdl2::math::sqrt(1.0 + alpha2 * tan2I));
        }
        break;
    case ispc::SHADOW_TERMINATOR_FIX_COSINE_COMPENSATION:
        // "Taming the Shadow Terminator" Chiang, Li, Burley  SIGGRAPH 2019
        G = scene_rdl2::math::clamp(scene_rdl2::math::min(1.0f, scene_rdl2::math::dot(mNg, wi) /
                (std::max(scene_rdl2::math::sEpsilon, scene_rdl2::math::dot(N, wi)) *
                 std::max(scene_rdl2::math::sEpsilon, scene_rdl2::math::dot(mNg, N)))));
        G = -(G * G * G) + G * G + G;
        break;
    case ispc::SHADOW_TERMINATOR_FIX_NUM_MODES:
    default:
        MNRY_ASSERT(0 && "unknown shadow terminator fix type");
        G = 1.0f;
        break;
    }

    return G;
}

} // namespace shading
} // namespace moonray

