// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
#include "Displacement.h"

#include <moonray/rendering/shading/EvalShader.h>

#include <moonray/rendering/bvh/shading/Intersection.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>

namespace moonray {
namespace shading {


scene_rdl2::math::Vec3f
evalDisplacementMap(
    const scene_rdl2::rdl2::Displacement* displacementMap,
    const scene_rdl2::math::Vec3f& position,
    const scene_rdl2::math::Vec3f& normal,
    const scene_rdl2::math::Vec3f& dPds,
    const scene_rdl2::math::Vec3f& dPdt,
    const scene_rdl2::math::Vec2f& uv,
    float dSdx, float dSdy,
    float dTdx, float dTdy)
{
    mcrt_common::ThreadLocalState* tls = mcrt_common::getFrameUpdateTLS();
    shading::TLState *shadingTls = MNRY_VERIFY(tls->mShadingTls.get());
    Intersection intersection;

    // TODO: if we want the displacement map to evaluate attributes
    // we need to fill in all the parameters.
    intersection.initDisplacement(tls,
            nullptr /*Attribute Table*/,
            nullptr /*rdl2::Geometry*/,
            nullptr /*rdl2::Layer*/,
            -1 /*assignmentId*/,
            position, normal, dPds, dPdt, uv, 0.0f, 0.0f, 0.0f, 0.0f);

    scene_rdl2::math::Vec3f delta = scene_rdl2::math::Vec3f(0.0f, 0.0f, 0.0f);
    displace(displacementMap, shadingTls, State(&intersection), &delta);

    return delta;
}

} // shading
} // moonray


