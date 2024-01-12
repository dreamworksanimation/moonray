// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

namespace moonray {
namespace shading  {

scene_rdl2::math::Vec3f
evalDisplacementMap(
    const scene_rdl2::rdl2::Displacement* displacementMap,
    const scene_rdl2::math::Vec3f& position,
    const scene_rdl2::math::Vec3f& normal,
    const scene_rdl2::math::Vec3f& dPds,
    const scene_rdl2::math::Vec3f& dPdt,
    const scene_rdl2::math::Vec2f& uv,
    float dSdx, float dSdy,
    float dTdx, float dTdy);

} // shading
} // moonray


