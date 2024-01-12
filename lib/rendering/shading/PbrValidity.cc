// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#include "PbrValidity.h"

namespace moonray {
namespace shading {

scene_rdl2::math::Color
computeAlbedoPbrValidity(const scene_rdl2::math::Color& albedo)
{
    scene_rdl2::math::Color res = scene_rdl2::math::sBlack;
    const float value = scene_rdl2::math::max(albedo.r, albedo.g, albedo.b);

    if (value > sPbrValidityAlbedoMin && value < sPbrValidityAlbedoMax) {
        res = sPbrValidityValidColor;
    } else if (value < sPbrValidityAlbedoMin) {
        const float gradient = value / sPbrValidityAlbedoMin;
        res = scene_rdl2::math::lerp(sPbrValidityInvalidColor, sPbrValidityValidColor, gradient);
    } else { // value > LAMBERT_PBR_VALIDITY_MAX
        const float gradient = scene_rdl2::math::saturate((value - sPbrValidityAlbedoMax) /
                                              (1.0f - sPbrValidityAlbedoMax));
        res = scene_rdl2::math::lerp(sPbrValidityValidColor, sPbrValidityInvalidColor, gradient);
    }
    return res;
}

} // namespace shading
} // namespace moonray




