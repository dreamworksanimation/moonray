// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#include "PbrValidity.isph"

varying Color
computeAlbedoPbrValidity(const varying Color& albedo)
{
    varying Color res = sBlack;
    const float value = max(max(albedo.r, albedo.g), albedo.b);

    if (value > sPbrValidityAlbedoMin && value < sPbrValidityAlbedoMax) {
        res = sPbrValidityValidColor;
    } else if (value < sPbrValidityAlbedoMin) {
        const float gradient = value / sPbrValidityAlbedoMin;
        res = lerp(sPbrValidityInvalidColor, sPbrValidityValidColor, gradient);
    } else { // value > LAMBERT_PBR_VALIDITY_MAX
        const float gradient = saturate((value - sPbrValidityAlbedoMax) /
                                        (1.0f - sPbrValidityAlbedoMax));
        res = lerp(sPbrValidityValidColor, sPbrValidityInvalidColor, gradient);
    }
    return res;
}

