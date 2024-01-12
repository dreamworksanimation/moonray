// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

// Contains the Albedo Tables for BsdfFabricVelvet

#include "VelvetAlbedo.h"

using namespace scene_rdl2::math;

namespace moonray {
namespace shading {

/// Returns the AvgAlbedo over all possible thetas for this roughness
float
VelvetAlbedo::avg(float roughness)
{
    const float* avgAlbedoTable = reinterpret_cast<const float*>(ispc::VelvetAlbedo_getAvgAlbedoTable());
    int lowerIndex, upperIndex;
    getAvgTableIndices(roughness, lowerIndex, upperIndex);

    const float frac = roughness - static_cast<int>(roughness);
    // linear blending
    return ((1.f-frac) * avgAlbedoTable[lowerIndex] +
            frac       * avgAlbedoTable[upperIndex]);
}

/// Returns the albedo at this theta, roughness
float
VelvetAlbedo::at(float cosTheta, float roughness)
{
    // cosTheta is divided into 100 buckets and ranges from 0 to pi/2
    // so the indirection here converts back into a lookup index
    const float* albedoTable = reinterpret_cast<const float*>(ispc::VelvetAlbedo_getAlbedoTable());
    int lowerIndex, upperIndex;
    getAlbedoTableIndices(cosTheta, roughness,
                          lowerIndex, upperIndex);

    const float frac = roughness - static_cast<int>(roughness);
    // linear blending
    return ((1.f-frac) * albedoTable[lowerIndex] +
            frac       * albedoTable[upperIndex]);
}

} // namespace shading
} // namespace moonray

