// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

// Contains the Albedo Tables for BsdfFabricVelvet calculated based on
// "A Microfacet Based Coupled Specular-Matte BRDF Model
//  with Importance Sampling" - EGSR'01 - Kelemen et al.

#pragma once

#include <moonray/rendering/shading/ispc/bsdf/fabric/VelvetAlbedo_ispc_stubs.h>
#include <scene_rdl2/common/math/Math.h>

namespace moonray {
namespace shading {

class VelvetAlbedo {

public:
    /// Returns the AverageAlbedo for this specific roughness
    static float avg(float roughness);
    /// Returns the Albedo for this specific roughness, cosTheta combo
    static float at(float cosTheta, float roughness);

private:

    static void getAvgTableIndices(float roughness,
                                   int& index1,
                                   int& index2)
    {
        MNRY_ASSERT(roughness >= 0.0f && roughness <= 1.0f);
        roughness = scene_rdl2::math::clamp(roughness, 0.0f, 1.0f);
        index1 = static_cast<int>(floor(roughness*ispc::VELVET_ALBEDO_ROUGHNESS_STEPS-1.0f));
        index2 = static_cast<int>(ceil(roughness*ispc::VELVET_ALBEDO_ROUGHNESS_STEPS-1.0f));
    }

    static void getAlbedoTableIndices(float cosTheta,
                                      float roughness,
                                      int& index1,
                                      int& index2)
    {
        int rIndex1, rIndex2;
        getAvgTableIndices(roughness,
                           rIndex1, rIndex2);

        cosTheta  = scene_rdl2::math::clamp(cosTheta,  0.0f, 1.0f);
        const int cosIndex = cosTheta * ispc::VELVET_ALBEDO_COS_THETA_STEPS - 1;
        const int albedoTableSize = ispc::VELVET_ALBEDO_TABLE_SIZE - 1;
        index1 = scene_rdl2::math::clamp(rIndex1*ispc::VELVET_ALBEDO_COS_THETA_STEPS + cosIndex, 0, albedoTableSize);
        index2 = scene_rdl2::math::clamp(rIndex2*ispc::VELVET_ALBEDO_COS_THETA_STEPS + cosIndex, 0, albedoTableSize);
    }
};

} // namespace shading
} // namespace moonray

