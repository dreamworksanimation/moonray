// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
///

// Contains the Albedo Tables for BsdfFabricVelvet calculated based on
// "A Microfacet Based Coupled Specular-Matte BRDF Model
//  with Importance Sampling" - EGSR'01 - Kelemen et al.

#include <scene_rdl2/common/platform/IspcUtil.isph>
#include "VelvetAlbedo.isph"
#include "VelvetAlbedoTable.isph"

ISPC_UTIL_EXPORT_ENUM_TO_HEADER(VelvetAlbedo_Constants);

// Helper Functions for VelvetAlbedo Calculations
// Get upper and lower index for a given roughness value
void
getAvgTableIndices(float roughness,
                   int& index1,
                   int& index2)
{
    roughness = clamp(roughness, 0.0f, 1.0f);
    index1 = (int)floor(roughness*(float)VELVET_ALBEDO_ROUGHNESS_STEPS-1.0f);
    index2 = (int)ceil(roughness*(float)VELVET_ALBEDO_ROUGHNESS_STEPS-1.0f);
}

void
getAlbedoTableIndices(float cosTheta,
                      float roughness,
                      int& lowerIndex,
                      int& upperIndex)
{
    int rIndex1, rIndex2;
    getAvgTableIndices(roughness,
                       rIndex1, rIndex2);

    // cosTheta is divided into 100 buckets and ranges from 0 to pi/2
    // so the indirection here converts back into a lookup index
    cosTheta  = clamp(cosTheta,  0.0f, 1.0f);
    const int cosIndex = cosTheta * VELVET_ALBEDO_COS_THETA_STEPS - 1;
    const int albedoTableSize = VELVET_ALBEDO_TABLE_SIZE-1;
    lowerIndex = clamp(rIndex1*VELVET_ALBEDO_COS_THETA_STEPS + cosIndex, 0, albedoTableSize);
    upperIndex = clamp(rIndex2*VELVET_ALBEDO_COS_THETA_STEPS + cosIndex, 0, albedoTableSize);
}

float
VelvetAlbedo_avg(float roughness)
{
    int lowerIndex, upperIndex;
    getAvgTableIndices(roughness, lowerIndex, upperIndex);

    const float frac = roughness - (int)(roughness);

    return ((1.f-frac) * VelvetAlbedo_avgAlbedo[lowerIndex] +
            frac       * VelvetAlbedo_avgAlbedo[upperIndex]);
}

float
VelvetAlbedo_at(float cosTheta,
                float roughness)
{
    int lowerIndex, upperIndex;
    getAlbedoTableIndices(cosTheta, roughness,
                          lowerIndex, upperIndex);

    const float frac = roughness - (int)(roughness);

    const float v1 = VelvetAlbedo_albedoTable[lowerIndex];
    const float v2 = VelvetAlbedo_albedoTable[upperIndex];
    return ((1.0f-frac) * v1 +
             frac      * v2);
}

// Accesor functions for the velvet albedo tables
export void *uniform
VelvetAlbedo_getAvgAlbedoTable() { return VelvetAlbedo_avgAlbedo; }

export void *uniform
VelvetAlbedo_getAlbedoTable() { return VelvetAlbedo_albedoTable; }

