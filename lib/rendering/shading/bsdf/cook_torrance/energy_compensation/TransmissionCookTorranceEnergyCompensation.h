// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TransmissionCookTorranceEnergyCompensation.h
/// $Id$
///

/*
 * This file is used to compensate for energy loss from
 * microfacet transmission lobes as outlined in Walter'07.
 * We use the Kelemen'01 energy compensation BRDF as described in
 * Kulla'17 to be added to the regular microfacet lobe.
 * This file describes the energy to be added in reflection and
 * transmission to be perfectly energy 'preserving'.
 * It provides a factor used to calculate the energy
 * distribution between the two that follows reciprocity and is based
 * on average fresnel reflectance parameterized by (etaT/etaI).
 * It also provides sampling curves, PDF and a one-sample weight to sample
 * the reflection term efficiently.
 * Ref:
 * Revisiting Physically Based Shading, PBS Kulla'17
 */
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/ReferenceFrame.h>

#pragma once

namespace moonray {
namespace shading {

//----------------------------------------------------------------------------
class TransmissionCookTorranceEnergyCompensation
{
public:

    // Transmission Energy Compensation BRDF Terms
    // Reflection Compensation
    static scene_rdl2::math::Color evalR(
            const float cosNO,
            const float cosNI,
            float inputRoughness,
            float etaI, float etaT,
            float favg, float favgInv,
            bool includeCosineTerm);
    // Transmission Compensation
    static scene_rdl2::math::Color evalT(
            const float cosNO,
            const float cosNI,
            float inputRoughness,
            float etaI, float etaT,
            float favg, float favgInv,
            bool includeCosineTerm);

    // Sampling R Lobe
    static void sampleR(const scene_rdl2::math::Vec3f& wo,
                        const scene_rdl2::math::ReferenceFrame& frame,
                        float r1, float r2,
                        scene_rdl2::math::Vec3f &wi,
                        float r,
                        float etaI, float etaT);
    static float pdfR(float cosNI,
                     float etaI, float etaT,
                     float roughness);
    // One-Sample Weight for Compensation Lobe
    static float weightR(float etaI, float etaT,
                         float roughness);
private:
    // Factor to distribute energy between reflection and
    // transmission as described in Kulla'17
    static float
    computeFactor(float favg,
                  float favgInv,
                  float etaI, float etaT,
                  float roughness);

};

//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

