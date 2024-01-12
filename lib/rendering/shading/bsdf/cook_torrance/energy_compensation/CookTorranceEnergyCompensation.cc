// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file CookTorranceEnergyCompensation.cc
/// $Id$
///

/*
 * This file is used to compensate for energy loss from
 * microfacet reflection lobes as outlined in Walter'07.
 * We use the Kelemen'01 energy compensation BRDF as described in
 * Kulla'17 to be added to the regular microfacet BRDF.
 * This file describes the term to be added in reflection
 * to be perfectly energy 'preserving'. It also provides
 * sampling curves, PDF and a one-sample weight to sample
 * this term efficiently.
 * Ref:
 * (1) A Microfacet Based Coupled Specular-Matte BRDF Model
 * with Importance Sampling, Kelemen, 01
 * (2) Revisiting Physically Based Shading, PBS Kulla'17
 */
#include "CookTorranceEnergyCompensation.h"
#include "ReflectionAlbedo.h"
#include <scene_rdl2/common/math/MathUtil.h>
#include <moonray/rendering/shading/Util.h>

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

// BRDF Compensation Term as outlined in
// Kelemen'01 Section 2.2
// (1 - E(cosThetaO)*(1-E(cosThetaI))/(1 - Eavg) * cosThetaI
scene_rdl2::math::Color
CookTorranceEnergyCompensation::eval(
        ispc::MicrofacetDistribution type,
        const float cosNO,
        const float cosNI,
        float roughness,
        const scene_rdl2::math::Color& favg,
        bool includeCosineTerm)
{
    float compensation =
            (1.0f - ReflectionAlbedo::E(type,
                                        cosNO, roughness)) *
            (1.0f - ReflectionAlbedo::E(type,
                                        cosNI, roughness));
    const float divisor = ReflectionAlbedo::oneMinusAvg(type,
                                                        roughness);
    if (!scene_rdl2::math::isZero(divisor)) {
        compensation /= divisor;
    } else {
        compensation = 0.0f;
    }

    const float eavg = 1.0f - ReflectionAlbedo::oneMinusAvg(type,
                                                            roughness);
    const Color multiplier = favg * favg * eavg /
            (scene_rdl2::math::sWhite - favg*(1.0f - eavg));

    const Color comp = compensation * multiplier * sOneOverPi;

    return (comp *
            (includeCosineTerm  ?  cosNI : 1.0f));
}

// Sampling Curves Determined by fitting a curve
// to tabulated CDFInverse data calculated using
// moonshine_devtools/compute_cdf based on the PDF
// outlined in Kelemen'01
void
CookTorranceEnergyCompensation::sample(
        ispc::MicrofacetDistribution type,
        const scene_rdl2::math::ReferenceFrame& frame,
        float r1, float r2,
        float r,
        scene_rdl2::math::Vec3f& wi)
{
    float cosThetaI;
    switch(type) {
    case ispc::MICROFACET_DISTRIBUTION_BECKMANN:
        cosThetaI = pow(r1,
                        2.767f*r*r-5.4f*r+2.99412f);
        break;
    case ispc::MICROFACET_DISTRIBUTION_GGX:
    default:
        cosThetaI = pow(r1,
                        1.1933f*r*r-2.0969f*r+1.3698f);
    }

    const float sinThetaI = sqrt(1.0f - cosThetaI*cosThetaI);
    const float phiI = 2.0f * sPi * r2;

    const Vec3f m = computeLocalSphericalDirection(cosThetaI, sinThetaI, phiI);
    wi = frame.localToGlobal(m);
}

// PDF -
// Kelemen'01 Section 2.2.1
// (1-E(cosThetaI))/(pi*(1 - Eavg)) * cosThetaI
float
CookTorranceEnergyCompensation::pdf(
        ispc::MicrofacetDistribution type,
        float cosNI,
        float roughness)
{
    float result = 0.0f;
    const float norm = sPi * ReflectionAlbedo::oneMinusAvg(type,
                                                           roughness);
    if (isZero(norm)) return 0.0f;
    result = (cosNI * (1.0f - ReflectionAlbedo::E(type,
                                                  cosNI,
                                                  roughness)));
    result /= norm;
    return max(0.0f, result);
}

float
CookTorranceEnergyCompensation::weight(
        float roughness)
{
    const float w = max(0.0f, (roughness-0.5f))*1.0f;
    return w;
}


} // namespace shading
} // namespace moonray

