// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file CookTorranceEnergyCompensation.ispc
/// $Id$
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

#include <moonray/rendering/shading/ispc/Util.isph>
#include "CookTorranceEnergyCompensation.isph"
#include "ReflectionAlbedo.isph"

// BRDF Compensation Term as outlined in
// Kelemen'01 Section 2.2
// (1 - E(cosThetaO)*(1-E(cosThetaI))/(1 - Eavg) * cosThetaI
Color
CookTorranceEnergyCompensation_eval(
        MicrofacetDistribution type,
        const float cosNO,
        const float cosNI,
        float roughness,
        const Color& favg,
        bool includeCosineTerm)
{
    float compensation =
            (1.0f - ReflectionAlbedo_E(type,
                                        cosNO, roughness)) *
            (1.0f - ReflectionAlbedo_E(type,
                                        cosNI, roughness));
    const float divisor = ReflectionAlbedo_oneMinusAvg(type,
                                                        roughness);
    if (!isZero(divisor)) {
        compensation /= divisor;
    } else {
        compensation = 0.0f;
    }

    const float eavg = 1.0f - ReflectionAlbedo_oneMinusAvg(type,
                                                            roughness);
    Color multiplier = favg * favg * eavg;
    const Color divideBy = (sWhite - favg*(1.0f - eavg));
    multiplier.r = multiplier.r / divideBy.r;
    multiplier.g = multiplier.g / divideBy.g;
    multiplier.b = multiplier.b / divideBy.b;
    const Color comp = compensation * multiplier * sOneOverPi;

    return (comp *
            (includeCosineTerm  ?  cosNI : 1.0f));
}

// Sampling Curves Determined by fitting a curve
// to tabulated CDFInverse data calculated using
// moonshine_devtools/compute_cdf based on the PDF
// outlined in Kelemen'01
void
CookTorranceEnergyCompensation_sample(
        MicrofacetDistribution type,
        const ReferenceFrame& frame,
        float r1, float r2,
        float r,
        Vec3f& wi)
{
    float cosThetaI;
    switch(type) {
    case MICROFACET_DISTRIBUTION_BECKMANN:
        cosThetaI = pow(r1,
                        2.767f*r*r-5.4f*r+2.99412f);
        break;
    case MICROFACET_DISTRIBUTION_GGX:
    default:
        cosThetaI = pow(r1,
                        1.1933f*r*r-2.0969f*r+1.3698f);
    }

    const float sinThetaI = sqrt(1.0f - cosThetaI*cosThetaI);
    const float phiI = 2.0f * sPi * r2;

    const Vec3f m = computeLocalSphericalDirection(cosThetaI, sinThetaI, phiI);
    wi = localToGlobal(frame, m);
}

// PDF -
// Kelemen'01 Section 2.2.1
// (1-E(cosThetaI))/(pi*(1 - Eavg)) * cosThetaI
float
CookTorranceEnergyCompensation_pdf(
        MicrofacetDistribution type,
        float cosNI,
        float roughness)
{
    float result = 0.0f;
    const float norm = sPi * ReflectionAlbedo_oneMinusAvg(type,
                                                           roughness);
    if (isZero(norm)) return 0.0f;
    result = (cosNI * (1.0f - ReflectionAlbedo_E(type,
                                                  cosNI,
                                                  roughness)));
    result /= norm;
    return max(0.0f, result);
}

float
CookTorranceEnergyCompensation_weight(
        float roughness)
{
    float w = max(0.0f, (roughness-0.5f))*1.0f;
    return w;
}


