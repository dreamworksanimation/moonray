// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TransmissionCookTorranceEnergyCompensation.cc
/// $Id$
///

/*
 * This file is used to compensate for energy loss from
 * microfacet transmission lobes as outlined in Walter'07.
 * We use the Kelemen'01 energy compensation BRDF as described in
 * Kulla'17 to be added to the regular microfacet lobe.
 * This file describes the energy to be added in reflection and
 * transmission to be perfectly energy 'preserving'.
 * It also provides a factor used to calculate the energy
 * distribution between the two that follows reciprocity and is based
 * on average fresnel reflectance parameterized by (etaT/etaI).
 * Ref:
 * Revisiting Physically Based Shading, PBS Kulla'17
 */

#include "TransmissionCookTorranceEnergyCompensation.h"
#include "TransmissionAlbedo.h"
#include <moonray/rendering/shading/Util.h>

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

// BRDF Compensation Term as outlined in
// Kelemen'01 Section 2.2
// (1 - E(cosThetaO)*(1-E(cosThetaI))/(1 - Eavg) * cosThetaI
Color
TransmissionCookTorranceEnergyCompensation::evalR(
        const float cosNO,
        const float cosNI,
        float roughness,
        float etaI, float etaT,
        float favg,
        float favgInv,
        bool includeCosineTerm)
{
    const float neta = etaT / etaI;

    float compensation = (1.0f - TransmissionAlbedo::E(neta, cosNO, roughness)) *
                         (1.0f - TransmissionAlbedo::E(neta, cosNI, roughness));
    const float divisor = TransmissionAlbedo::oneMinusAvg(neta, roughness);
    const float factor = computeFactor(favg, favgInv,
                                       etaI, etaT,
                                       roughness);
    float ratio = 1.0f;
    if (neta > 1.0f) {
        ratio = 1.0f - (1.0f - favg) * factor;
    } else {
        ratio = 1.0f - (1.0f - favgInv) * (1.0f - factor);
    }
    if (!scene_rdl2::math::isZero(divisor)) {
        compensation /= divisor;
    } else {
        compensation = 0.0f;
    }
    compensation *= sOneOverPi;

    const float  f = ratio * compensation *
                     (includeCosineTerm  ?  cosNI : 1.0f);

    return (sWhite * max(0.0f, f));
}

// BRDF Compensation Term as outlined in
// Kelemen'01 Section 2.2
// (1 - E(cosThetaO)*(1-E(cosThetaI))/(1 - Eavg) * cosThetaI
Color
TransmissionCookTorranceEnergyCompensation::evalT(
        const float cosNO,
        const float cosNI,
        float roughness,
        float etaI, float etaT,
        float favg,
        float favgInv,
        bool includeCosineTerm)
{
    const float neta = etaT / etaI;
    float n;
    const float divisor = TransmissionAlbedo::oneMinusAvg(1.0f/neta, roughness);
    const float factor = computeFactor(favg, favgInv,
                                       etaI, etaT,
                                       roughness);
    float ratio = 1.0f;
    if (etaI < etaT) {
        n = 1.0;
        ratio = (1.0f - favg) * factor;
    } else {
        n = etaT/etaI;
        ratio = (1.0f - favgInv) * (1.0f - factor);
    }

    float compensation =
            (1.0f - TransmissionAlbedo::E(neta, cosNO, roughness)) *
            (1.0f - TransmissionAlbedo::E(1.0f/neta, cosNI, roughness));
    if (!scene_rdl2::math::isZero(divisor)) {
        compensation /= divisor;
    } else {
        compensation = 0.0f;
    }
    compensation *= sOneOverPi;

    const float f = ratio * compensation * n * n *
                    (includeCosineTerm  ?  cosNI : 1.0f);

    return (scene_rdl2::math::sWhite * max(0.0f,f));
}

float
TransmissionCookTorranceEnergyCompensation::computeFactor(
        float favg,
        float favgInv,
        float etaI, float etaT,
        float roughness)
{
    const float neta = (etaI < etaT) ?
            etaT/etaI : etaI/etaT;
    const float divisor =
            TransmissionAlbedo::oneMinusAvg(neta,
                                            roughness);
    if (isZero(divisor))
        return 1.0f;

    float y = TransmissionAlbedo::oneMinusAvg(1.0f/neta,
                                              roughness);
    y = y / divisor;
    y *= neta * neta;
    const float t = y * (1.0f - favgInv) /
                    (1.0f - favg);

    const float x = max(0.0f, t / (1.0f + t));
    return x;
}


// Sampling Curves Determined by fitting a curve
// to tabulated CDFInverse data calculated using
// moonshine_devtools/compute_cdf based on the PDF
// outlined in Kelemen'01
void
TransmissionCookTorranceEnergyCompensation::sampleR(
        const scene_rdl2::math::Vec3f& wo,
        const scene_rdl2::math::ReferenceFrame& frame,
        float r1, float r2,
        Vec3f &wi,
        float r,
        float etaI, float etaT)
{
    float cosThetaI;
    if (etaI < etaT) {
        const float n = TransmissionAlbedo::netaRange(etaT/etaI);
        const float power = (0.133f*n+3.02f)*r*r+(-0.66f*n-5.12f)*r+3.009f;
        cosThetaI = pow(r1, power);
    } else {
        const float n = TransmissionAlbedo::netaRange(etaI/etaT);
        const float power = (0.38f*n+2.4f)*r*r+(-0.68f*n-4.34f)*r+2.57f;
        cosThetaI = pow(r1, power);
    }
    cosThetaI = scene_rdl2::math::clamp(cosThetaI, 0.0f, 0.99f);
    float sinThetaI =
            scene_rdl2::math::clamp((1.0f - cosThetaI * cosThetaI));
    if (sinThetaI > 0.0f)
        sinThetaI = sqrt(sinThetaI);
    else
        sinThetaI = 0.0f;
    const float phiI = 2.0f * sPi * r2;
    const Vec3f m = computeLocalSphericalDirection(cosThetaI,
                                                   sinThetaI,
                                                   phiI);
    wi = frame.localToGlobal(m);
}

// PDF -
// Kelemen'01 Section 2.2.1
// (1-E(cosThetaI))/(pi*(1 - Eavg)) * cosThetaI
float
TransmissionCookTorranceEnergyCompensation::pdfR(
        float cosNI,
        float etaI, float etaT,
        float roughness)
{
    const float neta = etaT/etaI;

    const float norm = sPi * TransmissionAlbedo::oneMinusAvg(neta,
                                                              roughness);
    const float result = (cosNI * (1.0f - TransmissionAlbedo::E(neta,
                                                                cosNI,
                                                                roughness)));
    return max(0.0f, result/norm);
}

float
TransmissionCookTorranceEnergyCompensation::weightR(
        float etaI, float etaT,
        float roughness)
{
    float w = 0.0f;
    // Distinct weights for enter and exit based on curve
    // fittings
    if (etaI < etaT) {
        w = (roughness-0.5f)*1.0f;
    } else {
        w = (roughness-0.3f)*1.0f;
    }
    return max(0.0f, w);
}

//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

