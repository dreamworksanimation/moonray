// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file FabricBsdfLobe.cc
/// $Id$
///

#include "BsdfFabric.h"

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

///
/// @class DwaFabricBsdfLobe BsdfFabric.h <pbr/fabric/BsdfFabric.h>
/// @brief DreamWorks specific fabric specular - (1 - |sin(thetaH)|)^N
/// thetaH = 0.5*(thetaO+thetaI), theta wrt normal plane to hair tangent
/// Results in really sharp specular peaks with a long tail, preferable for fabrics.

// Precomputed Normalization Integrals for This Model
// Precomputed List of 1 / (4*pi*integral (1-sin(x))^N*(1-2*sin^2(x))), x from 0 to pi/4
// N from 0 to 30
static const std::vector<float>
precomputedDwaFabricNormalizationFactor = {
                                           0.41788f, 0.52691f, 0.64072f, 0.76163f, 0.88620f,
                                           1.00729f, 1.13352f, 1.26158f, 1.39352f, 1.52527f,
                                           1.64034f, 1.78520f, 1.91616f, 2.01158f, 2.18194f,
                                           2.27028f, 2.41961f, 2.54223f, 2.69755f, 2.81754f,
                                           2.98730f, 3.07603f, 3.22963f, 3.28170f, 3.48190f,
                                           3.61855f, 3.75108f, 3.83304f, 3.96912f, 4.09358f,
                                           4.27677f };

//----------------------------------------------------------------------------
// Constructor / Destructor
DwaFabricBsdfLobe::DwaFabricBsdfLobe(const Vec3f &N,
                                     const Vec3f &T,
                                     const Vec3f &threadDirection,
                                     const float threadElevation,
                                     const float roughness,
                                     const Color& color) :
                                               FabricBsdfLobe(N, T,
                                                              threadDirection, threadElevation,
                                                              roughness,
                                                              color)
{
    calculateNormalizationFactor();
}

// Initialize Specular Exponent and NormalizationFactor
void
DwaFabricBsdfLobe::calculateNormalizationFactor()
{
    // Roughness remapping, [0,1] -> [1,30]
    // Square the roughness for a visually linear effect
    const float rgh = 1.0f - mRoughness;
    mSpecularExponent = scene_rdl2::math::ceil(1.0f + 29.0f * rgh*rgh);

    mNormalizationFactor =
            precomputedDwaFabricNormalizationFactor[static_cast<int>(clamp(mSpecularExponent-1, 0.0f, 30.0f))];
}

// (1-|sin(thetaH)|)^N lobe
Color
DwaFabricBsdfLobe::eval(const BsdfSlice &slice,
                        const Vec3f &wi,
                        float *pdf) const
{
    // Prepare for early exit
    if (pdf != nullptr) {
        *pdf = 0.0f;
    }
    const float cosThetaWi = dot(mFrame.getN(), wi);
    if (cosThetaWi <= sEpsilon) return sBlack;
    const float cosThetaWo = dot(mFrame.getN(), slice.getWo());
    if (cosThetaWo <= sEpsilon) return sBlack;

    Vec3f H;
    if (!computeNormalizedHalfVector(slice.getWo(), wi, H)) {
        return scene_rdl2::math::sBlack;
    }

    const float cosHWo = dot(H, slice.getWo());
    if (cosHWo < sEpsilon)   return scene_rdl2::math::sBlack;

    // thetaH is wrt the normal plane, so this dot product equals cos(pi/2 - thetaH) = sin(thetaH)
    const float sinThetaH = clamp(dot(H, mThreadDirection), -.99f, .99f);
    const float oneMinusAbsSinThetaH    = (1.0f - fabs(sinThetaH));
    if (oneMinusAbsSinThetaH < sEpsilon) return scene_rdl2::math::sBlack;
    const float oneMinusAbsSinThetaHPow = scene_rdl2::math::pow(oneMinusAbsSinThetaH, mSpecularExponent);

    const float fabricBRDF = mNormalizationFactor * oneMinusAbsSinThetaHPow;

    // PDF for sampling theta_h is (n+1)/pi * powf(1-fabs(sinThetaH), n)
    if (pdf != nullptr) {
        *pdf = (mSpecularExponent+1.0f)*sOneOverPi*oneMinusAbsSinThetaHPow;
        // Convert the PDF from sampling thetaH to sampling thetaI
        // Divide by Jacobian (dOmegaI/dOmegaH)
        *pdf *= 0.25 * rcp(cosHWo);
    }

    const Color f =  computeScaleAndFresnel(cosThetaWo) *
                     fabricBRDF * mColor *
                     (slice.getIncludeCosineTerm() ? cosThetaWi : 1.0f);

    // Soften hard shadow terminator due to shading normals
    const float Gs = slice.computeShadowTerminatorFix(mFrame.getN(), wi);

    return Gs * f;
}

Color
DwaFabricBsdfLobe::sample(const BsdfSlice &slice,
                          float r1, float r2,
                          Vec3f &wi,
                          float &pdf) const
{
    const float cosNO = dot(mFrame.getN(), slice.getWo());
    if (cosNO <= sEpsilon) {
        pdf = 0.0f;
        return sBlack;
    }

    Vec3f omegaI;
    sampleVector(slice.getWo(), r1, r2, omegaI);

    // Convert back in Global Space
    wi = normalize(mFrame.localToGlobal(omegaI));

    return eval(slice, wi, &pdf);
}


void
DwaFabricBsdfLobe::sampleVector(const Vec3f& wo,
                                float r1, float r2,
                                Vec3f& wi) const
{
    // Theta is wrt the normal plane
    const float thetaO = safeASin((dot(mThreadDirection, wo)));

    // We sample thetaH and use it to calculate thetaI
    float thetaH = scene_rdl2::math::asin(1.0f - scene_rdl2::math::pow(r1, 1.0f/(mSpecularExponent+1.0f)));
    if (r2 < 0.5f) thetaH *= -1.0f;

    const float thetaI = 2.0f * thetaH - thetaO;
    float cosThetaI, sinThetaI;
    sincos(thetaI, &sinThetaI, &cosThetaI);

    // Phi Sampling
    const float phiM = sPi*r2;
    float cosPhi, sinPhi;
    sincos(phiM, &sinPhi, &cosPhi);

    // Theta wrt normal plane
    wi.x = sinThetaI;
    wi.y = cosThetaI * cosPhi;
    wi.z = cosThetaI * sinPhi;
}

} // namespace shading
} // namespace moonray

