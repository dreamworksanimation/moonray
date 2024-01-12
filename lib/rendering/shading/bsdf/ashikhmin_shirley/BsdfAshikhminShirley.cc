// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfAshikhminShirley.cc
/// $Id$
///


#include "BsdfAshikhminShirley.h"

#include <moonray/rendering/shading/Util.h>
#include <scene_rdl2/common/platform/IspcUtil.h>

namespace moonray {
namespace shading {


using namespace scene_rdl2::math;


// These are for numerical stability and consistency across sample() / eval()
    static const float sCosGrazingAngleSample = scene_rdl2::math::cos((90.0 - 0.11) / 180.0 * double(pi));
static const float sCosGrazingAngle = scene_rdl2::math::cos((90.0 - 0.1) / 180.0 * double(pi));

static const float sSinNormalAngleSqrd = 1e-7;
static const float sCosNormalAngleSample = scene_rdl2::math::sqrt(1.0 - (sSinNormalAngleSqrd * 2.0));


// Hardcoded constants
static const float sDiffuseConstant = 28.0 / (23.0 * double(pi));


//----------------------------------------------------------------------------

AshikhminShirleyGlossyBsdfLobe::AshikhminShirleyGlossyBsdfLobe(
        const Vec3f &N, const Vec3f &anisoDirection,
        float roughnessU, float roughnessV) :
    BsdfLobe(Type(REFLECTION | GLOSSY), DifferentialFlags(0), false,
             PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
    mFrame(N, anisoDirection),
    mInputRoughnessU(roughnessU),
    mInputRoughnessV(roughnessV)
{
    // Apply roughness squaring to linearize roughness response
    // See "Physically-Based Shading at Disney" Siggraph course notes.
    roughnessU *= roughnessU;
    roughnessV *= roughnessV;

    // Convert roughness to exponent
    mExponentU = shading::roughness2Exponent(roughnessU);
    mExponentV = shading::roughness2Exponent(roughnessV);

    mScaleFactor = scene_rdl2::math::sqrt((mExponentU + 1.0f) * (mExponentV + 1.0f)) / (8.0f * sPi);

    mSampleFactor = scene_rdl2::math::sqrt((mExponentU + 1.0f) / (mExponentV + 1.0f));

    // Use the sharper of the two roughnesses to derive a directional differential
    // scale that varies with roughness
    float minRoughness = min(roughnessU, roughnessV);
    mdDFactor = sdDFactorMin + minRoughness * sdDFactorSlope;

    // TODO: Set lobe category based on roughness
}

//----------------------------------------------------------------------------
bool
AshikhminShirleyGlossyBsdfLobe::getProperty(Property property, float *dest) const
{
    bool handled = true;

    switch (property)
    {
    case PROPERTY_ROUGHNESS:
        *dest       = mInputRoughnessU;
        *(dest + 1) = mInputRoughnessV;
        break;
    case PROPERTY_NORMAL:
        {
            const Vec3f &N = mFrame.getN();
            *dest       = N.x;
            *(dest + 1) = N.y;
            *(dest + 2) = N.z;
        }
        break;
    default:
        handled = BsdfLobe::getProperty(property, dest);
        break;
    }

    return handled;
}

//----------------------------------------------------------------------------

Color
AshikhminShirleyGlossyBsdfLobe::eval(const BsdfSlice &slice, const Vec3f &wi,
        float *pdf) const
{
    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    // Compute normalized half-vector
    Vec3f H;
    if (!computeNormalizedHalfVector(slice.getWo(), wi, H)) {
        return Color(zero);
    }

    float NdotWi = dot(mFrame.getN(), wi);
    float maxNdotW = max(dot(mFrame.getN(), slice.getWo()), NdotWi);
    if (maxNdotW < sCosGrazingAngle) {
        return Color(zero);
    }

    float HdotN = dot(H, mFrame.getN());
    if (HdotN < sCosGrazingAngle) {
        return Color(zero);
    }

    float sinThetaSqrd = 1.0f - HdotN * HdotN;
    if (sinThetaSqrd < sSinNormalAngleSqrd) {
        return Color(zero);
    }

    float HdotWi = dot(H, wi);
    HdotWi = max(HdotWi, sEpsilon);

    float HdotX = dot(H, mFrame.getX());
    float HdotY = dot(H, mFrame.getY());

    float exponent = (mExponentU * HdotX * HdotX +
                      mExponentV * HdotY * HdotY) / sinThetaSqrd;
    float common = mScaleFactor * scene_rdl2::math::pow(HdotN, exponent);


    // Compute bsdf contribution
    float result = common / (HdotWi * maxNdotW);

    Color f = result * computeScaleAndFresnel(HdotWi) *
            (slice.getIncludeCosineTerm()  ?  max(NdotWi, 0.0f)  :  1.0f);


    // Compute pdf
    if (pdf != NULL) {
        float HdotWo = HdotWi;
        *pdf = common / HdotWo;
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = slice.computeShadowTerminatorFix(mFrame.getN(), wi);

    return Gs * f;
}


finline static float
samplePhi(float r2, float scale)
{
    float phi = scene_rdl2::math::atan(scale * scene_rdl2::math::tan(sHalfPi * r2));

    return phi;
}


Color
AshikhminShirleyGlossyBsdfLobe::sample(const BsdfSlice &slice, float r1, float r2,
        Vec3f &wi, float &pdf) const
{
    // Sample phi according to equation (9)
    float phi;
    if (r2 < 0.25f) {
        r2 = 4.0f * r2;
        phi = samplePhi(r2, mSampleFactor);
    } else if (r2 < 0.5f) {
        r2 = 4.0f * (0.5f - r2);
        phi = sPi - samplePhi(r2, mSampleFactor);
    } else if (r2 < 0.75f) {
        r2 = 1.0f - 4.0f * (0.75f - r2);
        phi = sPi + samplePhi(r2, mSampleFactor);
    } else {
        r2 = 4.0f * (1.0f - r2);
        phi = sTwoPi - samplePhi(r2, mSampleFactor);
    }
    float cosPhi;
    float sinPhi;
    sincos(phi, &sinPhi, &cosPhi);

    // Sample theta according to equation (10)
    // Note: Above this value for r1 and the math below sends samples in
    // regions where the brdf decreases very sharply, causing a lot of noise
    // with sharper glossy reflections. This only introduces a very small error.
    // TODO: we should probably stay away from the Bsdf due to its poor
    // normalization properties, so I won't spend too much time figuring this
    // one out at this point.
    float exponent = mExponentU * cosPhi * cosPhi +
                     mExponentV * sinPhi * sinPhi;
    float cosTheta = scene_rdl2::math::pow(1.0f - r1, 1.0f / (exponent + 1.0f));
    if (cosTheta < sCosGrazingAngleSample  ||  cosTheta > sCosNormalAngleSample) {
        pdf = 0.0f;
        return Color(zero);
    }
    float cosThetaSqrd = cosTheta * cosTheta;
    float sinTheta = scene_rdl2::math::sqrt(1.0f - cosThetaSqrd);

    // Map the canonical half vector into the local space
    Vec3f H(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    MNRY_ASSERT(isNormalized(H));
    H = mFrame.localToGlobal(H);

    // Compute reflection direction about the half vector
    float HdotWo = computeReflectionDirection(H, slice.getWo(), wi);

    // If the sampled microfacet is facing away from wo, we return a
    // zero probability sample
    // TODO: Should this be using: HdotWo = abs(HdotWo);
    if (HdotWo < sCosGrazingAngleSample) {
        pdf = 0.0f;
        return Color(zero);
    }

    float NdotWi = dot(mFrame.getN(), wi);
    float maxNdotW = max(dot(mFrame.getN(), slice.getWo()), NdotWi);
    if (maxNdotW < sCosGrazingAngleSample) {
        pdf = 0.0f;
        return Color(zero);
    }

    float common = mScaleFactor * scene_rdl2::math::pow(cosTheta, exponent);

    // Compute the probability of selecting wi
    pdf = common / HdotWo;

    // Compute the brdf
    float result = pdf / maxNdotW;

    // Apply scale and fresnel if any
    float HdotWi = HdotWo;
    Color f = result * computeScaleAndFresnel(HdotWi) *
            (slice.getIncludeCosineTerm()  ?  max(NdotWi, 0.0f)  :  1.0f);

    return f;
}


void
AshikhminShirleyGlossyBsdfLobe::differentials(const Vec3f &wo, const Vec3f &wi,
        float r1, float r2, const Vec3f &dNdx, const Vec3f &dNdy,
        Vec3f &dDdx, Vec3f &dDdy) const
{
    // It's complex to capture the full derivative. Instead we use the
    // derivative of a mirror reflection about the H vector, and scale the
    // length of the directional derivative proportionally with roughness.
    Vec3f H;
    if (!computeNormalizedHalfVector(wo, wi, H)) {
        H = mFrame.getN();
    }
    computeReflectionDirectionDifferential(wo, wi, H, dNdx, dNdy, dDdx, dDdy);
    dDdx *= mdDFactor;
    dDdy *= mdDFactor;
}


//----------------------------------------------------------------------------

AshikhminShirleyDiffuseBsdfLobe::AshikhminShirleyDiffuseBsdfLobe(const Vec3f &N) :
    BsdfLobe(Type(REFLECTION | DIFFUSE), DifferentialFlags(IGNORES_INCOMING_DIFFERENTIALS), false,
             PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
    mFrame(N)
{
}

//----------------------------------------------------------------------------
bool
AshikhminShirleyDiffuseBsdfLobe::getProperty(Property property, float *dest) const
{
    bool handled = true;

    switch (property)
    {
    case PROPERTY_ROUGHNESS:
        *dest       = 1.0f;
        *(dest + 1) = 1.0f;
        break;
    case PROPERTY_NORMAL:
        {
            const Vec3f &N = mFrame.getN();
            *dest       = N.x;
            *(dest + 1) = N.y;
            *(dest + 2) = N.z;
        }
        break;
    default:
        handled = BsdfLobe::getProperty(property, dest);
        break;
    }

    return handled;
}

Color
AshikhminShirleyDiffuseBsdfLobe::eval(const BsdfSlice &slice, const Vec3f &wi,
        float *pdf) const
{
    float tmp, tmp2;

    float NdotWo = max(dot(mFrame.getN(), slice.getWo()), 0.0f);
    tmp = 1.0f - 0.5f * NdotWo;
    tmp2 = tmp * tmp;
    float powerWo = tmp2 * tmp2 * tmp;

    float NdotWi = max(dot(mFrame.getN(), wi), 0.0f);
    tmp = 1.0f - 0.5f * NdotWi;
    tmp2 = tmp * tmp;
    float powerWi = tmp2 * tmp2 * tmp;

    float result = sDiffuseConstant * (1.0f - powerWo) * (1.0f - powerWi);

    Color f = result * computeScaleAndFresnel(NdotWi) *
            (slice.getIncludeCosineTerm()  ?  NdotWi  :  1.0f);

    if (pdf != NULL) {
        *pdf = NdotWi * sOneOverPi;
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = slice.computeShadowTerminatorFix(mFrame.getN(), wi);

    return Gs * f;
}


//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

