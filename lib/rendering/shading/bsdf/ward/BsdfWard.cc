// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfWard.cc
/// $Id$
///


#include "BsdfWard.h"
#include <moonray/rendering/shading/Util.h>

#include <math.h>

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

// These are for numerical stability
static const float sCosGrazingAngleSample = scene_rdl2::math::cos((90.0 - 0.11) / 180.0 * double(pi));
static const float sCosGrazingAngle = scene_rdl2::math::cos((90.0 - 0.1) / 180.0 * double(pi));

static const float sMinTheta = 0.1 / 180.0 * double(pi);
static const float sMinSinTheta = scene_rdl2::math::sin(sMinTheta);
static const float sMaxCosTheta = scene_rdl2::math::cos(sMinTheta);


//----------------------------------------------------------------------------

WardBsdfLobe::WardBsdfLobe(const Vec3f &N, const Vec3f &anisoDirection,
        float roughnessU, float roughnessV, bool squareRoughness) :
    BsdfLobe(Type(REFLECTION | GLOSSY),
             (DifferentialFlags)0,
             false,
             PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
    mFrame(N, anisoDirection)
{
    setRoughness(roughnessU, roughnessV, squareRoughness);
}


WardBsdfLobe::~WardBsdfLobe() { }

void
WardBsdfLobe::setRoughness(float roughnessU, float roughnessV, bool squareRoughness)
{
    mInputRoughnessU = roughnessU;
    mInputRoughnessV = roughnessV;

    // Apply roughness squaring to linearize roughness response
    // See "Physically-Based Shading at Disney" Siggraph course notes.
    if (squareRoughness) {
        roughnessU *= roughnessU;
        roughnessV *= roughnessV;
    }

    mRoughnessU = roughnessU;
    if (mRoughnessU < sEpsilon) {
        mRoughnessU = sEpsilon;
    }

    mRoughnessV = roughnessV;
    if (mRoughnessV < sEpsilon) {
        mRoughnessV = sEpsilon;
    }

    mInvRoughUSqrd = 1.0f / (mRoughnessU * mRoughnessU);
    mInvRoughVSqrd = 1.0f / (mRoughnessV * mRoughnessV);
    mScaleFactor = sOneOverFourPi / (mRoughnessU * mRoughnessV);

    // Use the sharper of the two roughnesses to derive a directional differential
    // scale that varies with roughness
    float minRoughness = min(mRoughnessU, mRoughnessV);
    mdDFactor = sdDFactorMin + minRoughness * sdDFactorSlope;

    // TODO: Set lobe category based on roughness
}


void
WardBsdfLobe::getRoughness(float &roughnessU, float &roughnessV) const
{
    roughnessU = mInputRoughnessU;
    roughnessV = mInputRoughnessV;
}


void
WardBsdfLobe::differentials(const Vec3f &wo, const Vec3f &wi,
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

bool
WardBsdfLobe::getProperty(Property property, float *dest) const
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

WardOriginalBsdfLobe::WardOriginalBsdfLobe(const Vec3f &N,
        const Vec3f &anisoDirection, float roughnessU, float roughnessV,
        bool squareRoughness) :
    WardBsdfLobe(N, anisoDirection, roughnessU, roughnessV, squareRoughness)
{
}


Color
WardOriginalBsdfLobe::eval(const BsdfSlice &slice, const Vec3f &wi, float *pdf) const
{
    // eval() == pdf()
    float result = this->pdf(slice.getWo(), wi);
    if (pdf != NULL) {
        *pdf = result;
    }

    // Apply scale and fresnel if any
    float HdotWi = 1.0f;
    if (getFresnel() != NULL) {
        Vec3f H;
        if (computeNormalizedHalfVector(slice.getWo(), wi, H)) {
            HdotWi = dot(H, wi);
        }
    }
    Color f = result * computeScaleAndFresnel(HdotWi) *
            (slice.getIncludeCosineTerm()  ?
             max(dot(mFrame.getN(), wi), 0.0f)  :  1.0f);

    // Soften hard shadow terminator due to shading normals
    const float Gs = slice.computeShadowTerminatorFix(mFrame.getN(), wi);

    return Gs * f;
}


float
WardOriginalBsdfLobe::pdf(const Vec3f &wo, const Vec3f &wi) const
{
    // Compute normalized half-vector
    Vec3f H;
    if (!computeNormalizedHalfVector(wo, wi, H)) {
        return 0.0f;
    }

    float cosTheta = dot(H, mFrame.getN());
    if (cosTheta < sCosGrazingAngle) {
        return 0.0f;
    }

    float HdotWo = dot(H, wo);
    if (HdotWo < sCosGrazingAngle) {
        return 0.0f;
    }

    cosTheta = min(cosTheta, sMaxCosTheta);
    float theta = scene_rdl2::math::acos(cosTheta);
    float sinThetaSqrd = 1.0f - cosTheta * cosTheta;
    // TODO: is sqrt(sinThetaSqrd) faster than sin(theta) ?
    float sinTheta = scene_rdl2::math::sqrt(sinThetaSqrd);

    float HdotX = dot(H, mFrame.getX());
    float HdotY = dot(H, mFrame.getY());

    // Careful: HdotX = cosPhi * sinTheta and HdotY = sinPhi * sinTheta
    // which means the expFactor term includes a sinThetaSqrd, so here we cancel
    // the sinThetaSqrd out to get to the equation described in the header.
    float expFactor = HdotX * HdotX * mInvRoughUSqrd +
                      HdotY * HdotY * mInvRoughVSqrd;
    expFactor *= -theta * theta / sinThetaSqrd;

    float proba = theta * mScaleFactor / (HdotWo * sinTheta) * scene_rdl2::math::exp(expFactor);
    MNRY_ASSERT(proba >= 0.0f);

    return proba;
}


Color
WardOriginalBsdfLobe::sample(const BsdfSlice &slice, float r1, float r2, Vec3f &wi,
        float &pdf) const
{
    // Sample phi and compute sin and cos
    // TODO: Why all this complexity to keep track of the quadrant ?
    // TODO: Should the anisotropy modify how phi is sampled at all ?
    float phi = sTwoPi * r2;
    float cosOrigPhi;
    float sinOrigPhi;
    sincos(phi, &sinOrigPhi, &cosOrigPhi);
    cosOrigPhi *= mRoughnessU;
    sinOrigPhi *= mRoughnessV;
    float cosSqrdPhi = 1.0f
            / (sinOrigPhi * sinOrigPhi / (cosOrigPhi * cosOrigPhi) + 1.0f);
    float sinSqrdPhi = 1.0f - cosSqrdPhi;
    float sinPhi = (sinOrigPhi < 0.0f ? -scene_rdl2::math::sqrt(sinSqrdPhi) : scene_rdl2::math::sqrt(sinSqrdPhi));
    float cosPhi = (cosOrigPhi < 0.0f ? -scene_rdl2::math::sqrt(cosSqrdPhi) : scene_rdl2::math::sqrt(cosSqrdPhi));

    // Sample theta
    float roughnessTerm = cosSqrdPhi * mInvRoughUSqrd + sinSqrdPhi * mInvRoughVSqrd;
    r1 = max(r1, sEpsilon);
    float theta = scene_rdl2::math::sqrt(-scene_rdl2::math::log(r1) / roughnessTerm);

    // Because we use the un-corrected sampling equations of the original
    // Ward paper, the resulting theta can be bigger than pi/2, which doesn't
    // make any sense in the microfacet theory. In this case we return a zero
    // probability sample
    if (theta >= sHalfPi) {
        pdf = 0.0f;
        return Color(zero);
    }

    float sinTheta;
    float cosTheta;
    sincos(theta, &sinTheta, &cosTheta);

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


    // Compute the probability of selecting wi
    theta = max(theta, sMinTheta);
    sinTheta = max(sinTheta, sMinSinTheta);
    float result = theta * mScaleFactor / (HdotWo * sinTheta) * r1;
    MNRY_ASSERT(result > 0.0f);

    // Apply scale and fresnel if any
    float HdotWi = HdotWo;
    Color f = result * computeScaleAndFresnel(HdotWi) *
            (slice.getIncludeCosineTerm()  ?
             max(dot(mFrame.getN(), wi), 0.0f)  :  1.0f);

    pdf = result;

    return f;
}


//----------------------------------------------------------------------------

WardCorrectedBsdfLobe::WardCorrectedBsdfLobe(const Vec3f &N,
        const Vec3f &anisoDirection, float roughnessU, float roughnessV,
        bool squareRoughness) :
    WardBsdfLobe(N, anisoDirection, roughnessU, roughnessV, squareRoughness)
{
}


Color
WardCorrectedBsdfLobe::eval(const BsdfSlice &slice, const Vec3f &wi,
        float *pdf) const
{
    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    // We don't need to normalize H when it only appears at equal power both
    // on the nominator and denominator of expFactor
    Vec3f H;
    if (pdf != NULL  ||  getFresnel() != NULL) {
        if (!computeNormalizedHalfVector(slice.getWo(), wi, H)) {
            return Color(zero);
        }
    } else {
        H = slice.getWo() + wi;
    }

    float NdotWo = dot(mFrame.getN(), slice.getWo());
    float NdotWi = dot(mFrame.getN(), wi);
    if (NdotWo < sCosGrazingAngle  ||  NdotWi < sCosGrazingAngle) {
        return Color(zero);
    }

    // TODO: careful, H may not be normalized here! But, if NdotO and NdotI
    // are above the threshold, so should normalized HdotN
    float HdotN = dot(H, mFrame.getN());
    if (HdotN < sCosGrazingAngle) {
        return Color(zero);
    }

    float HdotX = dot(H, mFrame.getX());
    float HdotY = dot(H, mFrame.getY());

    // Careful: HdotX = cosPhi * sinTheta and HdotY = sinPhi * sinTheta
    // which means the expFactor term already includes a sinThetaSqrd
    float expFactor = -(HdotX * HdotX * mInvRoughUSqrd +
                        HdotY * HdotY * mInvRoughVSqrd);
    float HdotNSqrd = HdotN * HdotN;
    expFactor /= HdotNSqrd;
    expFactor = mScaleFactor * scene_rdl2::math::exp(expFactor);

    float product = NdotWo * NdotWi;
    float result = expFactor / scene_rdl2::math::sqrt(product);
    MNRY_ASSERT(result >= 0.0f);

    // Apply scale and fresnel if any
    float HdotWi = 1.0f;
    if (getFresnel() != NULL) {
        HdotWi = dot(H, wi);
    }
    Color f = result * computeScaleAndFresnel(HdotWi) *
            (slice.getIncludeCosineTerm()  ?  max(NdotWi, 0.0f)  :  1.0f);

    // Compute pdf
    if (pdf != NULL) {
        float HdotWo = dot(H, slice.getWo());
        if (HdotWo < sCosGrazingAngle) {
            return Color(zero);
        }

        *pdf = expFactor / (HdotWo * HdotNSqrd * HdotN);
        MNRY_ASSERT(*pdf >= 0.0f);
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = slice.computeShadowTerminatorFix(mFrame.getN(), wi);

    return Gs * f;
}


Color
WardCorrectedBsdfLobe::sample(const BsdfSlice &slice, float r1, float r2, Vec3f &wi,
        float &pdf) const
{
    // Sample phi and compute sin and cos
    // This is equivalent (but faster) to using eq. (7)
    float phi = sTwoPi * r2;
    float cosOrigPhi;
    float sinOrigPhi;
    sincos(phi, &sinOrigPhi, &cosOrigPhi);
    cosOrigPhi *= mRoughnessU;
    sinOrigPhi *= mRoughnessV;
    float cosSqrdPhi = 1.0f
            / (sinOrigPhi * sinOrigPhi / (cosOrigPhi * cosOrigPhi) + 1.0f);
    float sinSqrdPhi = 1.0f - cosSqrdPhi;
    float sinPhi = (sinOrigPhi < 0.0f ? -scene_rdl2::math::sqrt(sinSqrdPhi) : scene_rdl2::math::sqrt(sinSqrdPhi));
    float cosPhi = (cosOrigPhi < 0.0f ? -scene_rdl2::math::sqrt(cosSqrdPhi) : scene_rdl2::math::sqrt(cosSqrdPhi));

    // Sample theta
    float roughnessTerm = cosSqrdPhi * mInvRoughUSqrd + sinSqrdPhi * mInvRoughVSqrd;
    r1 = max(r1, sEpsilon);
    float cosThetaSqrd = roughnessTerm / (roughnessTerm - scene_rdl2::math::log(r1));
    float cosTheta = scene_rdl2::math::sqrt(cosThetaSqrd);
    float sinTheta = scene_rdl2::math::sqrt(1.0f - cosThetaSqrd);
    MNRY_ASSERT(cosTheta > sEpsilon);

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


    // Compute the bsdf contribution
    float NdotWo = dot(mFrame.getN(), slice.getWo());
    float NdotWi = dot(mFrame.getN(), wi);
    if (NdotWo < sCosGrazingAngle  ||  NdotWi < sCosGrazingAngle) {
        pdf = 0.0f;
        return Color(zero);
    }

    float expFactor = mScaleFactor * r1;
    float product = NdotWo * NdotWi;
    float result = expFactor / scene_rdl2::math::sqrt(product);
    MNRY_ASSERT(result >= 0.0f);

    // Apply scale and fresnel if any
    float HdotWi = HdotWo;
    Color f = result * computeScaleAndFresnel(HdotWi) *
            (slice.getIncludeCosineTerm()  ?  max(NdotWi, 0.0f)  :  1.0f);


    // Compute the probability of selecting wi
    pdf = expFactor / (HdotWo * cosThetaSqrd * cosTheta);
    MNRY_ASSERT(pdf > 0.0f);

    return f;
}


//----------------------------------------------------------------------------

WardDuerBsdfLobe::WardDuerBsdfLobe(const Vec3f &N,
        const Vec3f &anisoDirection, float roughnessU, float roughnessV,
        bool squareRoughness) :
    WardBsdfLobe(N, anisoDirection, roughnessU, roughnessV, squareRoughness)
{
}


finline Color
WardDuerBsdfLobe::eval(const BsdfSlice &slice, const Vec3f &wi, const Vec3f &H,
        float *pdf) const
{
    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    float NdotWo = dot(mFrame.getN(), slice.getWo());
    float NdotWi = dot(mFrame.getN(), wi);
    if (NdotWo < sCosGrazingAngle  ||  NdotWi < sCosGrazingAngle) {
        return Color(zero);
    }

    float HdotN = dot(H, mFrame.getN());
    if (HdotN < sCosGrazingAngle) {
        return Color(zero);
    }

    float HdotX = dot(H, mFrame.getX());
    float HdotY = dot(H, mFrame.getY());

    // Careful: HdotX = cosPhi * sinTheta and HdotY = sinPhi * sinTheta
    // which means the expFactor term already includes a sinThetaSqrd
    float expFactor = -(HdotX * HdotX * mInvRoughUSqrd +
                        HdotY * HdotY * mInvRoughVSqrd);
    float HdotNSqrd = HdotN * HdotN;
    expFactor /= HdotNSqrd;
    expFactor = mScaleFactor * scene_rdl2::math::exp(expFactor);

    // PKC --
    // Description in the paper uses HpdotHp and HpdotN (i.e., using unnormalized H)
    // NOTE : The derivation for why this is can be found in
    // "The Halfway Vector Disk for BRDF Modeling" by Edwards et. al.
    // albeit that the particular context is slightly different.
    // We could also use unnormalized H in the exponential, but this really only causes
    // discrepancy in how roughness is represented.
    //
    // The actual equations used correspond to the technical report by Geisler-Moroder et. al
    // "Bounding the Albedo of the Ward Reflectance Model"
    // Eq. (23) for the BRDF eval.
    Vec3f Hp = slice.getWo() + wi;
    float HpdotN = dot(Hp, mFrame.getN());
    float HpdotHp = dot(Hp, Hp);
    float product = HpdotHp / (HpdotN * HpdotN * HpdotN * HpdotN);

    // Compute the brdf
    float result = 4.0f * expFactor * product  *
              (slice.getIncludeCosineTerm()  ?  max(NdotWi, 0.0f)  :  1.0f);
    MNRY_ASSERT(result >= 0.0f);

    float HdotWi = dot(H, wi);
    Color f = result * computeScaleAndFresnel(HdotWi);

    // Compute pdf.
    // We use "Notes on the Ward BRDF", Tech report 2005, by Bruce Walter:
    // - equations (6) and (7) for the sampling,
    // - equation  (9) for the sampling pdf
    if (pdf != NULL) {
        // TODO: Optimization:
        // - Compute pdf in terms of the bsdf
        float HdotWo = dot(H, slice.getWo());
        if (HdotWo < sCosGrazingAngle) {
            return Color(zero);
        }

        *pdf = expFactor / (HdotWo * HdotNSqrd * HdotN);
        MNRY_ASSERT(*pdf >= 0.0f);
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = slice.computeShadowTerminatorFix(mFrame.getN(), wi);

    return Gs * f;
}


Color
WardDuerBsdfLobe::eval(const BsdfSlice &slice, const Vec3f &wi,
        float *pdf) const
{
    // Compute normalized half-vector
    Vec3f H;
    if (!computeNormalizedHalfVector(slice.getWo(), wi, H)) {
        return Color(zero);
    }

    return eval(slice, wi, H, pdf);
}


Color
WardDuerBsdfLobe::sample(const BsdfSlice &slice, float r1, float r2, Vec3f &wi,
        float &pdf) const
{
    // Sample phi and compute sin and cos
    // This is equivalent (but faster) to using eq. (7)
    float phi = sTwoPi * r2;
    float cosOrigPhi;
    float sinOrigPhi;
    sincos(phi, &sinOrigPhi, &cosOrigPhi);
    cosOrigPhi *= mRoughnessU;
    sinOrigPhi *= mRoughnessV;
    float cosSqrdPhi = 1.0f
            / (sinOrigPhi * sinOrigPhi / (cosOrigPhi * cosOrigPhi) + 1.0f);
    float sinSqrdPhi = 1.0f - cosSqrdPhi;
    float sinPhi = (sinOrigPhi < 0.0f ? -scene_rdl2::math::sqrt(sinSqrdPhi) : scene_rdl2::math::sqrt(sinSqrdPhi));
    float cosPhi = (cosOrigPhi < 0.0f ? -scene_rdl2::math::sqrt(cosSqrdPhi) : scene_rdl2::math::sqrt(cosSqrdPhi));

    // Sample theta
    float roughnessTerm = cosSqrdPhi * mInvRoughUSqrd + sinSqrdPhi * mInvRoughVSqrd;
    r1 = max(r1, sEpsilon);
    float cosThetaSqrd = roughnessTerm / (roughnessTerm - scene_rdl2::math::log(r1));
    float cosTheta = scene_rdl2::math::sqrt(cosThetaSqrd);
    float sinTheta = scene_rdl2::math::sqrt(1.0f - cosThetaSqrd);
    MNRY_ASSERT(cosTheta > sEpsilon);

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

    // TODO: Optimizations:
    // - Compute pdf in terms of the bsdf
    // - Compute them both re-using values computed thus far
    return eval(slice, wi, &pdf);
}


//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

