// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfCookTorrance.cc
/// $Id$
///


#include "BsdfCookTorrance.h"
#include "energy_compensation/CookTorranceEnergyCompensation.h"
#include "energy_compensation/TransmissionCookTorranceEnergyCompensation.h"
#include <scene_rdl2/common/math/MathUtil.h>
#include <moonray/rendering/shading/Util.h>

#include <algorithm>

#include <iostream>

// Sample with a slightly widened roughness at grazing angles
#define PBR_CT_SAMPLE_WIDENED_ROUGHNESS 1

namespace moonray {
namespace shading {


using namespace scene_rdl2::math;


// The notation used in this module is based on the paper:
//      "Microfacet Models for Refraction through Rough Surfaces"
//      (Walter et al. - EGSR 2007)
//
// The formula for the reflection term (or BRDF) of the BSDF
// described in the paper is the following (eq. 20 with m == Hr):
// (Note: except for the Anisotropic Cook-Torrance)
//
//                       F(i, m) * G(i, o, m) * D(m)
// reflectance(i,o,n) = -----------------------------
//                           4 * |i.n| * |o.n|
//
// Where the Fresnel term is provided by the lobe Fresnel closure
//
// where:
//    i     - incoming (light) direction
//    o     - outgoing (view) direction
//    n     - surface normal
//    m     - half-direction for reflection
//    F     - Fresnel term
//    G     - shadow-masking function


//----------------------------------------------------------------------------

CookTorranceBsdfLobe::CookTorranceBsdfLobe(const Vec3f &N,
                                           float roughness,
                                           const Color& favg,
                                           const Color& favgInv,
                                           float etaI,
                                           float etaT,
                                           bool coupledWithTransmission) :
    BsdfLobe(Type(REFLECTION | GLOSSY), DifferentialFlags(0), false,
             PROPERTY_NORMAL | PROPERTY_ROUGHNESS | PROPERTY_PBR_VALIDITY),
    mFrame(N),
    mInputRoughness(roughness),
    mFavg(favg),
    mFavgInv(favgInv),
    mEtaI(etaI),
    mEtaT(etaT),
    mCoupledWithTransmission(coupledWithTransmission)
{
    // Apply roughness squaring to linearize roughness response
    // See "Physically-Based Shading at Disney" Siggraph course notes.
    mRoughness = mInputRoughness * mInputRoughness;
    if (mRoughness < 0.001f) {   // TODO: Smaller thresholds trigger nans
        mRoughness = 0.001f;
    }

    mInvRoughness = rcp(mRoughness);

    // Use a directional differential scale that varies with roughness
    mdDFactor = sdDFactorMin + mRoughness * sdDFactorSlope;
}

void
CookTorranceBsdfLobe::computeBeckmannMicrofacet(float r1, float r2,
                                                float cosNO,
                                                scene_rdl2::math::Vec3f &m) const
{
    // generate a random microfacet normal m (eq. 28, 29)
    // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
    // and sin(atan(x)) == x/sqrt(1+x^2)
    const float alpha = widenRoughness(mRoughness, cosNO);
    const float alpha2 = alpha * alpha;
    const float tanThetaMSqr = -alpha2 * scene_rdl2::math::log(1.0f - r1);
    const float tanThetaM = scene_rdl2::math::sqrt(tanThetaMSqr);
    const float cosThetaM = rsqrt(1.0f + tanThetaMSqr);
    const float sinThetaM = cosThetaM * tanThetaM;
    const float phiM = 2.0f * sPi * r2;

    // Compute the half vector
    m = computeLocalSphericalDirection(cosThetaM, sinThetaM, phiM);
    m = mFrame.localToGlobal(m);
}

Color
CookTorranceBsdfLobe::eval(const BsdfSlice &slice, const Vec3f &wi,
        float *pdf) const
{
    // Based on the paper:
    //      "Microfacet Models for Refraction through Rough Surfaces"
    //      (Walter et al. - EGSR 2007)

    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }
    
    float cosNO = dot(mFrame.getN(), slice.getWo());
    if (cosNO <= sEpsilon) return sBlack;
    cosNO = min(cosNO, sOneMinusEpsilon);

    float cosNI = dot(mFrame.getN(), wi);
    if (cosNI <= sEpsilon) return sBlack;
    cosNI = min(cosNI, sOneMinusEpsilon);

    // get half vector
    Vec3f m;
    if (!computeNormalizedHalfVector(slice.getWo(), wi, m)) {
        return sBlack;
    }

    const float cosMI       = dot(m, wi);
    if (cosMI <= sEpsilon) return sBlack;

    const float cosThetaM   = dot(m, mFrame.getN());
    if (cosThetaM <= sEpsilon) return sBlack;
    
    // eq. 25: calculate D(m):
    const float alpha2          = mRoughness * mRoughness;
    const float cosThetaM2      = cosThetaM * cosThetaM;
    const float cosThetaM4      = cosThetaM2 * cosThetaM2;
    const float minusTanThetaM2 = (cosThetaM2 - 1.0f) * rcp(cosThetaM2);
    const float D4              = 0.25f * scene_rdl2::math::exp(minusTanThetaM2 * rcp(alpha2)) *
                                  rcp(sPi * alpha2 * cosThetaM4);
    
    // eq. 26, 27: now calculate G1(i,m) and G1(o,m)
    const float ao = cosNO * mInvRoughness * rsqrt(1.0f - cosNO * cosNO);
    const float ai = cosNI * mInvRoughness * rsqrt(1.0f - cosNI * cosNI);
    const float G1o = (ao < 1.6f) ? (3.535f * ao + 2.181f * ao * ao) *
            rcp(1.0f + 2.276f * ao + 2.577f * ao * ao) : 1.0f;
    const float G1i = (ai < 1.6f) ? (3.535f * ai + 2.181f * ai * ai) *
            rcp(1.0f + 2.276f * ai + 2.577f * ai * ai) : 1.0f;
    const float G  = G1o * G1i;
    

    // Compute the Cook-Torrance bsdf
    Color f = computeScaleAndFresnel(cosMI) * G * D4 * rcp(cosNO) *
            (slice.getIncludeCosineTerm()  ?  1.0f  :  rcp(cosNI));

    const float w2 = energyCompensationWeight();
    if (w2 > 0.0f) {
        const Color compen = evalEnergyCompensation(ispc::MICROFACET_DISTRIBUTION_BECKMANN,
                                                    cosNO, cosNI,
                                                    slice.getIncludeCosineTerm());
        f += compen;
    }

    if (pdf != NULL) {
#if PBR_CT_SAMPLE_WIDENED_ROUGHNESS
        // Compute pdf of sampling wi using eq. (17) in [1]
        // pdf(wi) = D(m) * |m.n| / (4 * |o.m|)
        // We cannot re-use D4 computed above without adjusting for the widened
        // roughness we use in sample(), otherwise our pdf is biased
        const float alpha = widenRoughness(mRoughness, cosNO);
        const float alpha2 = alpha * alpha;
        const float D4 = 0.25f * scene_rdl2::math::exp(minusTanThetaM2 * rcp(alpha2)) *
                         rcp(sPi * alpha2 * cosThetaM4);
#endif
        const float p2 = energyCompensationPDF(ispc::MICROFACET_DISTRIBUTION_BECKMANN,
                                               cosNI);
        const float w1 = (1.0f - w2);
        const float p1 = D4 * cosThetaM * rcp(cosMI);
        // One Sample PDF Weight
        *pdf = (w1 * p1 + w2 * p2);
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = slice.computeShadowTerminatorFix(mFrame.getN(), wi);

    return Gs * f;
}

Color
CookTorranceBsdfLobe::sample(const BsdfSlice &slice, float r1, float r2,
        Vec3f &wi, float &pdf) const
{
    const float cosNO = dot(mFrame.getN(), slice.getWo());
    if (cosNO <= 0.0f) {
        wi = Vec3f(zero);
        pdf = 0.0f;
        return sBlack;
    }

    // One Sample between energy compensation and regular
    // microfacet distribution sampling
    const float w2 = energyCompensationWeight();
    if (r1 < w2) {
        r1 = r1 / w2;
        sampleEnergyCompensation(ispc::MICROFACET_DISTRIBUTION_BECKMANN,
                                 slice.getWo(), r1, r2, wi);
    } else {

        const float w1 = (1.0f - w2);
        r1 = (r1 - w2) / w1;

        Vec3f m;
        computeBeckmannMicrofacet(r1, r2, cosNO, m);
        // Compute reflection direction about the half vector
        computeReflectionDirection(m, slice.getWo(), wi);
    }

    Color brdf = eval(slice, wi, &pdf);
    return brdf;
}

float
CookTorranceBsdfLobe::energyCompensationWeight() const
{
    // Early Exit
    if (isBlack(mFavg)) return 0.0f;
    float w;
    if (mCoupledWithTransmission) {
        w = TransmissionCookTorranceEnergyCompensation::weightR(
                mEtaI, mEtaT,
                mInputRoughness);
    }
    else {
        w = CookTorranceEnergyCompensation::weight(
                mInputRoughness);
    }
    return w;
}

float
CookTorranceBsdfLobe::energyCompensationPDF(
        ispc::MicrofacetDistribution type,
        float cosNI) const
{
    // Early Exit
    if (isBlack(mFavg)) return 0.0f;
    float pdf;
    if (mCoupledWithTransmission) {
        pdf = TransmissionCookTorranceEnergyCompensation::pdfR(
                cosNI,
                mEtaI, mEtaT,
                mInputRoughness);
    } else {
        pdf = CookTorranceEnergyCompensation::pdf(
                type,
                cosNI,
                mInputRoughness);
    }
    return pdf;
}

void
CookTorranceBsdfLobe::sampleEnergyCompensation(
        ispc::MicrofacetDistribution type,
        const scene_rdl2::math::Vec3f& wo,
        float r1, float r2,
        scene_rdl2::math::Vec3f& wi) const
{
    MNRY_ASSERT(!isBlack(mFavg),
               "Should not be called with a black Favg\n");
    if (mCoupledWithTransmission) {
        TransmissionCookTorranceEnergyCompensation::sampleR(
                wo,
                mFrame,
                r1, r2,
                wi,
                mInputRoughness,
                mEtaI, mEtaT);
    } else {
        CookTorranceEnergyCompensation::sample(type,
                                               mFrame,
                                               r1, r2,
                                               mInputRoughness,
                                               wi);
    }
}

Color
CookTorranceBsdfLobe::evalEnergyCompensation(
        ispc::MicrofacetDistribution type,
        float cosNO, float cosNI,
        bool includeCosineTerm) const
{
    // Early Exit
    if (isBlack(mFavg)) return sBlack;
    Color c;
    if (mCoupledWithTransmission) {
        c = TransmissionCookTorranceEnergyCompensation::evalR(
                cosNO, cosNI,
                mInputRoughness,
                mEtaI, mEtaT,
                mFavg[0], mFavgInv[0],
                includeCosineTerm);
    } else {
        c = CookTorranceEnergyCompensation::eval(
                type,
                cosNO, cosNI,
                mInputRoughness,
                mFavg,
                includeCosineTerm);
    }
    return c;
}

void
CookTorranceBsdfLobe::differentials(const Vec3f &wo, const Vec3f &wi,
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
CookTorranceBsdfLobe::getProperty(Property property, float *dest) const
{
    bool handled = true;

    switch (property)
    {
    case PROPERTY_ROUGHNESS:
        *dest       = mInputRoughness;
        *(dest + 1) = mInputRoughness;
        break;
    case PROPERTY_NORMAL:
        {
            const Vec3f &N = mFrame.getN();
            *dest       = N.x;
            *(dest + 1) = N.y;
            *(dest + 2) = N.z;
        }
        break;
    case PROPERTY_PBR_VALIDITY:
        {
            const Fresnel* fresnel = getFresnel();
            // Get the property from the fresnel object
            Color res = scene_rdl2::math::sBlack;
            if (fresnel) {
                res = fresnel->computePbrValidity();
            }
            *dest       = res.r;
            *(dest + 1) = res.g;
            *(dest + 2) = res.b;
        }
        break;
    default:
        handled = BsdfLobe::getProperty(property, dest);
        break;
    }

    return handled;
}

void
TransmissionCookTorranceBsdfLobe::differentials(const Vec3f &wo, const Vec3f &wi,
        float r1, float r2, const Vec3f &dNdx, const Vec3f &dNdy,
        Vec3f &dDdx, Vec3f &dDdy) const
{
    // It's complex to capture the full derivative. Instead we use the
    // derivative of a mirror refraction about the H vector, and scale the
    // length of the directional derivative proportionally with roughness.
    Vec3f H;
    if (computeNormalizedRefractionHalfVector(mEtaI, wo,
                                              mEtaT, wi, H)) {
        // Make sure microfacet points to the same side of the surface wrt. N and wo
        H = (mEtaI > mEtaT  ?  -H  :  H);
    } else {
        H = mFrame.getN();
    }
    computeRefractionDirectionDifferential(mNeta, wo, wi, H, dNdx, dNdy,
            dDdx, dDdy);
    dDdx *= mdDFactor;
    dDdy *= mdDFactor;
}

//----------------------------------------------------------------------------
GGXCookTorranceBsdfLobe::GGXCookTorranceBsdfLobe(const Vec3f &N,
        float roughness,
        const Color& favg,
        const Color& favgInv,
        float etaI,
        float etaT,
        bool coupledWithTransmission) :
    CookTorranceBsdfLobe(N, roughness, favg, favgInv,
                         etaI, etaT, coupledWithTransmission)
{
}

Color
GGXCookTorranceBsdfLobe::eval(const BsdfSlice &slice, const Vec3f &wi,
                              float *pdf, const float cosNO, const float cosNI,
                              const ReferenceFrame& frame) const
{
    // Based on the paper:
    //      "Microfacet Models for Refraction through Rough Surfaces"
    //      (Walter et al. - EGSR 2007)

    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    if (cosNO <= 0.0f || cosNI <= 0.0f) {
        return sBlack;
    }

    // get half vector
    Vec3f m;
    if (!computeNormalizedHalfVector(slice.getWo(), wi, m)) {
        return sBlack;
    }

    const float cosMI = dot(m, wi);
    if (cosMI <= 0.0f) return sBlack;

    const float cosThetaM = dot(m, frame.getN());
    if (cosThetaM <= 0.0f) return sBlack;

    // eq. 33: calculate D(m):
    const float alpha2 = mRoughness * mRoughness;
    const float cosThetaM2 = cosThetaM * cosThetaM;
    const float piCosThetaM4 = sPi * cosThetaM2 * cosThetaM2;
    const float tanThetaM2 = (1.0f - cosThetaM2) / cosThetaM2;
    const float tmp = alpha2 + tanThetaM2;
    const float D4 = 0.25f * alpha2 / (piCosThetaM4 * tmp * tmp);

    // eq. 34: now calculate G1(i,m) and G1(o,m)
    const float invCosNO = 1.0f / cosNO;
    const float invCosNI = 1.0f / cosNI;
    const float G1o = 2.0f / (1.0f + scene_rdl2::math::sqrt(1.0f + alpha2 *
                        (1.0f - cosNO * cosNO) * (invCosNO * invCosNO)));
    const float G1i = 2.0f / (1.0f + scene_rdl2::math::sqrt(1.0f + alpha2 *
                        (1.0f - cosNI * cosNI) * (invCosNI * invCosNI)));
    const float G = G1o * G1i;

    // Compute the Cook-Torrance bsdf
    Color f = computeScaleAndFresnel(cosMI) * G * D4 * invCosNO *
            (slice.getIncludeCosineTerm()  ?  1.0f  :  invCosNI);

    const float w2 = energyCompensationWeight();
    if (w2 > 0.0f) {
        const Color compen = evalEnergyCompensation(ispc::MICROFACET_DISTRIBUTION_GGX,
                                                    cosNO, cosNI,
                                                    slice.getIncludeCosineTerm());
        f = f + compen;
    }

    // Compute pdf of sampling wi using eq. (17) in [1]
    // pdf(wi) = D(m) * |m.n| / (4 * |o.m|)
    if (pdf != NULL) {
#if PBR_CT_SAMPLE_WIDENED_ROUGHNESS
        // We cannot re-use D4 computed above without adjusting for the widened
        // roughness we use in sample(), otherwise our pdf is biased
        const float alpha = widenRoughness(mRoughness, cosNO);
        const float alpha2 = alpha * alpha;
        const float tmp = alpha2 + tanThetaM2;
        const float D4 = 0.25f * alpha2 / (piCosThetaM4 * tmp * tmp);
#endif
        const float p2 = energyCompensationPDF(ispc::MICROFACET_DISTRIBUTION_GGX,
                                               cosNI);
        const float w1 = (1.0f - w2);
        const float p1 = D4 * cosThetaM * rcp(cosMI);
        // One Sample PDF Weight
        *pdf = (w1 * p1 + w2 * p2);
    }

    return f;
}

Vec3f
GGXCookTorranceBsdfLobe::sample(const BsdfSlice &slice,
                                float cosNO,
                                float r1, float r2,
                                const ReferenceFrame& frame) const
{
    // One Sample between energy compensation and regular
    // microfacet distribution sampling
    Vec3f wi;
    // generate a random microfacet normal m (eq. 35,36):
    // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
    // and sin(atan(x)) == x/sqrt(1+x^2)
    const float alpha = widenRoughness(mRoughness, cosNO);
    const float alpha2 = alpha * alpha;
    const float tanThetaM2  = alpha2 * r1 / (1.0f - r1);
    const float cosThetaM   = 1.0f / scene_rdl2::math::sqrt(1.0f + tanThetaM2);
    const float sinThetaM   = cosThetaM * scene_rdl2::math::sqrt(tanThetaM2);
    const float phiM = sTwoPi * r2;

    // Compute the half vector
    Vec3f m = computeLocalSphericalDirection(cosThetaM, sinThetaM, phiM);
    m = frame.localToGlobal(m);

    computeReflectionDirection(m, slice.getWo(), wi);
    return wi;
}

Color
GGXCookTorranceBsdfLobe::eval(const BsdfSlice &slice, const Vec3f &wi,
        float *pdf) const
{
    const float cosNO = dot(mFrame.getN(), slice.getWo());
    const float cosNI = dot(mFrame.getN(), wi);

    // Soften hard shadow terminator due to shading normals
    // The code is here to avoid applying the softening in the case of glitter
    const float Gs = slice.computeShadowTerminatorFix(mFrame.getN(), wi);

    return Gs * eval(slice, wi, pdf, cosNO, cosNI, mFrame);
}

Color
GGXCookTorranceBsdfLobe::sample(const BsdfSlice &slice, float r1, float r2,
        Vec3f &wi, float &pdf) const
{
    const float cosNO = dot(mFrame.getN(), slice.getWo());
    if (cosNO <= 0.0f) {
        wi = Vec3f(zero);
        pdf = 0.0f;
        return sBlack;
    }

    const float w2 = energyCompensationWeight();
    if (r1 < w2) {
        r1 = r1 / w2;
        sampleEnergyCompensation(ispc::MICROFACET_DISTRIBUTION_GGX,
                                 slice.getWo(), r1, r2, wi);
    } else {

        const float w1 = (1.0f - w2);
        r1 = (r1 - w2) / w1;
        wi = sample(slice, cosNO, r1, r2, mFrame);
    }

    return eval(slice, wi, &pdf);
}


//----------------------------------------------------------------------------

GlitterGGXCookTorranceBsdfLobe::GlitterGGXCookTorranceBsdfLobe(const Vec3f &N,
                                                               const Vec3f &flakeN,
                                                               float roughness,
                                                               const Color& favg) :
    GGXCookTorranceBsdfLobe(N, roughness, favg),
    mFlakeNormal(flakeN)
{
}

Color
GlitterGGXCookTorranceBsdfLobe::eval(const BsdfSlice &slice, const Vec3f &wi,
                                     float *pdf) const
{
    // Try evaluation with the flake normal
    float cosNO = dot(mFlakeNormal, slice.getWo());
    float cosNI = dot(mFlakeNormal, wi);

    ReferenceFrame frame;
    // If flake normal is too perturbed, evaluate with shading normal
    if (cosNO <= 0.0f || cosNI <= 0.0f) {
        frame = mFrame;
        cosNO = dot(frame.getN(), slice.getWo());
        cosNI = dot(frame.getN(), wi);
    } else {
        frame = ReferenceFrame(mFlakeNormal);
    }

    return GGXCookTorranceBsdfLobe::eval(slice, wi, pdf, cosNO, cosNI, frame);
}

Color
GlitterGGXCookTorranceBsdfLobe::sample(const BsdfSlice &slice, float r1, float r2,
                                       Vec3f &wi, float &pdf) const
{
    // Try sampling with the flake normal
    bool isUsingFlakeNormal = true;
    float cosNO = dot(mFlakeNormal, slice.getWo());

    ReferenceFrame frame;
    // If wo is outside the hemisphere of the flake normal, try sampling with shading normal
    if (cosNO <= 0.0f) {
        frame = mFrame;
        cosNO = dot(frame.getN(), slice.getWo());
        isUsingFlakeNormal = false;
        // Unsuccessful even with shading normal
        if (cosNO <= 0.0f) {
            wi = Vec3f(zero);
            pdf = 0.0f;
            return sBlack;
        }
    } else {
        frame = ReferenceFrame(mFlakeNormal);
    }

    wi = GGXCookTorranceBsdfLobe::sample(slice, cosNO, r1, r2, frame);

    float cosNI = dot(frame.getN(), wi);
    // If wi is created with flake normal, check if it is within the hemisphere of shading normal
    // If not, generate another wi using the shading normal instead
    if (isUsingFlakeNormal) {
        const float cosNIShading = dot(mFrame.getN(), wi);
        if (cosNIShading <= 0.0f) {
            frame = mFrame;
            cosNO = dot(frame.getN(), slice.getWo());
            // Unsuccessful even with shading normal
            if (cosNO <= 0.0f) {
                wi = Vec3f(zero);
                pdf = 0.0f;
                return sBlack;
            }
            wi = GGXCookTorranceBsdfLobe::sample(slice, cosNO, r1, r2, frame);
            cosNI = dot(frame.getN(), wi);
        }
    }

    return GGXCookTorranceBsdfLobe::eval(slice, wi, &pdf, cosNO, cosNI, frame);
}

bool
GlitterGGXCookTorranceBsdfLobe::getProperty(Property property, float *dest) const
{
    bool handled = true;

    switch (property)
    {
    case PROPERTY_NORMAL:
        {
            const Vec3f &N = mFlakeNormal;
            *dest       = N.x;
            *(dest + 1) = N.y;
            *(dest + 2) = N.z;
        }
        break;
    default:
        handled = CookTorranceBsdfLobe::getProperty(property, dest);
        break;
    }

    return handled;
}

//----------------------------------------------------------------------------

BerryCookTorranceBsdfLobe::BerryCookTorranceBsdfLobe(const Vec3f &N,
        float roughness) :
    CookTorranceBsdfLobe(N, min(roughness, 0.99f))
{
}


Color
BerryCookTorranceBsdfLobe::eval(const BsdfSlice &slice, const Vec3f &wi,
        float *pdf) const
{
    // Based on the paper:
    //      "Physically-Based Shading at Disney"
    //      (Burley et al. - Siggraph Course 2012)

    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }
    
    const float cosNO = dot(mFrame.getN(), slice.getWo());
    if (cosNO <= 0.0f)      return sBlack;

    const float cosNI = dot(mFrame.getN(), wi);
    if (cosNI <= 0.0f)      return sBlack;
    
    // get half vector
    Vec3f m;
    if (!computeNormalizedHalfVector(slice.getWo(), wi, m)) {
        return sBlack;
    }
    
    const float cosMI       = dot(m, wi);
    if (cosMI <= 0.0f)      return sBlack;

    const float cosThetaM   = dot(m, mFrame.getN());
    if (cosThetaM <= 0.0f)  return sBlack;
    
    // eq. "D_Berry": calculate D(m) (with m=Hr):
    const float alpha2      = mRoughness * mRoughness;
    const float cosThetaM2  = cosThetaM * cosThetaM;
    const float sinThetaM2  = 1.0f - cosThetaM2;
    const float d0          = (alpha2 - 1.0f) / (sPi * scene_rdl2::math::log(alpha2));
    const float d1          = 1.0f / (alpha2 * cosThetaM2 + sinThetaM2);
    const float D4          = 0.25f * d0 * d1;

    // Per section 5.6 "Specular G details", use GGX shadowing and masking functions
    // However, we don't seem to use the remapping: alpha2 = (0.5 + mRoughness / 2)^2
    const float invCosNO = 1.0f / cosNO;
    const float invCosNI = 1.0f / cosNI;
    const float G1o = 2.0f / (1.0f + scene_rdl2::math::sqrt(1.0f + alpha2 *
                        (1.0f - cosNO * cosNO) * (invCosNO * invCosNO)));
    const float G1i = 2.0f / (1.0f + scene_rdl2::math::sqrt(1.0f + alpha2 *
                        (1.0f - cosNI * cosNI) * (invCosNI * invCosNI)));
    const float G = G1o * G1i;

    // Compute the Cook-Torrance bsdf
    const Color f = computeScaleAndFresnel(cosMI) * G * D4 * invCosNO *
            (slice.getIncludeCosineTerm()  ?  1.0f  :  invCosNI);

    // Compute pdf of sampling wi using eq. (17) in [1]
    // pdf(wi) = D(m) * |m.n| / (4 * |o.m|)
    if (pdf != NULL) {
#if PBR_CT_SAMPLE_WIDENED_ROUGHNESS
        // We cannot re-use D4 computed above without adjusting for the widened
        // roughness we use in sample(), otherwise our pdf is biased
        const float alpha = widenRoughness(mRoughness, cosNO);
        const float alpha2 = alpha * alpha;
        const float d0          = (alpha2 - 1.0f) / (sPi * scene_rdl2::math::log(alpha2));
        const float d1          = 1.0f / (alpha2 * cosThetaM2 + sinThetaM2);
        const float D4          = 0.25f * d0 * d1;
#endif
        *pdf = D4 * cosThetaM / cosMI;
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = slice.computeShadowTerminatorFix(mFrame.getN(), wi);

    return Gs * f;
}


Color
BerryCookTorranceBsdfLobe::sample(const BsdfSlice &slice, float r1, float r2, Vec3f &wi,
        float &pdf) const
{
    const float cosNO = dot(mFrame.getN(), slice.getWo());
    if (cosNO <= 0.0f) {
        wi = Vec3f(zero);
        pdf = 0.0f;
        return sBlack;
    }

    // generate a random microfacet normal m (eq. 35,36):
    // using: tan(acos(x)) == sqrt(1-x^2) / x
    //        sin(x) == cos(x) * tan(x)
    const float alpha = widenRoughness(mRoughness, cosNO);
    const float alpha2 = alpha * alpha;
    const float cosThetaM = scene_rdl2::math::sqrt((1.0f - scene_rdl2::math::pow(alpha2, (1.0f - r1))) / (1.0f - alpha2));
    const float tanThetaM = scene_rdl2::math::sqrt(1.0f - cosThetaM * cosThetaM) / cosThetaM;
    const float sinThetaM = cosThetaM * tanThetaM;
    const float phiM = 2.0f * sPi * r2;

    Vec3f m = computeLocalSphericalDirection(cosThetaM, sinThetaM, phiM);
    // TODO: check and remove: m.normalize();
    m = mFrame.localToGlobal(m);

    // Compute reflection direction about the half vector
    computeReflectionDirection(m, slice.getWo(), wi);

    return eval(slice, wi, &pdf);
}


//----------------------------------------------------------------------------

AnisoCookTorranceBsdfLobe::AnisoCookTorranceBsdfLobe(const Vec3f &N,
        const Vec3f &anisoDirection, float roughnessU, float roughnessV) :
    CookTorranceBsdfLobe(N, roughnessU),
    mInputRoughnessV(roughnessV)
{
    // Apply roughness squaring to linearize roughness response
    // See "Physically-Based Shading at Disney" Siggraph course notes.
    mRoughnessV = mInputRoughnessV * mInputRoughnessV;
    if (mRoughnessV < 0.001) {   // TODO: Smaller thresholds trigger nans
        mRoughnessV = 0.001;
    }

    // Use the sharper of the two roughnesses to derive a directional differential
    // scale that varies with roughness
    const float minRoughness = min(mRoughness, mRoughnessV);
    mdDFactor = sdDFactorMin + minRoughness * sdDFactorSlope;

    mFrame = ReferenceFrame(N, anisoDirection);
}


Color
AnisoCookTorranceBsdfLobe::eval(const BsdfSlice &slice, const Vec3f &wi,
        float *pdf) const
{
    // Based on the papers:
    // [1]: "Notes on Ward BRDF"
    //      Bruce Walter, 2005, http://www.graphics.cornell.edu/~bjw/wardnotes.pdf
    // [2]: "Microfacet Models for Refraction through Rough Surfaces"
    //      Walter et al., EGSR 2007
    // [3]: "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
    //      Eric Heitz, Journal of Computer Graphics Techniques, JCGT, 2014

    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    const float cosNO = dot(mFrame.getN(), slice.getWo());
    if (cosNO <= 0.0f)      return sBlack;

    const float cosNI = dot(mFrame.getN(), wi);
    if (cosNI <= 0.0f)      return sBlack;
    
    // get half vector
    Vec3f m;
    if (!computeNormalizedHalfVector(slice.getWo(), wi, m)) {
        return sBlack;
    }

    const float cosMI       = dot(m, wi);
    if (cosMI <= 0.0f)      return sBlack;

    const float cosThetaM   = dot(m, mFrame.getN());
    if (cosThetaM <= 0.0f)  return sBlack;

    // Calculate D(m) (with m=Hr), using eq. (82) in [3]. Note that we use the
    // (exact and cheaper) vector form of the exponential term from eq. (4) in [1].
    const float dotMX = dot(m, mFrame.getX());
    const float dotMY = dot(m, mFrame.getY());
    const float cosThetaMX = dotMX / mRoughness;
    const float cosThetaMX2 = cosThetaMX * cosThetaMX;
    const float cosThetaMY = dotMY / mRoughnessV;
    const float cosThetaMY2 = cosThetaMY * cosThetaMY;
    const float cosThetaM2  = cosThetaM * cosThetaM;
    const float D4 = 0.25f * scene_rdl2::math::exp(-(cosThetaMX2 + cosThetaMY2) / cosThetaM2)
                  / (sPi * mRoughness * mRoughnessV * cosThetaM2 * cosThetaM2);

    // Calculate G1(i,m) and G1(o,m) using equations (83) and (80) in [3].
    // First we simplify the expression for a, also using the exact vector form
    // (similarly to how [1] derives the exponential term in eq. (4)):
    // a = (v.n) / sqrt(((v.x)*alphax)^2 + ((v.y)*alphay)^2)
    const float tmpXO = dot(mFrame.getX(), slice.getWo()) * mRoughness;
    const float tmpYO = dot(mFrame.getY(), slice.getWo()) * mRoughnessV;
    const float ao = cosNO / scene_rdl2::math::sqrt(tmpXO * tmpXO + tmpYO * tmpYO);
    const float tmpXI = dot(mFrame.getX(), wi) * mRoughness;
    const float tmpYI = dot(mFrame.getY(), wi) * mRoughnessV;
    const float ai = cosNI / scene_rdl2::math::sqrt(tmpXI * tmpXI + tmpYI * tmpYI);
    // Use the rational approximation of G1: eq. (27) in [2]
    const float G1o = (ao < 1.6f) ? (3.535f * ao + 2.181f * ao * ao) /
                                    (1.0f + 2.276f * ao + 2.577f * ao * ao) : 1.0f;
    const float G1i = (ai < 1.6f) ? (3.535f * ai + 2.181f * ai * ai) /
                                    (1.0f + 2.276f * ai + 2.577f * ai * ai) : 1.0f;
    const float G   = G1o * G1i;

    // Compute the Cook-Torrance bsdf
    const Color f = computeScaleAndFresnel(cosMI) * G * D4 / cosNO *
            (slice.getIncludeCosineTerm()  ?  1.0f  :  1.0f / cosNI);

    // Compute pdf of sampling wi using eq. (17) in [1]
    // pdf(wi) = D(m) * |m.n| / (4 * |o.m|)
    if (pdf != NULL) {
#if PBR_CT_SAMPLE_WIDENED_ROUGHNESS
        // We cannot re-use D4 computed above without adjusting for the widened
        // roughness we use in sample(), otherwise our pdf is biased
        const float alphaU = widenRoughness(mRoughness, cosNO);
        const float alphaV = widenRoughness(mRoughnessV, cosNO);
        const float cosThetaMX = dotMX / alphaU;
        const float cosThetaMX2 = cosThetaMX * cosThetaMX;
        const float cosThetaMY = dotMY / alphaV;
        const float cosThetaMY2 = cosThetaMY * cosThetaMY;
        const float D4 = 0.25f * scene_rdl2::math::exp(-(cosThetaMX2 + cosThetaMY2) / cosThetaM2)
                      / (sPi * alphaU * alphaV * cosThetaM2 * cosThetaM2);
#endif
        *pdf = D4 * cosThetaM / cosMI;
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = slice.computeShadowTerminatorFix(mFrame.getN(), wi);

    return Gs * f;
}


Color
AnisoCookTorranceBsdfLobe::sample(const BsdfSlice &slice, float r1, float r2,
        Vec3f &wi, float &pdf) const
{
    const float cosNO = dot(mFrame.getN(), slice.getWo());
    if (cosNO <= 0.0f) {
        wi = Vec3f(zero);
        pdf = 0.0f;
        return sBlack;
    }

    // We use the anisotropic sampling equations (6) and (7) in [1], which apply
    // directly here, since they are in fact sampling the anisotropic
    // Beckmann distribution we use here (which the Ward model is also based on).

    // For sampling, use the modified sampling distribution in sect 5.3 in [2]
    // Reduce maximum weight with widened distribution
    const float alphaU = widenRoughness(mRoughness, cosNO);
    const float alphaV = widenRoughness(mRoughnessV, cosNO);
    const float mInvAlphaUSqrd = 1.0f / (alphaU * alphaU);
    const float mInvAlphaVSqrd = 1.0f / (alphaV * alphaV);

    // Sample phi and compute sin and cos
    // This is equivalent (but faster) to using eq. (7) in [1].
    // Using the identity: cos(x)^2 = 1 / (1 + tan(x)^2)
    const float phi = sTwoPi * r2;
    const float cosOrigPhi = alphaU * scene_rdl2::math::cos(phi);
    const float sinOrigPhi = alphaV * scene_rdl2::math::sin(phi);
    const float cosSqrdPhi = 1.0f
            / (sinOrigPhi * sinOrigPhi / (cosOrigPhi * cosOrigPhi) + 1.0f);
    const float sinSqrdPhi = 1.0f - cosSqrdPhi;
    const float sinPhiM = (sinOrigPhi < 0.0f ? -scene_rdl2::math::sqrt(sinSqrdPhi) : scene_rdl2::math::sqrt(sinSqrdPhi));
    const float cosPhiM = (cosOrigPhi < 0.0f ? -scene_rdl2::math::sqrt(cosSqrdPhi) : scene_rdl2::math::sqrt(cosSqrdPhi));

    // Sample theta using eq (6) in [1], also simplifying using the identity above
    const float denominator = cosSqrdPhi * mInvAlphaUSqrd + sinSqrdPhi * mInvAlphaVSqrd;
    const float cosThetaMSqrd = denominator /
            (denominator - scene_rdl2::math::log(max(r1, sEpsilon)));
    const float cosThetaM = scene_rdl2::math::sqrt(cosThetaMSqrd);
    const float sinThetaM = scene_rdl2::math::sqrt(1.0f - cosThetaMSqrd);
    MNRY_ASSERT(cosThetaM > sEpsilon);

    // Compute the half vector
    Vec3f m = computeLocalSphericalDirection(cosThetaM, sinThetaM, cosPhiM, sinPhiM);
    m = mFrame.localToGlobal(m);

    // Compute reflection direction about the half vector
    computeReflectionDirection(m, slice.getWo(), wi);

    return eval(slice, wi, &pdf);
}

bool
AnisoCookTorranceBsdfLobe::getProperty(Property property, float *dest) const
{
    bool handled = true;

    switch (property)
    {
    case PROPERTY_ROUGHNESS:
        *dest       = mInputRoughness;
        *(dest + 1) = mInputRoughnessV;
        break;
    default:
        handled = CookTorranceBsdfLobe::getProperty(property, dest);
        break;
    }

    return handled;
}

//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

