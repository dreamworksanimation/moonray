// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfCookTorrance.cc
/// $Id$
///


#include "BsdfCookTorrance.h"
#include "energy_compensation/TransmissionCookTorranceEnergyCompensation.h"
#include <moonray/rendering/shading/bsdf/Fresnel.h>
#include <scene_rdl2/common/math/MathUtil.h>
#include <moonray/rendering/shading/Util.h>
#include <algorithm>
#include <iostream>


using namespace scene_rdl2::math;

namespace moonray {
namespace shading {


//----------------------------------------------------------------------------

TransmissionCookTorranceBsdfLobe::TransmissionCookTorranceBsdfLobe(const Vec3f &N,
                                                                   float roughness,
                                                                   float etaI,
                                                                   float etaT,
                                                                   const Color &tint,
                                                                   float favg,
                                                                   float favginv,
                                                                   float abbeNumber) :
            BsdfLobe(Type(TRANSMISSION | GLOSSY),
                     DifferentialFlags(0),
                     false,
                     PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
            mFrame(N),
            mTint(tint),
            mInputRoughness(roughness),
            mEtaI(etaI),
            mEtaT(etaT),
            mNeta(mEtaI * rcp(mEtaT))

{
    mFavg = favg;
    mFavgInv = favginv;
    // Apply roughness squaring to linearize roughness response
    // See "Physically-Based Shading at Disney" Siggraph course notes.
    mRoughness = mInputRoughness * mInputRoughness;

    if (mRoughness < 0.001f) {
        mRoughness = 0.001f;
    }

    mInvRoughness = rcp(mRoughness);

    if (scene_rdl2::math::isZero(abbeNumber)) {
        mAllowDispersion = false;
    } else {
        shading::computeSpectralIOR(etaT,
                                    abbeNumber,
                                    &mEtaR,
                                    &mEtaG,
                                    &mEtaB);
        mAllowDispersion = true;
    }

    // Use a directional differential scale that varies with roughness
    mdDFactor = sdDFactorMin + mRoughness * sdDFactorSlope;
}


Color
TransmissionCookTorranceBsdfLobe::eval(const BsdfSlice &slice,
                                       const Vec3f &wi,
                                       float *pdf) const
{
    // Based on the paper:
    //      "Microfacet Models for Refraction through Rough Surfaces"
    //      (Walter et al. - EGSR 2007)

    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }
    return eval(slice, wi, mEtaT, pdf);
}

Color
TransmissionCookTorranceBsdfLobe::eval(const BsdfSlice &slice,
                                       const Vec3f &wi,
                                       float etaT,
                                       float *pdf) const
{
    float cosNO = dot(mFrame.getN(), slice.getWo());
    float cosNI = dot(mFrame.getN(), wi);

    // Exclude cases where wo and wi point to the same side of the surface
    if (cosNO * cosNI > 0.0f) {
        return sBlack;
    }

    // Compute abs of cosines and exclude degenerate cases
    cosNO = (cosNO > 0.0f  ?  min(cosNO, sOneMinusEpsilon)
                           :  max(cosNO, -sOneMinusEpsilon));
    cosNI = (cosNI > 0.0f  ?  min(cosNI, sOneMinusEpsilon)
                           :  max(cosNI, -sOneMinusEpsilon));
    const float absCosNO = scene_rdl2::math::abs(cosNO);
    if (absCosNO <= sEpsilon) return sBlack;
    const float absCosNI = scene_rdl2::math::abs(cosNI);
    if (absCosNI <= sEpsilon) return sBlack;

    // Compute microfacet / half vector, pointing towards the surface side
    // with the lowest ior
    Vec3f m;
    if (!computeNormalizedRefractionHalfVector(mEtaI,
                                               slice.getWo(),
                                               etaT,
                                               wi,
                                               m)) {
        return sBlack;
    }

    // Make sure microfacet points to the same side of the surface wrt. N and wo
    m = (mEtaI > etaT  ?  -m  :  m);

    // Exclude cases where microfacet m is pointing to the opposite side of
    // the surface (wrt. N). This corresponds to the X+(m.n) term of the normal
    // distribution function D(m); see eq. (25), (30) or (33)
    const float cosThetaM = dot(m, mFrame.getN());
    if (cosThetaM <= sEpsilon) return sBlack;

    // Note: computing cosMI and cosMO after potentially flipping m may change
    // their sign compared to the semantics in the paper, but this is necessary
    // for the next exclusion test below, and doesn't affect the outcome of
    // computing denom further below.
    const float cosMI = dot(m, wi);
    const float cosMO = dot(m, slice.getWo());

    // Exclude cases where wo or wi is not on the same side as m and N.
    // This corresponds to the X+(v, m, n) term of the shadow-masking
    // function G1(v, m); see eq. (27) or (34)
    if (cosNO * cosMO <= sEpsilon  ||  cosNI * cosMI <= sEpsilon) {
        return sBlack;
    }

    // By construction above (we flipped m towards wo), we know that
    // cosMI < 0 and cosMO > 0, so we don't need to use abs()
    const float absCosMI = -cosMI;
    if (absCosMI <= sEpsilon) return sBlack;
    const float absCosMO = cosMO;
    if (absCosMO <= sEpsilon) return sBlack;


    // eq. 25: calculate D(m):
    const float alpha2          = mRoughness * mRoughness;
    const float cosThetaM2      = cosThetaM * cosThetaM;
    const float cosThetaM4      = cosThetaM2 * cosThetaM2;
    const float minusTanThetaM2 = (cosThetaM2 - 1.0f) * rcp(cosThetaM2);
    const float D               = scene_rdl2::math::exp(minusTanThetaM2 * rcp(alpha2)) *
                                  rcp(sPi * alpha2 * cosThetaM4);

    // eq. 26, 27: now calculate G1(i,m) and G1(o,m)
    const float ao = absCosNO * mInvRoughness * rsqrt(1.0f - cosNO * cosNO);
    const float ai = absCosNI * mInvRoughness * rsqrt(1.0f - cosNI * cosNI);
    const float G1o = (ao < 1.6f) ? (3.535f * ao + 2.181f * ao * ao) *
            rcp(1.0f + 2.276f * ao + 2.577f * ao * ao) : 1.0f;
    const float G1i = (ai < 1.6f) ? (3.535f * ai + 2.181f * ai * ai) *
            rcp(1.0f + 2.276f * ai + 2.577f * ai * ai) : 1.0f;
    const float G  = G1o * G1i;

    // Compute the Cook-Torrance bsdf, using equation (21)
    // Note: we assume this lobe has been setup with a OneMinus*Fresnel
    // as we want to use 1 - specular_fresnel. Also notice we use
    // cosThetaWo to evaluate the fresnel term, as an approximation of what
    // hDotWi would be for the specular lobe.
//    float denom = mEtaI * cosMO + etaT * cosMI;
    float denom = mEtaI * cosMO + etaT * cosMI;
    denom = rcp(denom * denom);
    const float etaTSqr = etaT * etaT;
    Color fresnel = computeScaleAndFresnel(absCosMO);
    Color f = absCosMO * absCosMI * etaTSqr * denom *
             fresnel * G * D * rcp(absCosNO) *
            (slice.getIncludeCosineTerm()  ?  1.0f  :  rcp(absCosNI));

    f += TransmissionCookTorranceEnergyCompensation::evalT(
            absCosNO, absCosNI,
            mInputRoughness,
            mEtaI, mEtaT,
            mFavg,
            mFavgInv,
            slice.getIncludeCosineTerm());

    // Compute pdf of sampling wi by solving for pdf(wi) == po(o) in eq. (41),
    // by substituting fs(i,o,n) in eq. (41) with eq. (21).
    // pdf(wi) = etaT^2 * D(m) * |m.n| * |wi.m|
    //         / (etaI * (wo.m) + etaT * (wi.m))^2
    if (pdf != NULL) {
#if PBR_CT_SAMPLE_WIDENED_ROUGHNESS
        // We cannot re-use D computed above without adjusting for the widened
        // roughness we use in sample(), otherwise our pdf is biased
        const float alpha = CookTorranceBsdfLobe::widenRoughness(mRoughness,
                                                                 absCosNO);
        const float alpha2 = alpha * alpha;
        const float D = scene_rdl2::math::exp(minusTanThetaM2 * rcp(alpha2)) *
                        rcp(sPi * alpha2 * cosThetaM4);
#endif
        *pdf = D * cosThetaM * etaTSqr * absCosMI * denom;
    }

    // apply transmission color tint hack
    return f * mTint;
}

Color
TransmissionCookTorranceBsdfLobe::sample(const BsdfSlice &slice,
                                         float r1, float r2,
                                         Vec3f &wi,
                                         float &pdf) const
{
    const float cosNO = dot(mFrame.getN(), slice.getWo());
    const float absCosNO = scene_rdl2::math::abs(cosNO);

    Color dispersionColor = scene_rdl2::math::sWhite;
    float neta = mNeta;
    float etaT = mEtaT;
    float spectralPDF = 1.0f;
    pdf = 0.0f;
    if (mAllowDispersion) {
        shading::sampleSpectralIOR(r1,
                                   mEtaR,
                                   mEtaG,
                                   mEtaB,
                                   &etaT,
                                   &spectralPDF,
                                   dispersionColor);

        neta = mEtaI / etaT;
    }


    Color val;
    // generate a random microfacet normal m (eq. 28, 29)
    // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
    // and sin(atan(x)) == x/sqrt(1+x^2)
    const float alpha = CookTorranceBsdfLobe::widenRoughness(mRoughness,
                                                             absCosNO);
    const float alpha2 = alpha * alpha;
    const float tanThetaMSqr = -alpha2 * scene_rdl2::math::log(1.0f - r1);
    const float tanThetaM = scene_rdl2::math::sqrt(tanThetaMSqr);
    const float cosThetaM = rsqrt(1.0f + tanThetaMSqr);
    const float sinThetaM = cosThetaM * tanThetaM;
    const float phiM = 2.0f * sPi * r2;

    // Compute the half vector. By construction it is facing the same side
    // as N and as wo
    Vec3f m = computeLocalSphericalDirection(cosThetaM, sinThetaM, phiM);
    m = mFrame.localToGlobal(m);

    // Compute the transmission direction
    float cosMO, cosMI;
    if (!computeRefractionDirection(m,
                                    slice.getWo(),
                                    neta,
                                    wi,
                                    cosMO, cosMI)) {
        wi = Vec3f(zero);
        pdf = 0.0f;
        return Color(0.0f, 0.0f, 0.0f);
    }

    Color result = eval(slice, wi, etaT, &pdf);
    // Dispersion PDF
    pdf *= spectralPDF;

    val = (dispersionColor * result);
    return val;
}

bool
TransmissionCookTorranceBsdfLobe::getProperty(Property property, float *dest) const
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
    default:
        handled = BsdfLobe::getProperty(property, dest);
        break;
    }

    return handled;
}

//----------------------------------------------------------------------------

} // namespace shading
} // namespace moonray

