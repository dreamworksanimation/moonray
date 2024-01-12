// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfCookTorrance.ispc

#include "BsdfCookTorrance.isph"
#include "BsdfAnisoCookTorrance.isph"
#include "BsdfBerryCookTorrance.isph"
#include "BsdfGGXCookTorrance.isph"
#include "BsdfGlitterGGXCookTorrance.isph"
#include "energy_conservation/CookTorranceEnergyCompensation.isph"
#include "energy_conservation/TransmissionCookTorranceEnergyCompensation.isph"
#include <moonray/rendering/shading/ispc/Util.isph>
#include <moonray/rendering/shading/ispc/bsdf/BsdfUtil.isph>
#include <moonray/rendering/shading/ispc/bsdf/BsdfSlice.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>


//-----------------------------------------------------------------------------

float
CookTorranceBsdfLobe_energyCompensationWeight(
        const varying CookTorranceBsdfLobe * uniform ct)
{

    float w = 0.0f;
    if (isBlack(ct->mFavg)) return w;
    if (ct->mCoupledWithTransmission) {
        w = TransmissionCookTorranceEnergyCompensation_weightR(
                ct->mEtaI, ct->mEtaT,
                ct->mInputRoughness);
    }
    else {
        w = CookTorranceEnergyCompensation_weight(
                ct->mInputRoughness);
    }
    return w;
}

float
CookTorranceBsdfLobe_energyCompensationPDF(
        const varying CookTorranceBsdfLobe * uniform ct,
        MicrofacetDistribution type,
        float cosNI)
{
    float pdf = 0.0f;
    if (isBlack(ct->mFavg)) return pdf;
    if (ct->mCoupledWithTransmission) {
        pdf = TransmissionCookTorranceEnergyCompensation_pdfR(
                cosNI,
                ct->mEtaI, ct->mEtaT,
                ct->mInputRoughness);
    } else {
        pdf = CookTorranceEnergyCompensation_pdf(
                type,
                cosNI,
                ct->mInputRoughness);
    }
    return pdf;
}

void
CookTorranceBsdfLobe_sampleEnergyCompensation(
        const varying CookTorranceBsdfLobe * uniform ct,
        MicrofacetDistribution type,
        const Vec3f& wo,
        float r1, float r2,
        Vec3f& wi)
{
    if (isBlack(ct->mFavg)) return;
    if (ct->mCoupledWithTransmission) {
        TransmissionCookTorranceEnergyCompensation_sampleR(
                wo,
                ct->mFrame,
                r1, r2,
                wi,
                ct->mInputRoughness,
                ct->mEtaI, ct->mEtaT);
    } else {
        CookTorranceEnergyCompensation_sample(
                type,
                ct->mFrame,
                r1, r2,
                ct->mInputRoughness,
                wi);
    }
}

Color
CookTorranceBsdfLobe_evalEnergyCompensation(
        const varying CookTorranceBsdfLobe * uniform ct,
        MicrofacetDistribution type,
        float cosNO, float cosNI,
        bool includeCosineTerm)
{
    Color c = sBlack;
    if (isBlack(ct->mFavg)) return c;

    if (ct->mCoupledWithTransmission) {
        c = TransmissionCookTorranceEnergyCompensation_evalR(
                cosNO, cosNI,
                ct->mInputRoughness,
                ct->mEtaI, ct->mEtaT,
                ct->mFavg.r,
                ct->mFavgInv.r,
                includeCosineTerm);
    } else {
        c = CookTorranceEnergyCompensation_eval(
                type,
                cosNO, cosNI,
                ct->mInputRoughness,
                ct->mFavg,
                includeCosineTerm);
    }
    return c;
}

static void
computeBeckmannMicrofacet(const varying BsdfLobe * uniform lobe,
                          const varying BsdfSlice &slice,
                          const varying float r1, const varying float r2,
                          const varying float cosNO, varying Vec3f &m)
{
    const varying CookTorranceBsdfLobe * uniform ct =
        (const varying CookTorranceBsdfLobe * uniform) lobe;

    // generate a random microfacet normal m (eq. 28, 29)
    // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
    // and sin(atan(x)) == x/sqrt(1+x^2)
    const float alpha = widenRoughness(ct->mRoughness, cosNO);
    const float alpha2 = alpha * alpha;
    const float tanThetaMSqr = -alpha2 * log(1.0f - r1);
    const float tanThetaM = sqrt(tanThetaMSqr);
    const float cosThetaM = rsqrt(1.0f + tanThetaMSqr);
    const float sinThetaM = cosThetaM * tanThetaM;
    const float phiM = 2.0f * sPi * r2;

    // Compute the half vector
    m = computeLocalSphericalDirection(cosThetaM, sinThetaM, phiM);
    m = localToGlobal(ct->mFrame, m);
}

//-----------------------------------------------------------------------------

varying Color
CookTorranceBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                          const varying BsdfSlice &slice,
                          const Vec3f &wi,
                          varying float * uniform pdf)
{
    const varying CookTorranceBsdfLobe * uniform ct =
            (const varying CookTorranceBsdfLobe * uniform) lobe;
    const Vec3f N = getN(ct->mFrame);
    const varying Color black = Color_ctor(0.f);

    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    float cosNO = dot(N, slice.mWo);
    if (cosNO <= sEpsilon) return black;
    cosNO = min(cosNO, sOneMinusEpsilon);

    float cosNI = dot(N, wi);
    if (cosNI <= sEpsilon) return black;
    cosNI = min(cosNI, sOneMinusEpsilon);

    // get half vector
    Vec3f m;
    if (!computeNormalizedHalfVector(slice.mWo, wi, m)) {
        return black;
    }

    const float cosMI       = dot(m, wi);
    if (cosMI <= sEpsilon) return black;

    const float cosThetaM   = dot(m, N);
    if (cosThetaM <= sEpsilon) return black;

    // eq. 25: calculate D(m):
    float alpha2                = ct->mRoughness * ct->mRoughness;
    const float cosThetaM2      = cosThetaM * cosThetaM;
    const float cosThetaM4      = cosThetaM2 * cosThetaM2;
    const float minusTanThetaM2 = (cosThetaM2 - 1.0f) * rcp(cosThetaM2);
    float D4                    = 0.25f * exp(minusTanThetaM2 * rcp(alpha2)) *
                                  rcp(sPi * alpha2 * cosThetaM4);

    // eq. 26, 27: now calculate G1(i,m) and G1(o,m)
    const float ao = cosNO * ct->mInvRoughness * rsqrt(1.0f - cosNO * cosNO);
    const float ai = cosNI * ct->mInvRoughness * rsqrt(1.0f - cosNI * cosNI);
    const float G1o = (ao < 1.6f) ? (3.535f * ao + 2.181f * ao * ao) *
            rcp(1.0f + 2.276f * ao + 2.577f * ao * ao) : 1.0f;
    const float G1i = (ai < 1.6f) ? (3.535f * ai + 2.181f * ai * ai) *
            rcp(1.0f + 2.276f * ai + 2.577f * ai * ai) : 1.0f;
    const float G  = G1o * G1i;

    // Compute the Cook-Torrance bsdf
    Color f = BsdfLobe_computeScaleAndFresnel(lobe, cosMI) * G * D4 * rcp(cosNO) *
              (slice.mIncludeCosineTerm  ?  1.0f  :  rcp(cosNI));

    const float w2 = CookTorranceBsdfLobe_energyCompensationWeight(ct);
    if (w2 > 0.0f) {
        const Color compen = CookTorranceBsdfLobe_evalEnergyCompensation(
                                 ct,
                                 MICROFACET_DISTRIBUTION_BECKMANN,
                                 cosNO, cosNI,
                                 slice.mIncludeCosineTerm);
        f = f + compen;
    }

    // Compute pdf of sampling wi using eq. (17) in [1]
    // pdf(wi) = D(m) * |m.n| / (4 * |o.m|)
    if (pdf != NULL) {
#if PBR_CT_SAMPLE_WIDENED_ROUGHNESS
        // We cannot re-use D4 computed above without adjusting for the widened
        // roughness we use in sample(), otherwise our pdf is biased
        const float alpha = widenRoughness(ct->mRoughness, cosNO);
        alpha2 = alpha * alpha;
        D4 = 0.25f * exp(minusTanThetaM2 * rcp(alpha2)) *
                rcp(sPi * alpha2 * cosThetaM4);
#endif
        const float p2 = CookTorranceBsdfLobe_energyCompensationPDF(
                             ct,
                             MICROFACET_DISTRIBUTION_BECKMANN,
                             cosNI);
        const float w1 = (1.0f - w2);
        const float p1 = D4 * cosThetaM * rcp(cosMI);
        // One Sample PDF Weight
        *pdf = (w1 * p1 + w2 * p2);
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = BsdfSlice_computeShadowTerminatorFix(&slice, N, wi);

    return Gs * f;
}


varying Color
CookTorranceBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice,
                            varying float r1, varying float r2,
                            varying Vec3f &wi, varying float &pdf)
{
    const varying CookTorranceBsdfLobe * uniform ct =
        (const varying CookTorranceBsdfLobe * uniform) lobe;

    const float cosNO = dot(getN(ct->mFrame), slice.mWo);
    if (cosNO <= 0.0f) {
        wi = Vec3f_ctor(0.0f);
        pdf = 0.0f;
        return Color_ctor(0.0f);
    }

    // One Sample between energy compensation and regular
    // microfacet distribution sampling
    const float w2 = CookTorranceBsdfLobe_energyCompensationWeight(ct);
    if (r1 < w2) {
        r1 = r1 / w2;
        CookTorranceBsdfLobe_sampleEnergyCompensation(
            ct,
            MICROFACET_DISTRIBUTION_BECKMANN,
            slice.mWo, r1, r2, wi);
    } else {
        const float w1 = (1.0f - w2);
        r1 = (r1 - w2) / w1;
        // Compute a microfacet half-vector from the Beckmann distribution and
        // the reflection about the half-vector
        Vec3f m;
        computeBeckmannMicrofacet(lobe, slice, r1, r2, cosNO, m);
        computeReflectionDirection(m, slice.mWo, wi);
    }

    return CookTorranceBsdfLobe_eval(lobe, slice, wi, &pdf);
}


varying Color
CookTorranceBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice)
{
    const varying CookTorranceBsdfLobe * uniform ct =
        (const varying CookTorranceBsdfLobe * uniform) lobe;

    float cosThetaWo = max(dot(getN(ct->mFrame), slice.mWo), 0.0f);
    // TODO: Improve this approximation!
    return BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo);
}


void
CookTorranceBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                   const varying Vec3f &wo,
                                   const varying Vec3f &wi,
                                   varying float r1, varying float r2,
                                   const varying Vec3f &dNdx,
                                   const varying Vec3f &dNdy,
                                   varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    const varying CookTorranceBsdfLobe * uniform ct =
        (const varying CookTorranceBsdfLobe * uniform) lobe;

    // It's complex to capture the full derivative. Instead we use the
    // derivative of a mirror reflection about the H vector, and scale the
    // length of the directional derivative proportionally with roughness.
    Vec3f H;
    if (!computeNormalizedHalfVector(wo, wi, H)) {
        H = getN(ct->mFrame);
    }
    computeReflectionDirectionDifferential(wo, wi, H, dNdx, dNdy, dDdx, dDdy);
    dDdx = dDdx * ct->mdDFactor;
    dDdy = dDdy * ct->mdDFactor;
}


uniform bool
CookTorranceBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                 uniform BsdfLobeProperty property,
                                 varying float * uniform dest)
{
    const varying CookTorranceBsdfLobe * uniform ct =
        (const varying CookTorranceBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = ct->mInputRoughness;
        *(dest + 1) = ct->mInputRoughness;
        break;
    case BSDF_LOBE_PROPERTY_PBR_VALIDITY:
        // Forward call to fresnel
        {
            varying Color res = sBlack;
            if (isActive(lobe->mFresnel)) {
                res = Fresnel_computePbrValidity(lobe->mFresnel);
            }
            *dest       = res.r;
            *(dest + 1) = res.g;
            *(dest + 2) = res.b;
        }
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        {
            const Vec3f N = getN(ct->mFrame);
            *dest       = N.x;
            *(dest + 1) = N.y;
            *(dest + 2) = N.z;
        }
        break;
    default:
        handled = false;
        break;
    }

    return handled;
}

//-----------------------------------------------------------------------------
//
// AnisoCookTorranceBsdfLobe
//
/// @brief evaluate an anisotropic cook torrance lobe
inline varying Color
AnisoCookTorranceBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                               const varying BsdfSlice &slice,
                               const Vec3f &wi,
                               varying float * uniform pdf)
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

    const varying AnisoCookTorranceBsdfLobe * uniform act =
        (const varying AnisoCookTorranceBsdfLobe * uniform) lobe;
    const Vec3f N = getN(act->mFrame);
    const varying Color black = Color_ctor(0.f);

    const float cosNO = dot(N, slice.mWo);
    if (cosNO <= 0.0f) return black;

    const float cosNI = dot(N, wi);
    if (cosNI <= 0.0f) return black;

    // get half vector
    Vec3f m;
    if (!computeNormalizedHalfVector(slice.mWo, wi, m)) {
       return black;
    }

    const float cosMI       = dot(m, wi);
    if (cosMI <= 0.0f)      return black;

    const float cosThetaM   = dot(m, N);
    if (cosThetaM <= 0.0f)  return black;

    // Calculate D(m) (with m=Hr), using eq. (82) in [3]. Note that we use the
    // (exact and cheaper) vector form of the exponential term from eq. (4) in [1].
    const float dotMX = dot(m, act->mFrame.mX);
    const float dotMY = dot(m, act->mFrame.mY);
    float cosThetaMX = dotMX / act->mRoughness;
    float cosThetaMX2 = cosThetaMX * cosThetaMX;
    float cosThetaMY = dotMY / act->mRoughnessV;
    float cosThetaMY2 = cosThetaMY * cosThetaMY;
    const float cosThetaM2  = cosThetaM * cosThetaM;
    float D4 = 0.25f * exp(-(cosThetaMX2 + cosThetaMY2) / cosThetaM2)
        / (sPi * act->mRoughness * act->mRoughnessV * cosThetaM2 * cosThetaM2);

    // Calculate G1(i,m) and G1(o,m) using equations (83) and (80) in [3].
    // First we simplify the expression for a, also using the exact vector form
    // (similarly to how [1] derives the exponential term in eq. (4)):
    // a = (v.n) / sqrt(((v.x)*alphax)^2 + ((v.y)*alphay)^2)
    const float tmpXO = dot(act->mFrame.mX, slice.mWo) * act->mRoughness;
    const float tmpYO = dot(act->mFrame.mY, slice.mWo) * act->mRoughnessV;
    const float ao = cosNO / sqrt(tmpXO * tmpXO + tmpYO * tmpYO);
    const float tmpXI = dot(act->mFrame.mX, wi) * act->mRoughness;
    const float tmpYI = dot(act->mFrame.mY, wi) * act->mRoughnessV;
    const float ai = cosNI / sqrt(tmpXI * tmpXI + tmpYI * tmpYI);
    // Use the rational approximation of G1: eq. (27) in [2]
    const float G1o = (ao < 1.6f) ? (3.535f * ao + 2.181f * ao * ao) /
                                   (1.0f + 2.276f * ao + 2.577f * ao * ao) : 1.0f;
    const float G1i = (ai < 1.6f) ? (3.535f * ai + 2.181f * ai * ai) /
                                   (1.0f + 2.276f * ai + 2.577f * ai * ai) : 1.0f;
    const float G   = G1o * G1i;

    const float invCosNO = 1.0f / cosNO;
    const float invCosNI = 1.0f / cosNI;
    // Compute the Cook-Torrance bsdf
    const Color f = BsdfLobe_computeScaleAndFresnel(lobe, cosMI) * G * D4 * invCosNO *
           (slice.mIncludeCosineTerm  ?  1.0f  :  invCosNI);

    // Compute pdf of sampling wi using eq. (17) in [1]
    // pdf(wi) = D(m) * |m.n| / (4 * |o.m|)
    if (pdf != NULL) {
    #if PBR_CT_SAMPLE_WIDENED_ROUGHNESS
       // We cannot re-use D4 computed above without adjusting for the widened
       // roughness we use in sample(), otherwise our pdf is biased
       const float alphaU = widenRoughness(act->mRoughness, cosNO);
       const float alphaV = widenRoughness(act->mRoughnessV, cosNO);
       cosThetaMX = dotMX / alphaU;
       cosThetaMX2 = cosThetaMX * cosThetaMX;
       cosThetaMY = dotMY / alphaV;
       cosThetaMY2 = cosThetaMY * cosThetaMY;
       D4 = 0.25f * exp(-(cosThetaMX2 + cosThetaMY2) / cosThetaM2)
           / (sPi * alphaU * alphaV * cosThetaM2 * cosThetaM2);
    #endif
       *pdf = D4 * cosThetaM / cosMI;
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = BsdfSlice_computeShadowTerminatorFix(&slice, N, wi);

    return Gs * f;
}

/// @brief sample an anisotropic cook torrance lobe
inline varying Color
AnisoCookTorranceBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                                 const varying BsdfSlice &slice,
                                 varying float r1, varying float r2,
                                 varying Vec3f &wi, varying float &pdf)
{
    const varying AnisoCookTorranceBsdfLobe * uniform act =
        (const varying AnisoCookTorranceBsdfLobe * uniform) lobe;

    const float cosNO = dot(getN(act->mFrame), slice.mWo);
    if (cosNO <= 0.0f) {
        wi = Vec3f_ctor(0.0f);
        pdf = 0.0f;
        return Color_ctor(0.0f);
    }

    // We use the anisotropic sampling equations (6) and (7) in [1], which apply
    // directly here, since they are in fact sampling the anisotropic
    // Beckmann distribution we use here (which the Ward model is also based on).

    // For sampling, use the modified sampling distribution in sect 5.3 in [2]
    // Reduce maximum weight with widened distribution
    const float alphaU = widenRoughness(act->mRoughness, cosNO);
    const float alphaV = widenRoughness(act->mRoughnessV, cosNO);
    const float mInvAlphaUSqrd = 1.0f / (alphaU * alphaU);
    const float mInvAlphaVSqrd = 1.0f / (alphaV * alphaV);

    // Sample phi and compute sin and cos
    // This is equivalent (but faster) to using eq. (7) in [1].
    // Using the identity: cos(x)^2 = 1 / (1 + tan(x)^2)
    const float phi = sTwoPi * r2;
    const float cosOrigPhi = alphaU * cos(phi);
    const float sinOrigPhi = alphaV * sin(phi);
    const float cosSqrdPhi = 1.0f
            / (sinOrigPhi * sinOrigPhi / (cosOrigPhi * cosOrigPhi) + 1.0f);
    const float sinSqrdPhi = 1.0f - cosSqrdPhi;
    const float sinPhiM = (sinOrigPhi < 0.0f ? -sqrt(sinSqrdPhi) : sqrt(sinSqrdPhi));
    const float cosPhiM = (cosOrigPhi < 0.0f ? -sqrt(cosSqrdPhi) : sqrt(cosSqrdPhi));

    // Sample theta using eq (6) in [1], also simplifying using the identity above
    const float denominator = cosSqrdPhi * mInvAlphaUSqrd + sinSqrdPhi * mInvAlphaVSqrd;
    const float cosThetaMSqrd = denominator /
            (denominator - log(max(r1, sEpsilon)));
    const float cosThetaM = sqrt(cosThetaMSqrd);
    const float sinThetaM = sqrt(1.0f - cosThetaMSqrd);
    MNRY_ASSERT(cosThetaM > sEpsilon);

    // Compute the half vector
    Vec3f m = computeLocalSphericalDirection(cosThetaM, sinThetaM, cosPhiM, sinPhiM);
    m = localToGlobal(act->mFrame, m);

    // Compute reflection direction about the half vector
    computeReflectionDirection(m, slice.mWo, wi);

    return AnisoCookTorranceBsdfLobe_eval(lobe, slice, wi, &pdf);
}

inline uniform bool
AnisoCookTorranceBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                      uniform BsdfLobeProperty property,
                                      varying float * uniform dest)
{
    const varying AnisoCookTorranceBsdfLobe * uniform act =
        (const varying AnisoCookTorranceBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = act->mInputRoughness;
        *(dest + 1) = act->mInputRoughnessV;
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        {
            const Vec3f N = getN(act->mFrame);
            *dest       = N.x;
            *(dest + 1) = N.y;
            *(dest + 2) = N.z;
        }
        break;
    default:
        handled = false;
        break;
    }

    return handled;
}

//-----------------------------------------------------------------------------

//
// BerryCookTorranceBsdfLobe
//

/// @brief evaluate an berry cook torrance lobe
varying Color
BerryCookTorranceBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                               const varying BsdfSlice &slice,
                               const Vec3f &wi,
                               varying float * uniform pdf)
{
    // Based on the paper:
    //      "Physically-Based Shading at Disney"
    //      (Burley et al. - Siggraph Course 2012)

    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    const varying BerryCookTorranceBsdfLobe * uniform bct =
        (const varying BerryCookTorranceBsdfLobe * uniform) lobe;
    const Vec3f N = getN(bct->mFrame);
    const varying Color black = Color_ctor(0.f);

    const float cosNO = dot(N, slice.mWo);
    if (cosNO <= 0.0f)      return black;

    const float cosNI = dot(N, wi);
    if (cosNI <= 0.0f)      return black;

    // get half vector
    Vec3f m;
    if (!computeNormalizedHalfVector(slice.mWo, wi, m)) {
        return black;
    }

    const float cosMI       = dot(m, wi);
    if (cosMI <= 0.0f)      return black;

    const float cosThetaM   = dot(m, N);
    if (cosThetaM <= 0.0f)  return black;

    // eq. "D_Berry": calculate D(m) (with m=Hr):
    float alpha2            = bct->mRoughness * bct->mRoughness;
    const float cosThetaM2  = cosThetaM * cosThetaM;
    const float sinThetaM2  = 1.0f - cosThetaM2;
    float d0                = (alpha2 - 1.0f) / (sPi * log(alpha2));
    float d1                = 1.0f / (alpha2 * cosThetaM2 + sinThetaM2);
    float D4                = 0.25f * d0 * d1;

    // Per section 5.6 "Specular G details", use GGX shadowing and masking functions
    // However, we don't seem to use the remapping: alpha2 = (0.5 + mRoughness / 2)^2
    const float invCosNO = 1.0f / cosNO;
    const float invCosNI = 1.0f / cosNI;
    const float G1o = 2.0f / (1.0f + sqrt(1.0f + alpha2 *
                        (1.0f - cosNO * cosNO) * (invCosNO * invCosNO)));
    const float G1i = 2.0f / (1.0f + sqrt(1.0f + alpha2 *
                        (1.0f - cosNI * cosNI) * (invCosNI * invCosNI)));
    const float G = G1o * G1i;

    // Compute the Cook-Torrance bsdf
    const Color f = BsdfLobe_computeScaleAndFresnel(lobe, cosMI) * G * D4 * invCosNO *
            (slice.mIncludeCosineTerm  ?  1.0f  :  invCosNI);

    // Compute pdf of sampling wi using eq. (17) in [1]
    // pdf(wi) = D(m) * |m.n| / (4 * |o.m|)
    if (pdf != NULL) {
#if PBR_CT_SAMPLE_WIDENED_ROUGHNESS
        // We cannot re-use D4 computed above without adjusting for the widened
        // roughness we use in sample(), otherwise our pdf is biased
        const float alpha = widenRoughness(bct->mRoughness, cosNO);
        alpha2            = alpha * alpha;
        d0                = (alpha2 - 1.0f) / (sPi * log(alpha2));
        d1                = 1.0f / (alpha2 * cosThetaM2 + sinThetaM2);
        D4                = 0.25f * d0 * d1;
#endif
        *pdf = D4 * cosThetaM / cosMI;
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = BsdfSlice_computeShadowTerminatorFix(&slice, N, wi);

    return Gs * f;
}

/// @brief sample an berry cook torrance lobe
varying Color
BerryCookTorranceBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                                 const varying BsdfSlice &slice,
                                 varying float r1, varying float r2,
                                 varying Vec3f &wi, varying float &pdf)
{
    const varying BerryCookTorranceBsdfLobe * uniform bct =
        (const varying BerryCookTorranceBsdfLobe * uniform) lobe;

    const float cosNO = dot(getN(bct->mFrame), slice.mWo);
    if (cosNO <= 0.0f) {
        wi = Vec3f_ctor(0.0f);
        pdf = 0.0f;
        return Color_ctor(0.0f);
    }

    // generate a random microfacet normal m (eq. 35,36):
    // using: tan(acos(x)) == sqrt(1-x^2) / x
    //        sin(x) == cos(x) * tan(x)
    const float alpha = widenRoughness(bct->mRoughness, cosNO);
    const float alpha2 = alpha * alpha;
    const float cosThetaM = sqrt((1.0f - pow(alpha2, (1.0f - r1))) / (1.0f - alpha2));
    const float tanThetaM = sqrt(1.0f - cosThetaM * cosThetaM) / cosThetaM;
    const float sinThetaM = cosThetaM * tanThetaM;
    const float phiM = 2.0f * sPi * r2;

    Vec3f m = computeLocalSphericalDirection(cosThetaM, sinThetaM, phiM);
    // TODO: check and remove: m.normalize();
    m = localToGlobal(bct->mFrame, m);

    // Compute reflection direction about the half vector
    computeReflectionDirection(m, slice.mWo, wi);

    return BerryCookTorranceBsdfLobe_eval(lobe, slice, wi, &pdf);
}

//-----------------------------------------------------------------------------

//
// GGXCookTorranceBsdfLobe
//

/// The below two functions allow sharing of common code between GGXCookTorrance and GlitterGGXCookTorrance
/// It is a specialization required to specify the normal (frame) to be used for the bsdf sampling/evaluation
varying Color
GGXCookTorranceBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                             const varying BsdfSlice &slice,
                             const Vec3f &wi,
                             varying float * uniform pdf,
                             const varying float roughness,
                             const varying float cosNO,
                             const varying float cosNI,
                             const varying ReferenceFrame& frame)
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
    if (!computeNormalizedHalfVector(slice.mWo, wi, m)) {
        return sBlack;
    }

    const float cosMI       = dot(m, wi);
    if (cosMI <= 0.0f)      return sBlack;

    const float cosThetaM   = dot(m, getN(frame));
    if (cosThetaM <= 0.0f)  return sBlack;

    // eq. 33: calculate D(m):
    float alpha2 = roughness * roughness;
    const float cosThetaM2 = cosThetaM * cosThetaM;
    const float piCosThetaM4 = sPi * cosThetaM2 * cosThetaM2;
    const float tanThetaM2 = (1.0f - cosThetaM2) / cosThetaM2;
    float tmp = alpha2 + tanThetaM2;
    float D4 = 0.25f * alpha2 / (piCosThetaM4 * tmp * tmp);

    // eq. 34: now calculate G1(i,m) and G1(o,m)
    const float invCosNO = 1.0f / cosNO;
    const float invCosNI = 1.0f / cosNI;
    const float G1o = 2.0f / (1.0f + sqrt(1.0f + alpha2 *
                        (1.0f - cosNO * cosNO) * (invCosNO * invCosNO)));
    const float G1i = 2.0f / (1.0f + sqrt(1.0f + alpha2 *
                        (1.0f - cosNI * cosNI) * (invCosNI * invCosNI)));
    const float G = G1o * G1i;

    // Compute the Cook-Torrance bsdf
    Color f = BsdfLobe_computeScaleAndFresnel(lobe, cosMI) * G * D4 * invCosNO *
            (slice.mIncludeCosineTerm  ?  1.0f  :  invCosNI);

    const varying CookTorranceBsdfLobe * uniform ct =
        (const varying CookTorranceBsdfLobe * uniform) lobe;

    const float w2 = CookTorranceBsdfLobe_energyCompensationWeight(ct);
    if (w2 > 0.0f) {
        const Color compen = CookTorranceBsdfLobe_evalEnergyCompensation(
                                 ct,
                                 MICROFACET_DISTRIBUTION_GGX,
                                 cosNO, cosNI,
                                 slice.mIncludeCosineTerm);
        f = f + compen;
    }

    // Compute pdf of sampling wi using eq. (17) in [1]
    // pdf(wi) = D(m) * |m.n| / (4 * |o.m|)
    if (pdf != NULL) {
#if PBR_CT_SAMPLE_WIDENED_ROUGHNESS
        // We cannot re-use D4 computed above without adjusting for the widened
        // roughness we use in sample(), otherwise our pdf is biased
        const float alpha = widenRoughness(roughness, cosNO);
        alpha2 = alpha * alpha;
        tmp = alpha2 + tanThetaM2;
        D4 = 0.25f * alpha2 / (piCosThetaM4 * tmp * tmp);
#endif
        const float p2 = CookTorranceBsdfLobe_energyCompensationPDF(
                             ct,
                             MICROFACET_DISTRIBUTION_GGX,
                             cosNI);
        const float w1 = (1.0f - w2);
        const float p1 = D4 * cosThetaM / cosMI;
        // One Sample PDF Weight
        *pdf = (w1 * p1 + w2 * p2);
    }

    return f;
}

varying Vec3f
GGXCookTorranceBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                               const varying BsdfSlice &slice,
                               varying float r1, varying float r2,
                               const varying float roughness,
                               const varying float cosNO,
                               const varying ReferenceFrame& frame)
{
    // generate a random microfacet normal m (eq. 35,36):
    // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
    // and sin(atan(x)) == x/sqrt(1+x^2)
    const float alpha = widenRoughness(roughness, cosNO);
    const float alpha2 = alpha * alpha;
    const float tanThetaM2  = alpha2 * r1 / (1.0f - r1);
    const float cosThetaM   = 1.0f / sqrt(1.0f + tanThetaM2);
    const float sinThetaM   = cosThetaM * sqrt(tanThetaM2);
    const float phiM = 2.0f * sPi * r2;

    // Compute the half vector
    Vec3f m = computeLocalSphericalDirection(cosThetaM, sinThetaM, phiM);
    m = localToGlobal(frame, m);

    // Compute reflection direction about the half vector
    Vec3f wi;
    computeReflectionDirection(m, slice.mWo, wi);
    return wi;
}

varying Color
GGXCookTorranceBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                             const varying BsdfSlice &slice,
                             const Vec3f &wi,
                             varying float * uniform pdf)
{
    const varying GGXCookTorranceBsdfLobe * uniform ggx =
        (const varying GGXCookTorranceBsdfLobe * uniform) lobe;
    const Vec3f N = getN(ggx->mFrame);

    const float cosNO = dot(N, slice.mWo);
    const float cosNI = dot(N, wi);

    // Soften hard shadow terminator due to shading normals
    // The code is here to avoid applying the softening in the case of glitter
    const float Gs = BsdfSlice_computeShadowTerminatorFix(&slice, N, wi);

    return Gs * GGXCookTorranceBsdfLobe_eval(lobe, slice, wi, pdf, ggx->mRoughness, cosNO, cosNI, ggx->mFrame);
}

varying Color
GGXCookTorranceBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                               const varying BsdfSlice &slice,
                               varying float r1, varying float r2,
                               varying Vec3f &wi, varying float &pdf)
{
    const varying GGXCookTorranceBsdfLobe * uniform ggx =
        (const varying GGXCookTorranceBsdfLobe * uniform) lobe;

    const float cosNO = dot(getN(ggx->mFrame), slice.mWo);
    if (cosNO <= 0.0f) {
        wi = Vec3f_ctor(0.0f);
        pdf = 0.0f;
        return sBlack;
    }

    const varying CookTorranceBsdfLobe * uniform ct =
        (const varying CookTorranceBsdfLobe * uniform) lobe;

    // One Sample between energy compensation and regular
    // microfacet distribution sampling
    const float w2 = CookTorranceBsdfLobe_energyCompensationWeight(ct);
    if (r1 < w2) {
        r1 = r1 / w2;
        CookTorranceBsdfLobe_sampleEnergyCompensation(
                ct,
                MICROFACET_DISTRIBUTION_GGX,
                slice.mWo, r1, r2, wi);
    } else {
        const float w1 = (1.0f - w2);
        r1 = (r1 - w2) / w1;

        wi = GGXCookTorranceBsdfLobe_sample(lobe, slice, r1, r2,
                                            ggx->mRoughness, cosNO, ggx->mFrame);
    }

    return GGXCookTorranceBsdfLobe_eval(lobe, slice, wi, &pdf);
}



//-----------------------------------------------------------------------------

//
// GlitterGGXCookTorranceBsdfLobe
//

inline uniform bool
GlitterGGXCookTorranceBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                           uniform BsdfLobeProperty property,
                                           varying float * uniform dest)
{
    const varying GlitterGGXCookTorranceBsdfLobe * uniform ct =
        (const varying GlitterGGXCookTorranceBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_NORMAL:
        {
            const Vec3f& N = ct->mFlakeNormal;
            *dest       = N.x;
            *(dest + 1) = N.y;
            *(dest + 2) = N.z;
        }
        break;
    default:
        handled = CookTorranceBsdfLobe_getProperty(lobe,
                                                   property,
                                                   dest);

        break;
    }

    return handled;
}

varying Color
GlitterGGXCookTorranceBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                                    const varying BsdfSlice &slice,
                                    const Vec3f &wi,
                                    varying float * uniform pdf)
{
    const varying GlitterGGXCookTorranceBsdfLobe * uniform glitter =
        (const varying GlitterGGXCookTorranceBsdfLobe * uniform) lobe;

    // Try evaluation with the flake normal
    float cosNO = dot(glitter->mFlakeNormal, slice.mWo);
    float cosNI = dot(glitter->mFlakeNormal, wi);

    varying ReferenceFrame frame;
    // If flake normal is too perturbed, evaluate with shading normal
    if (cosNO <= 0.0f || cosNI <= 0.0f) {
        frame = glitter->mFrame;
        cosNO = dot(getN(frame), slice.mWo);
        cosNI = dot(getN(frame), wi);
    } else {
        ReferenceFrame_init(frame, glitter->mFlakeNormal);
    }

    return GGXCookTorranceBsdfLobe_eval(lobe, slice, wi, pdf, glitter->mRoughness, cosNO, cosNI, frame);
}

varying Color
GlitterGGXCookTorranceBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                                      const varying BsdfSlice &slice,
                                      varying float r1, varying float r2,
                                      varying Vec3f &wi, varying float &pdf)
{
    const varying GlitterGGXCookTorranceBsdfLobe * uniform glitter =
        (const varying GlitterGGXCookTorranceBsdfLobe * uniform) lobe;

    // Try sampling with the flake normal
    varying bool isUsingFlakeNormal = true;
    float cosNO = dot(glitter->mFlakeNormal, slice.mWo);

    varying ReferenceFrame frame;
    // If wo is outside the hemisphere of the flake normal, try sampling with shading normal
    if (cosNO <= 0.0f) {
        frame = glitter->mFrame;
        cosNO = dot(getN(frame), slice.mWo);
        isUsingFlakeNormal = false;
        // Unsuccessful even with shading normal
        if (cosNO <= 0.0f) {
            wi = Vec3f_ctor(0.0f);
            pdf = 0.0f;
            return sBlack;
        }
    } else {
        ReferenceFrame_init(frame, glitter->mFlakeNormal);
    }

    wi = GGXCookTorranceBsdfLobe_sample(lobe, slice, r1, r2, glitter->mRoughness, cosNO, frame);

    varying float cosNI = dot(getN(frame), wi);
    // If wi is created with flake normal, check if it is within the hemisphere of shading normal
    // If not, generate another wi using the shading normal instead
    if (isUsingFlakeNormal) {
        const varying float cosNIShading = dot(getN(glitter->mFrame), wi);
        if (cosNIShading <= 0.0f) {
            frame = glitter->mFrame;
            cosNO = dot(getN(frame), slice.mWo);
            // Unsuccessful even with shading normal
            if (cosNO <= 0.0f) {
                wi = Vec3f_ctor(0.0f);
                pdf = 0.0f;
                return sBlack;
            }
            wi = GGXCookTorranceBsdfLobe_sample(lobe, slice, r1, r2, glitter->mRoughness, cosNO, frame);
            cosNI = dot(getN(frame), wi);
        }
    }

    return GGXCookTorranceBsdfLobe_eval(lobe, slice, wi, &pdf, glitter->mRoughness, cosNO, cosNI, frame);
}

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(CookTorranceBsdfLobe);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(AnisoCookTorranceBsdfLobe);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(BerryCookTorranceBsdfLobe);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(GGXCookTorranceBsdfLobe);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(GlitterGGXCookTorranceBsdfLobe);

