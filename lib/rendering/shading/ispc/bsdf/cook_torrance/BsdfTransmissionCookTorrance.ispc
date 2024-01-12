// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfCookTorrance.ispc

#include "BsdfTransmissionCookTorrance.isph"
#include "energy_conservation/TransmissionCookTorranceEnergyCompensation.isph"
#include <moonray/rendering/shading/ispc/bsdf/BsdfSlice.isph>
#include <scene_rdl2/common/platform/IspcUtil.isph>


//-----------------------------------------------------------------------------

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
TransmissionCookTorranceBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                                      const varying BsdfSlice &slice,
                                      const Vec3f &wi,
                                      const varying float etaT,
                                      varying float * uniform pdf)
{
    const varying TransmissionCookTorranceBsdfLobe * uniform ct =
            (const varying TransmissionCookTorranceBsdfLobe * uniform) lobe;
    const Vec3f N = getN(ct->mFrame);

    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    float cosNO = dot(N, slice.mWo);
    float cosNI = dot(N, wi);

    // Exclude cases where wo and wi point to the same side of the surface
    if (cosNO * cosNI > 0.0f) {
        return sBlack;
    }

    // Compute abs of cosines and exclude degenerate cases
    cosNO = (cosNO > 0.0f  ?  min(cosNO, sOneMinusEpsilon)
                           :  max(cosNO, -sOneMinusEpsilon));
    cosNI = (cosNI > 0.0f  ?  min(cosNI, sOneMinusEpsilon)
                           :  max(cosNI, -sOneMinusEpsilon));
    const float absCosNO = abs(cosNO);
    if (absCosNO <= sEpsilon) return sBlack;
    const float absCosNI = abs(cosNI);
    if (absCosNI <= sEpsilon) return sBlack;

    // Compute microfacet / half vector, pointing towards the surface side
    // with the lowest ior
    Vec3f m;
    if (!computeNormalizedRefractionHalfVector(
            ct->mEtaI, slice.mWo, etaT, wi, m)) {
        return sBlack;
    }

    // Make sure microfacet points to the same side of the surface wrt. N and wo
    m = (ct->mEtaI > etaT  ?  neg(m)  :  m);

    // Exclude cases where microfacet m is pointing to the opposite side of
    // the surface (wrt. N). This corresponds to the X+(m.n) term of the normal
    // distribution function D(m); see eq. (25), (30) or (33)
    const float cosThetaM = dot(m, N);
    if (cosThetaM <= sEpsilon) return sBlack;

    // Note: computing cosMI and cosMO after potentially flipping m may change
    // their sign compared to the semantics in the paper, but this is necessary
    // for the next exclusion test below, and doesn't affect the outcome of
    // computing denom further below.
    const float cosMI = dot(m, wi);
    const float cosMO = dot(m, slice.mWo);

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
    float alpha2                = ct->mRoughness * ct->mRoughness;
    const float cosThetaM2      = cosThetaM * cosThetaM;
    const float cosThetaM4      = cosThetaM2 * cosThetaM2;
    const float minusTanThetaM2 = (cosThetaM2 - 1.0f) * rcp(cosThetaM2);
    float D                     = exp(minusTanThetaM2 * rcp(alpha2)) *
                                  rcp(sPi * alpha2 * cosThetaM4);

    // eq. 26, 27: now calculate G1(i,m) and G1(o,m)
    const float ao = absCosNO * ct->mInvRoughness * rsqrt(1.0f - cosNO * cosNO);
    const float ai = absCosNI * ct->mInvRoughness * rsqrt(1.0f - cosNI * cosNI);
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
    float denom = ct->mEtaI * cosMO + etaT * cosMI;
    denom = rcp(denom * denom);
    const float mIorTransmittedSqr = etaT * etaT;
    Color f = absCosMO * absCosMI * mIorTransmittedSqr * denom *
            BsdfLobe_computeScaleAndFresnel(lobe, absCosMO) * G * D * rcp(absCosNO) *
            (slice.mIncludeCosineTerm  ?  1.0f  :  rcp(absCosNI));

    f = f + TransmissionCookTorranceEnergyCompensation_evalT(
            absCosNO, absCosNI,
            ct->mInputRoughness,
            ct->mEtaI, etaT,
            ct->mFavg.r,
            ct->mFavgInv.r,
            slice.mIncludeCosineTerm);

    // Compute pdf of sampling wi by solving for pdf(wi) == po(o) in eq. (41),
    // by substituting fs(i,o,n) in eq. (41) with eq. (21).
    // pdf(wi) = etaT^2 * D(m) * |m.n| * |wi.m|
    //         / (etaI * (wo.m) + etaT * (wi.m))^2
    if (pdf != NULL) {
#if PBR_CT_SAMPLE_WIDENED_ROUGHNESS
        // We cannot re-use D computed above without adjusting for the widened
        // roughness we use in sample(), otherwise our pdf is biased
        const float alpha = widenRoughness(ct->mRoughness, absCosNO);
        alpha2 = alpha * alpha;
        D = exp(minusTanThetaM2 * rcp(alpha2)) *
                        rcp(sPi * alpha2 * cosThetaM4);
#endif
        *pdf = mIorTransmittedSqr * D * cosThetaM * absCosMI * denom;
    }

    return f * ct->mTint;
}

varying Color
TransmissionCookTorranceBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                                      const varying BsdfSlice &slice,
                                      const Vec3f &wi,
                                      varying float * uniform pdf)
{
    const varying TransmissionCookTorranceBsdfLobe * uniform ct =
            (const varying TransmissionCookTorranceBsdfLobe * uniform) lobe;

    return TransmissionCookTorranceBsdfLobe_eval(lobe, slice, wi,
                                                 ct->mEtaT, pdf);
}

varying Color
TransmissionCookTorranceBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                                        const varying BsdfSlice &slice,
                                        varying float r1, varying float r2,
                                        varying Vec3f &wi, varying float &pdf)
{
    const varying TransmissionCookTorranceBsdfLobe * uniform ct =
        (const varying TransmissionCookTorranceBsdfLobe * uniform) lobe;

    const float cosNO = dot(getN(ct->mFrame), slice.mWo);
    const float absCosNO = abs(cosNO);

    Color dispersionColor = sWhite;
    float neta = ct->mNeta;
    float etaT = ct->mEtaT;
    float spectralPDF = 1.0f;
    pdf = 0.0f;
    if (ct->mAllowDispersion) {
        sampleSpectralIOR(r1,
                          ct->mEtaR,
                          ct->mEtaG,
                          ct->mEtaB,
                          etaT,
                          spectralPDF,
                          dispersionColor);

        neta = ct->mEtaI / etaT;
    }

    // Compute a microfacet half-vector from the Beckmann distribution and
    // the transmission direction
    Vec3f m;
    computeBeckmannMicrofacet(lobe, slice, r1, r2, absCosNO, m);
    computeReflectionDirection(m, slice.mWo, wi);
    float cosMO, cosMI;
    if (!computeRefractionDirection(m,
                                    slice.mWo,
                                    neta,
                                    wi,
                                    cosMO, cosMI)) {
        wi = Vec3f_ctor(0.0f);
        pdf = 0.0f;
        return Color_ctor(0.0f, 0.0f, 0.0f);
    }

    Color result = TransmissionCookTorranceBsdfLobe_eval(lobe, slice, wi,
                                                         etaT, &pdf);

    // Dispersion PDF
    pdf *= spectralPDF;
    return (dispersionColor * result);
}


varying Color
TransmissionCookTorranceBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                                        const varying BsdfSlice &slice)
{
    const varying TransmissionCookTorranceBsdfLobe * uniform ct =
        (const varying TransmissionCookTorranceBsdfLobe * uniform) lobe;

    float cosThetaWo = max(dot(getN(ct->mFrame), slice.mWo), 0.0f);
    // TODO: Improve this approximation!
    return BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo);
}


void
TransmissionCookTorranceBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                               const varying Vec3f &wo,
                                               const varying Vec3f &wi,
                                               varying float r1, varying float r2,
                                               const varying Vec3f &dNdx,
                                               const varying Vec3f &dNdy,
                                               varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    const varying TransmissionCookTorranceBsdfLobe * uniform ct =
        (const varying TransmissionCookTorranceBsdfLobe * uniform) lobe;

    // It's complex to capture the full derivative. Instead we use the
    // derivative of a mirror refraction about the H vector, and scale the
    // length of the directional derivative proportionally with roughness.
    Vec3f H;
    if (computeNormalizedRefractionHalfVector(ct->mEtaI, wo,
                                              ct->mEtaT, wi, H)) {
        // Make sure microfacet points to the same side of the surface wrt. N and wo
        H = (ct->mEtaI > ct->mEtaT  ?  neg(H)  :  H);
    } else {
        H = getN(ct->mFrame);
    }
    computeRefractionDirectionDifferential(ct->mNeta, wo, wi, H, dNdx, dNdy,
                                           dDdx, dDdy);
    dDdx = dDdx * ct->mdDFactor;
    dDdy = dDdy * ct->mdDFactor;

}


uniform bool
TransmissionCookTorranceBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                             uniform BsdfLobeProperty property,
                                             varying float * uniform dest)
{
    const varying TransmissionCookTorranceBsdfLobe * uniform ct =
        (const varying TransmissionCookTorranceBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = ct->mInputRoughness;
        *(dest + 1) = ct->mInputRoughness;
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

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(TransmissionCookTorranceBsdfLobe);

