// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfWard.ispc

#include "BsdfWard.isph"
#include "BsdfWardCorrected.isph"
#include "BsdfWardDuer.isph"
#include "BsdfWardOriginal.isph"
#include <moonray/rendering/shading/ispc/bsdf/BsdfSlice.isph>
#include <moonray/rendering/shading/ispc/Util.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>

//-----------------------------------------------------------------------------

//
// WardBsdfLobe
//

void
WardBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                           const varying Vec3f &wo,
                           const varying Vec3f &wi,
                           varying float r1, varying float r2,
                           const varying Vec3f &dNdx,
                           const varying Vec3f &dNdy,
                           varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    const varying WardBsdfLobe * uniform ward =
        (const varying WardBsdfLobe * uniform) lobe;
    
    // It's complex to capture the full derivative. Instead we use the
    // derivative of a mirror reflection about the H vector, and scale the
    // length of the directional derivative proportionally with roughness.
    Vec3f H;
    if (!computeNormalizedHalfVector(wo, wi, H)) {
        H = getN(ward->mFrame);
    }
    computeReflectionDirectionDifferential(wo, wi, H, dNdx, dNdy, dDdx, dDdy);
    dDdx = dDdx * ward->mdDFactor;
    dDdy = dDdy * ward->mdDFactor;
}

varying Color
WardBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                    const varying BsdfSlice &slice)
{
    const varying WardBsdfLobe * uniform ward =
        (const varying WardBsdfLobe * uniform) lobe;
    
    float cosThetaWo = max(dot(getN(ward->mFrame), slice.mWo), 0.0f);
    return BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo);
}

uniform bool
WardBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                         uniform BsdfLobeProperty property,
                         varying float * uniform dest)
{
    const varying WardBsdfLobe * uniform ward =
        (const varying WardBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = ward->mInputRoughnessU;
        *(dest + 1) = ward->mInputRoughnessV;
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        {
            const Vec3f N = getN(ward->mFrame);
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
// WardCorrectedBsdfLobe
//

varying Color
WardCorrectedBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                           const varying BsdfSlice &slice,
                           const Vec3f &wi,
                           varying float * uniform pdf)
{
    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }
    
    const varying WardCorrectedBsdfLobe * uniform ward =
        (const varying WardCorrectedBsdfLobe * uniform) lobe;
    const Vec3f N = getN(ward->mFrame);
    const varying Color black = Color_ctor(0.f);
    float cosGrazingAngle = cos((90.0 - 0.1) / 180.0 * sPi);

    // We don't need to normalize H when it only appears at equal power both
    // on the nominator and denominator of expFactor
    Vec3f H;
    if (pdf != NULL  ||  BsdfLobe_getFresnel(lobe) != NULL) {
        if (!computeNormalizedHalfVector(slice.mWo, wi, H)) {
            return black;
        }
    } else {
        H = slice.mWo + wi;
    }

    float NdotWo = dot(N, slice.mWo);
    float NdotWi = dot(N, wi);
    if (NdotWo < cosGrazingAngle  ||  NdotWi < cosGrazingAngle) {
        return black;
    }

    // TODO: careful, H may not be normalized here! But, if NdotO and NdotI
    // are above the threshold, so should normalized HdotN
    float HdotN = dot(H, N);
    if (HdotN < cosGrazingAngle) {
        return black;
    }

    float HdotX = dot(H, ward->mFrame.mX);
    float HdotY = dot(H, ward->mFrame.mY);

    // Careful: HdotX = cosPhi * sinTheta and HdotY = sinPhi * sinTheta
    // which means the expFactor term already includes a sinThetaSqrd
    float expFactor = -(HdotX * HdotX * ward->mInvRoughUSqrd +
                        HdotY * HdotY * ward->mInvRoughVSqrd);
    float HdotNSqrd = HdotN * HdotN;
    expFactor /= HdotNSqrd;
    expFactor = ward->mScaleFactor * exp(expFactor);

    float product = NdotWo * NdotWi;
    float result = expFactor / sqrt(product);
    MNRY_ASSERT(result >= 0.0f);

    // Apply scale and fresnel if any
    float HdotWi = 1.0f;
    if (BsdfLobe_getFresnel(lobe) != NULL) {
        HdotWi = dot(H, wi);
    }
    Color f = result * BsdfLobe_computeScaleAndFresnel(lobe, HdotWi) *
            (slice.mIncludeCosineTerm  ?  max(NdotWi, 0.0f)  :  1.0f);

    // Compute pdf
    if (pdf != NULL) {
        float HdotWo = dot(H, slice.mWo);
        if (HdotWo < cosGrazingAngle) {
            return black;
        }

        *pdf = expFactor / (HdotWo * HdotNSqrd * HdotN);
        MNRY_ASSERT(*pdf >= 0.0f);
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = BsdfSlice_computeShadowTerminatorFix(&slice, N, wi);

    return Gs * f;
}

varying Color
WardCorrectedBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                             const varying BsdfSlice &slice,
                             varying float r1, varying float r2,
                             varying Vec3f &wi, varying float &pdf)
{
    const varying WardCorrectedBsdfLobe * uniform ward =
        (const varying WardCorrectedBsdfLobe * uniform) lobe;
    const Vec3f N = getN(ward->mFrame);
    const varying Color black = Color_ctor(0.f);
    float cosGrazingAngle = cos((90.0 - 0.1) / 180.0 * sPi);
    float cosGrazingAngleSample = cos((90.0 - 0.11) / 180.0 * sPi);
    
    // Sample phi and compute sin and cos
    // This is equivalent (but faster) to using eq. (7)
    float phi = sTwoPi * r2;
    float cosOrigPhi;
    float sinOrigPhi;
    sincos(phi, &sinOrigPhi, &cosOrigPhi);
    cosOrigPhi *= ward->mRoughnessU;
    sinOrigPhi *= ward->mRoughnessV;
    float cosSqrdPhi = 1.0f
            / (sinOrigPhi * sinOrigPhi / (cosOrigPhi * cosOrigPhi) + 1.0f);
    float sinSqrdPhi = 1.0f - cosSqrdPhi;
    float sinPhi = (sinOrigPhi < 0.0f ? -sqrt(sinSqrdPhi) : sqrt(sinSqrdPhi));
    float cosPhi = (cosOrigPhi < 0.0f ? -sqrt(cosSqrdPhi) : sqrt(cosSqrdPhi));

    // Sample theta
    float roughnessTerm = cosSqrdPhi * ward->mInvRoughUSqrd + sinSqrdPhi * ward->mInvRoughVSqrd;
    r1 = max(r1, sEpsilon);
    float cosThetaSqrd = roughnessTerm / (roughnessTerm - log(r1));
    float cosTheta = sqrt(cosThetaSqrd);
    float sinTheta = sqrt(1.0f - cosThetaSqrd);
    MNRY_ASSERT(cosTheta > sEpsilon);

    // Map the canonical half vector into the local space
    Vec3f H = Vec3f_ctor(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    MNRY_ASSERT(isNormalized(H));
    H = localToGlobal(ward->mFrame, H);

    // Compute reflection direction about the half vector
    float HdotWo = computeReflectionDirection(H, slice.mWo, wi);

    // If the sampled microfacet is facing away from wo, we return a
    // zero probability sample
    // TODO: Should this be using: HdotWo = abs(HdotWo);
    if (HdotWo < cosGrazingAngleSample) {
        pdf = 0.0f;
        return black;
    }


    // Compute the bsdf contribution
    float NdotWo = dot(N, slice.mWo);
    float NdotWi = dot(N, wi);
    if (NdotWo < cosGrazingAngle  ||  NdotWi < cosGrazingAngle) {
        pdf = 0.0f;
        return black;
    }

    float expFactor = ward->mScaleFactor * r1;
    float product = NdotWo * NdotWi;
    float result = expFactor / sqrt(product);
    MNRY_ASSERT(result >= 0.0f);

    // Apply scale and fresnel if any
    float HdotWi = HdotWo;
    Color f = result * BsdfLobe_computeScaleAndFresnel(lobe, HdotWi) *
            (slice.mIncludeCosineTerm  ?  max(NdotWi, 0.0f)  :  1.0f);


    // Compute the probability of selecting wi
    pdf = expFactor / (HdotWo * cosThetaSqrd * cosTheta);
    MNRY_ASSERT(pdf > 0.0f);

    return f;
}

//-----------------------------------------------------------------------------

//
// WardDuerBsdfLobe
//

inline varying Color
WardDuerBsdfLobe_evalInternal(const varying BsdfLobe * uniform lobe,
                              const varying BsdfSlice &slice,
                              const Vec3f &wi,
                              const Vec3f &H,
                              varying float * uniform pdf)
{
    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }
    
    varying WardDuerBsdfLobe * uniform ward =
        (varying WardDuerBsdfLobe * uniform) lobe;
    const Vec3f N = getN(ward->mFrame);
    const varying Color black = Color_ctor(0.f);
    float cosGrazingAngle = cos((90.0 - 0.1) / 180.0 * sPi);

    float NdotWo = dot(N, slice.mWo);
    float NdotWi = dot(N, wi);
    if (NdotWo < cosGrazingAngle  ||  NdotWi < cosGrazingAngle) {
        return black;
    }

    float HdotN = dot(H, N);
    if (HdotN < cosGrazingAngle) {
        return black;
    }

    float HdotX = dot(H, ward->mFrame.mX);
    float HdotY = dot(H, ward->mFrame.mY);

    // Careful: HdotX = cosPhi * sinTheta and HdotY = sinPhi * sinTheta
    // which means the expFactor term already includes a sinThetaSqrd
    float expFactor = -(HdotX * HdotX * ward->mInvRoughUSqrd +
                        HdotY * HdotY * ward->mInvRoughVSqrd);
    float HdotNSqrd = HdotN * HdotN;
    expFactor /= HdotNSqrd;
    expFactor = ward->mScaleFactor * exp(expFactor);

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
    Vec3f Hp = slice.mWo + wi;
    float HpdotN = dot(Hp, N);
    float HpdotHp = dot(Hp, Hp);
    float product = HpdotHp / (HpdotN * HpdotN * HpdotN * HpdotN);

    // Compute the brdf
    float result = 4.0f * expFactor * product  *
              (slice.mIncludeCosineTerm  ?  max(NdotWi, 0.0f)  :  1.0f);
    MNRY_ASSERT(result >= 0.0f);

    float HdotWi = dot(H, wi);
    Color f = result * BsdfLobe_computeScaleAndFresnel(lobe, HdotWi);

    // Compute pdf.
    // We use "Notes on the Ward BRDF", Tech report 2005, by Bruce Walter:
    // - equations (6) and (7) for the sampling,
    // - equation  (9) for the sampling pdf
    if (pdf != NULL) {
        // TODO: Optimization:
        // - Compute pdf in terms of the bsdf
        float HdotWo = dot(H, slice.mWo);
        if (HdotWo < cosGrazingAngle) {
            return black;
        }

        *pdf = expFactor / (HdotWo * HdotNSqrd * HdotN);
        assert(*pdf >= 0.0f);
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = BsdfSlice_computeShadowTerminatorFix(&slice, N, wi);

    return Gs * f;
}

varying Color
WardDuerBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                      const varying BsdfSlice &slice,
                      const Vec3f &wi,
                      varying float * uniform pdf)
{
    // Compute normalized half-vector
    Vec3f H;
    if (!computeNormalizedHalfVector(slice.mWo, wi, H)) {
        return Color_ctor(0.0f);
    }

    return WardDuerBsdfLobe_evalInternal(lobe, slice, wi, H, pdf);
}

varying Color
WardDuerBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                        const varying BsdfSlice &slice,
                        varying float r1, varying float r2,
                        varying Vec3f &wi, varying float &pdf)
{
    const varying WardDuerBsdfLobe * uniform ward =
        (const varying WardDuerBsdfLobe * uniform) lobe;
    float cosGrazingAngleSample = cos((90.0 - 0.11) / 180.0 * sPi);
    
    // Sample phi and compute sin and cos
    // This is equivalent (but faster) to using eq. (7)
    float phi = sTwoPi * r2;
    float cosOrigPhi;
    float sinOrigPhi;
    sincos(phi, &sinOrigPhi, &cosOrigPhi);
    cosOrigPhi *= ward->mRoughnessU;
    sinOrigPhi *= ward->mRoughnessV;
    float cosSqrdPhi = 1.0f
            / (sinOrigPhi * sinOrigPhi / (cosOrigPhi * cosOrigPhi) + 1.0f);
    float sinSqrdPhi = 1.0f - cosSqrdPhi;
    float sinPhi = (sinOrigPhi < 0.0f ? -sqrt(sinSqrdPhi) : sqrt(sinSqrdPhi));
    float cosPhi = (cosOrigPhi < 0.0f ? -sqrt(cosSqrdPhi) : sqrt(cosSqrdPhi));

    // Sample theta
    float roughnessTerm = cosSqrdPhi * ward->mInvRoughUSqrd + sinSqrdPhi * ward->mInvRoughVSqrd;
    r1 = max(r1, sEpsilon);
    float cosThetaSqrd = roughnessTerm / (roughnessTerm - log(r1));
    float cosTheta = sqrt(cosThetaSqrd);
    float sinTheta = sqrt(1.0f - cosThetaSqrd);
    MNRY_ASSERT(cosTheta > sEpsilon);

    // Map the canonical half vector into the local space
    Vec3f H = Vec3f_ctor(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    MNRY_ASSERT(isNormalized(H));
    H = localToGlobal(ward->mFrame, H);

    // Compute reflection direction about the half vector
    float HdotWo = computeReflectionDirection(H, slice.mWo, wi);

    // If the sampled microfacet is facing away from wo, we return a
    // zero probability sample
    // TODO: Should this be using: HdotWo = abs(HdotWo);
    if (HdotWo < cosGrazingAngleSample) {
        pdf = 0.0f;
        return Color_ctor(0.0f);
    }

    // TODO: Optimizations:
    // - Compute pdf in terms of the bsdf
    // - Compute them both re-using values computed thus far
    return WardDuerBsdfLobe_eval(lobe, slice, wi, &pdf);
}

//-----------------------------------------------------------------------------

//
// WardOriginalBsdfLobe
//

inline float
WardOriginalBsdfLobe_pdf(const varying BsdfLobe * uniform lobe,
                         const varying Vec3f &wo, 
                         const varying Vec3f &wi)
{
    const varying WardOriginalBsdfLobe * uniform ward =
        (const varying WardOriginalBsdfLobe * uniform) lobe;
    float cosGrazingAngle = cos((90.0 - 0.1) / 180.0 * sPi);
    float minTheta = 0.1 / 180.0 * sPi;
    float minSinTheta = sin(minTheta);
    float maxCosTheta = cos(minTheta);
    
    // Compute normalized half-vector
    Vec3f H;
    if (!computeNormalizedHalfVector(wo, wi, H)) {
        return 0.0f;
    }

    float cosTheta = dot(H, getN(ward->mFrame));
    if (cosTheta < cosGrazingAngle) {
        return 0.0f;
    }

    float HdotWo = dot(H, wo);
    if (HdotWo < cosGrazingAngle) {
        return 0.0f;
    }

    cosTheta = min(cosTheta, maxCosTheta);
    float theta = dw_acos(cosTheta);
    float sinThetaSqrd = 1.0f - cosTheta * cosTheta;
    // TODO: is sqrt(sinThetaSqrd) faster than sin(theta) ?
    float sinTheta = sqrt(sinThetaSqrd);

    float HdotX = dot(H, ward->mFrame.mX);
    float HdotY = dot(H, ward->mFrame.mY);

    // Careful: HdotX = cosPhi * sinTheta and HdotY = sinPhi * sinTheta
    // which means the expFactor term includes a sinThetaSqrd, so here we cancel
    // the sinThetaSqrd out to get to the equation described in the header.
    float expFactor = HdotX * HdotX * ward->mInvRoughUSqrd +
                      HdotY * HdotY * ward->mInvRoughVSqrd;
    expFactor *= -theta * theta / sinThetaSqrd;

    float proba = theta * ward->mScaleFactor / (HdotWo * sinTheta) * exp(expFactor);
    MNRY_ASSERT(proba >= 0.0f);

    return proba;
}

varying Color
WardOriginalBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                          const varying BsdfSlice &slice,
                          const Vec3f &wi,
                          varying float * uniform pdf)
{
    const varying WardOriginalBsdfLobe * uniform ward =
        (const varying WardOriginalBsdfLobe * uniform) lobe;
    
    // eval() == pdf()
    float result = WardOriginalBsdfLobe_pdf(lobe, slice.mWo, wi);
    if (pdf != NULL) {
        *pdf = result;
    }

    // Apply scale and fresnel if any
    float HdotWi = 1.0f;
    if (BsdfLobe_getFresnel(lobe) != NULL) {
        Vec3f H;
        if (computeNormalizedHalfVector(slice.mWo, wi, H)) {
            HdotWi = dot(H, wi);
        }
    }
    Color f = result * BsdfLobe_computeScaleAndFresnel(lobe, HdotWi) *
            (slice.mIncludeCosineTerm  ?
             max(dot(getN(ward->mFrame), wi), 0.0f)  :  1.0f);

    // Soften hard shadow terminator due to shading normals
    const float Gs = BsdfSlice_computeShadowTerminatorFix(&slice, getN(ward->mFrame), wi);

    return Gs * f;
}

varying Color
WardOriginalBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice,
                            varying float r1, varying float r2,
                            varying Vec3f &wi, varying float &pdf)
{
    const varying WardOriginalBsdfLobe * uniform ward =
        (const varying WardOriginalBsdfLobe * uniform) lobe;
    float cosGrazingAngleSample = cos((90.0 - 0.11) / 180.0 * sPi);
    float minTheta = 0.1 / 180.0 * sPi;
    float minSinTheta = sin(minTheta);
    float maxCosTheta = cos(minTheta);
    
    // Sample phi and compute sin and cos
    // TODO: Why all this complexity to keep track of the quadrant ?
    // TODO: Should the anisotropy modify how phi is sampled at all ?
    float phi = sTwoPi * r2;
    float cosOrigPhi;
    float sinOrigPhi;
    sincos(phi, &sinOrigPhi, &cosOrigPhi);
    cosOrigPhi *= ward->mRoughnessU;
    sinOrigPhi *= ward->mRoughnessV;
    float cosSqrdPhi = 1.0f
            / (sinOrigPhi * sinOrigPhi / (cosOrigPhi * cosOrigPhi) + 1.0f);
    float sinSqrdPhi = 1.0f - cosSqrdPhi;
    float sinPhi = (sinOrigPhi < 0.0f ? -sqrt(sinSqrdPhi) : sqrt(sinSqrdPhi));
    float cosPhi = (cosOrigPhi < 0.0f ? -sqrt(cosSqrdPhi) : sqrt(cosSqrdPhi));

    // Sample theta
    float roughnessTerm = cosSqrdPhi * ward->mInvRoughUSqrd + sinSqrdPhi * ward->mInvRoughVSqrd;
    r1 = max(r1, sEpsilon);
    float theta = sqrt(-log(r1) / roughnessTerm);

    // Because we use the un-corrected sampling equations of the original
    // Ward paper, the resulting theta can be bigger than pi/2, which doesn't
    // make any sense in the microfacet theory. In this case we return a zero
    // probability sample
    if (theta >= sHalfPi) {
        pdf = 0.0f;
        return Color_ctor(0.0f);
    }

    float sinTheta;
    float cosTheta;
    sincos(theta, &sinTheta, &cosTheta);

    // Map the canonical half vector into the local space
    Vec3f H = Vec3f_ctor(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    MNRY_ASSERT(isNormalized(H));
    H = localToGlobal(ward->mFrame, H);

    // Compute reflection direction about the half vector
    float HdotWo = computeReflectionDirection(H, slice.mWo, wi);

    // If the sampled microfacet is facing away from wo, we return a
    // zero probability sample
    // TODO: Should this be using: HdotWo = abs(HdotWo);
    if (HdotWo < cosGrazingAngleSample) {
        pdf = 0.0f;
        return Color_ctor(0.0f);
    }


    // Compute the probability of selecting wi
    theta = max(theta, minTheta);
    sinTheta = max(sinTheta, minSinTheta);
    float result = theta * ward->mScaleFactor / (HdotWo * sinTheta) * r1;
    MNRY_ASSERT(result > 0.0f);

    // Apply scale and fresnel if any
    float HdotWi = HdotWo;
    Color f = result * BsdfLobe_computeScaleAndFresnel(lobe, HdotWi) *
            (slice.mIncludeCosineTerm  ?
             max(dot(getN(ward->mFrame), wi), 0.0f)  :  1.0f);

    pdf = result;

    return f;
}

//-----------------------------------------------------------------------------

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(WardBsdfLobe);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(WardCorrectedBsdfLobe);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(WardDuerBsdfLobe);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(WardOriginalBsdfLobe);

