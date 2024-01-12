// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfAshikhminShirley.ispc

#include "BsdfAshikhminShirleyDiffuse.isph"
#include "BsdfAshikhminShirleyGlossy.isph"
#include <moonray/rendering/shading/ispc/bsdf/BsdfSlice.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>

//----------------------------------------------------------------------------

//
// AshikhminShirleyDiffuseBsdfLobe
//

varying Color
AshikhminShirleyDiffuseBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                          const varying BsdfSlice &slice,
                          const Vec3f &wi,
                          varying float * uniform pdf)
{
    const varying AshikhminShirleyDiffuseBsdfLobe * uniform ash =
        (const varying AshikhminShirleyDiffuseBsdfLobe * uniform) lobe;
    const Vec3f N = getN(ash->mFrame);

    float tmp, tmp2;

    float NdotWo = max(dot(N, slice.mWo), 0.0f);
    tmp = 1.0f - 0.5f * NdotWo;
    tmp2 = tmp * tmp;
    float powerWo = tmp2 * tmp2 * tmp;

    float NdotWi = max(dot(N, wi), 0.0f);
    tmp = 1.0f - 0.5f * NdotWi;
    tmp2 = tmp * tmp;
    float powerWi = tmp2 * tmp2 * tmp;

    float result = AshikhminShirley_sDiffuseConstant * (1.0f - powerWo) * (1.0f - powerWi);

    Color f = result * BsdfLobe_computeScaleAndFresnel(lobe, NdotWi) *
            (slice.mIncludeCosineTerm  ?  NdotWi  :  1.0f);

    if (pdf != NULL) {
        *pdf = NdotWi * sOneOverPi;
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = BsdfSlice_computeShadowTerminatorFix(&slice, N, wi);

    return Gs * f;
}

varying Color
AshikhminShirleyDiffuseBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice,
                            varying float r1, varying float r2,
                            varying Vec3f &wi, varying float &pdf)
{
    const varying AshikhminShirleyDiffuseBsdfLobe * uniform ash =
        (const varying AshikhminShirleyDiffuseBsdfLobe * uniform) lobe;

    wi = localToGlobal(ash->mFrame, sampleLocalHemisphereCosine(r1, r2));
    return AshikhminShirleyDiffuseBsdfLobe_eval(lobe, slice, wi, &pdf);
}

varying Color
AshikhminShirleyDiffuseBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice)
{
    const varying AshikhminShirleyDiffuseBsdfLobe * uniform ash =
        (const varying AshikhminShirleyDiffuseBsdfLobe * uniform) lobe;

    float cosThetaWo = max(dot(getN(ash->mFrame), slice.mWo), 0.0f);
    return BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo);
}

void
AshikhminShirleyDiffuseBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                   const varying Vec3f &wo,
                                   const varying Vec3f &wi,
                                   varying float r1, varying float r2,
                                   const varying Vec3f &dNdx,
                                   const varying Vec3f &dNdy,
                                   varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    const varying AshikhminShirleyDiffuseBsdfLobe * uniform ash =
        (const varying AshikhminShirleyDiffuseBsdfLobe * uniform) lobe;

    // See BsdfLambert.h for details
    localHemisphereCosineDifferentials(r1, r2, dDdx, dDdy);
    squarifyRectangle(dDdx, dDdy);

    dDdx = localToGlobal(ash->mFrame, dDdx);
    dDdy = localToGlobal(ash->mFrame, dDdy);
}

uniform bool
AshikhminShirleyDiffuseBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                            uniform BsdfLobeProperty property,
                                            varying float * uniform dest)
{
    const varying AshikhminShirleyDiffuseBsdfLobe * uniform ash =
        (const varying AshikhminShirleyDiffuseBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = 1.0f;
        *(dest + 1) = 1.0f;
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        {
            const Vec3f N = getN(ash->mFrame);
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

//----------------------------------------------------------------------------

//
// AshikhminShirleyGlossyBsdfLobe
//

inline float
samplePhi(const varying float r2, const varying float scale)
{
    float phi = atan(scale * tan(sHalfPi * r2));

    return phi;
}

varying Color
AshikhminShirleyGlossyBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                          const varying BsdfSlice &slice,
                          const Vec3f &wi,
                          varying float * uniform pdf)
{
    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    const varying AshikhminShirleyGlossyBsdfLobe * uniform ash =
        (const varying AshikhminShirleyGlossyBsdfLobe * uniform) lobe;
    const Vec3f N = getN(ash->mFrame);
    const Color black = Color_ctor(0.f);
    float cosGrazingAngle = cos((90.0 - 0.1) / 180.0 * sPi);
    float sinNormalAngleSqrd = 1e-7;

    // Compute normalized half-vector
    Vec3f H;
    if (!computeNormalizedHalfVector(slice.mWo, wi, H)) {
        return black;
    }

    float NdotWi = dot(N, wi);
    float maxNdotW = max(dot(N, slice.mWo), NdotWi);
    if (maxNdotW < cosGrazingAngle) {
        return black;
    }

    float HdotN = dot(H, N);
    if (HdotN < cosGrazingAngle) {
        return black;
    }

    float sinThetaSqrd = 1.0f - HdotN * HdotN;
    if (sinThetaSqrd < sinNormalAngleSqrd) {
        return black;
    }

    float HdotWi = dot(H, wi);
    HdotWi = max(HdotWi, sEpsilon);

    float HdotX = dot(H, ash->mFrame.mX);
    float HdotY = dot(H, ash->mFrame.mY);

    float exponent = (ash->mExponentU * HdotX * HdotX +
                      ash->mExponentV * HdotY * HdotY) / sinThetaSqrd;
    float common = ash->mScaleFactor * pow(HdotN, exponent);


    // Compute bsdf contribution
    float result = common / (HdotWi * maxNdotW);

    Color f = result * BsdfLobe_computeScaleAndFresnel(lobe, HdotWi) *
            (slice.mIncludeCosineTerm  ?  max(NdotWi, 0.0f)  :  1.0f);


    // Compute pdf
    if (pdf != NULL) {
        float HdotWo = HdotWi;
        *pdf = common / HdotWo;
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = BsdfSlice_computeShadowTerminatorFix(&slice, N, wi);

    return Gs * f;
}

varying Color
AshikhminShirleyGlossyBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice,
                            varying float r1, varying float r2,
                            varying Vec3f &wi, varying float &pdf)
{
    const varying AshikhminShirleyGlossyBsdfLobe * uniform ash =
        (const varying AshikhminShirleyGlossyBsdfLobe * uniform) lobe;
    const Vec3f N = getN(ash->mFrame);
    const Color black = Color_ctor(0.f);
    float cosGrazingAngleSample = cos((90.0 - 0.11) / 180.0 * sPi);
    float sinNormalAngleSqrd = 1e-7;
    float cosNormalAngleSample = sqrt(1.0 - (sinNormalAngleSqrd * 2.0));

    // Sample phi according to equation (9)
    float phi;
    if (r2 < 0.25f) {
        r2 = 4.0f * r2;
        phi = samplePhi(r2, ash->mSampleFactor);
    } else if (r2 < 0.5f) {
        r2 = 4.0f * (0.5f - r2);
        phi = sPi - samplePhi(r2, ash->mSampleFactor);
    } else if (r2 < 0.75f) {
        r2 = 1.0f - 4.0f * (0.75f - r2);
        phi = sPi + samplePhi(r2, ash->mSampleFactor);
    } else {
        r2 = 4.0f * (1.0f - r2);
        phi = sTwoPi - samplePhi(r2, ash->mSampleFactor);
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
    float exponent = ash->mExponentU * cosPhi * cosPhi +
                     ash->mExponentV * sinPhi * sinPhi;
    float cosTheta = pow(1.0f - r1, 1.0f / (exponent + 1.0f));
    if (cosTheta < cosGrazingAngleSample  ||  cosTheta > cosNormalAngleSample) {
        pdf = 0.0f;
        return black;
    }
    float cosThetaSqrd = cosTheta * cosTheta;
    float sinTheta = sqrt(1.0f - cosThetaSqrd);

    // Map the canonical half vector into the local space
    Vec3f H = Vec3f_ctor(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    assert(isNormalized(H));
    H = localToGlobal(ash->mFrame, H);

    // Compute reflection direction about the half vector
    float HdotWo = computeReflectionDirection(H, slice.mWo, wi);

    // If the sampled microfacet is facing away from wo, we return a
    // zero probability sample
    // TODO: Should this be using: HdotWo = abs(HdotWo);
    if (HdotWo < cosGrazingAngleSample) {
        pdf = 0.0f;
        return black;
    }

    float NdotWi = dot(N, wi);
    float maxNdotW = max(dot(N, slice.mWo), NdotWi);
    if (maxNdotW < cosGrazingAngleSample) {
        pdf = 0.0f;
        return black;
    }

    float common = ash->mScaleFactor * pow(cosTheta, exponent);

    // Compute the probability of selecting wi
    pdf = common / HdotWo;

    // Compute the brdf
    float result = pdf / maxNdotW;

    // Apply scale and fresnel if any
    float HdotWi = HdotWo;
    Color f = result * BsdfLobe_computeScaleAndFresnel(lobe, HdotWi) *
            (slice.mIncludeCosineTerm  ?  max(NdotWi, 0.0f)  :  1.0f);

    return f;
}

varying Color
AshikhminShirleyGlossyBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice)
{
    const varying AshikhminShirleyGlossyBsdfLobe * uniform ash =
        (const varying AshikhminShirleyGlossyBsdfLobe * uniform) lobe;

    float cosThetaWo = max(dot(getN(ash->mFrame), slice.mWo), 0.0f);
    return BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo);
}

void
AshikhminShirleyGlossyBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                   const varying Vec3f &wo,
                                   const varying Vec3f &wi,
                                   varying float r1, varying float r2,
                                   const varying Vec3f &dNdx,
                                   const varying Vec3f &dNdy,
                                   varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    const varying AshikhminShirleyGlossyBsdfLobe * uniform ash =
        (const varying AshikhminShirleyGlossyBsdfLobe * uniform) lobe;

    // It's complex to capture the full derivative. Instead we use the
    // derivative of a mirror reflection about the H vector, and scale the
    // length of the directional derivative proportionally with roughness.
    Vec3f H;
    if (!computeNormalizedHalfVector(wo, wi, H)) {
        H = getN(ash->mFrame);
    }
    computeReflectionDirectionDifferential(wo, wi, H, dNdx, dNdy, dDdx, dDdy);
    dDdx = dDdx * ash->mdDFactor;
    dDdy = dDdy * ash->mdDFactor;
}

uniform bool
AshikhminShirleyGlossyBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                           uniform BsdfLobeProperty property,
                                           varying float * uniform dest)
{
    const varying AshikhminShirleyGlossyBsdfLobe * uniform ash =
        (const varying AshikhminShirleyGlossyBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = ash->mInputRoughnessU;
        *(dest + 1) = ash->mInputRoughnessV;
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        {
            const Vec3f N = getN(ash->mFrame);
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

//----------------------------------------------------------------------------

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(AshikhminShirleyDiffuseBsdfLobe);
#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(AshikhminShirleyGlossyBsdfLobe);

