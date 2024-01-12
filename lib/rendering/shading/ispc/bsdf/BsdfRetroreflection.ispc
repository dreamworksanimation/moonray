// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfRetroreflection.ispc

#include "BsdfRetroreflection.isph"
#include "BsdfSlice.isph"
#include <moonray/rendering/shading/ispc/Util.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>

varying Color
RetroreflectionBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                             const varying BsdfSlice &slice,
                             const Vec3f &wi,
                             varying float * uniform pdf)
{
    // Prepare for early exit
    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    const varying RetroreflectionBsdfLobe *uniform retroLobe =
        (const varying RetroreflectionBsdfLobe *uniform) lobe;
    const Vec3f N = retroLobe->mN;
    const varying Color black = Color_ctor(0.f);

    const float cosThetaI = min(dot(N, wi), sOneMinusEpsilon);
    if (cosThetaI <= sEpsilon) return black;

    const float cosOmegaO_OmegaI = min(dot(wi, slice.mWo), sOneMinusEpsilon);
    if (cosOmegaO_OmegaI <= sEpsilon) return black;

    //convert roughness to phong exponent (see Walter'07 [2])
    const float alphaP = 2.0f * rcp(retroLobe->mRoughness * retroLobe->mRoughness) - 2.0f;

    // section 3.1 in [1]
    const float power = pow(cosOmegaO_OmegaI, alphaP) * sOneOverTwoPi;
    const float normalizationFactor2 = (alphaP + 2.0f);
    const Color f = BsdfLobe_computeScaleAndFresnel(lobe, cosThetaI) *
            normalizationFactor2 * power *
            (slice.mIncludeCosineTerm ? cosThetaI : 1.0);

    // Compute pdf of sampling
    // section 3.3.1 in [1])
    if (pdf != NULL) {
        const float normalizationFactor1 = (alphaP + 1.0f);
        *pdf = normalizationFactor1 * power;
    }

    // Soften hard shadow terminator due to shading normals
    const float Gs = BsdfSlice_computeShadowTerminatorFix(&slice, N, wi);

    return Gs * f;
}

varying Color
RetroreflectionBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                               const varying BsdfSlice &slice,
                               varying float r1, varying float r2,
                               varying Vec3f &wi, varying float &pdf)
{
    const varying RetroreflectionBsdfLobe * uniform retroLobe =
        (const varying RetroreflectionBsdfLobe * uniform) lobe;

    const float cosNO = dot(retroLobe->mN, slice.mWo);
    if (cosNO <= 0.0f) {
        wi = Vec3f_ctor(0.0f);
        pdf = 0.0f;
        return Color_ctor(0.0f);
    }

    // sample the retro-reflective phong lobe
    // section 3.3.1 in [1]
    // convert roughness to phong exponent (see Walter'07)
    const float alphaP = 2.0f * rcp(retroLobe->mRoughness * retroLobe->mRoughness) - 2.0f;
    const float cosThetaM = pow(r1, rcp(alphaP + 1.0f));
    const float sinThetaM = sqrt(1.0f - cosThetaM*cosThetaM);
    const float phiM = 2.0f * sPi * r2;

    Vec3f m = computeLocalSphericalDirection(cosThetaM, sinThetaM, phiM);

    //sample along the outgoing vector
    ReferenceFrame frame;
    ReferenceFrame_init(frame, slice.mWo);
    wi = localToGlobal(frame, m);

    return RetroreflectionBsdfLobe_eval(lobe, slice, wi, &pdf);
}

varying Color
RetroreflectionBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                               const varying BsdfSlice &slice)
{
    const varying RetroreflectionBsdfLobe *uniform retroLobe =
            (const varying RetroreflectionBsdfLobe * uniform) lobe;

    float cosThetaWo = max(dot(retroLobe->mN, slice.mWo), 0.0f);
    return BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo);
}

void
RetroreflectionBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                                      const varying Vec3f &wo,
                                      const varying Vec3f &wi,
                                      varying float r1, varying float r2,
                                      const varying Vec3f &dNdx,
                                      const varying Vec3f &dNdy,
                                      varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    const varying RetroreflectionBsdfLobe * uniform retroLobe =
        (const varying RetroreflectionBsdfLobe * uniform) lobe;

    // Reverse incoming differentials, based on [3]
    // Factors taken from CookTorrance roughness-based differential computation
    // TODO: Aren't we missing the call to computeReflectionDirectionDifferential() ?
    dDdx = dDdx * -retroLobe->mdDFactor;
    dDdy = dDdy * -retroLobe->mdDFactor;
}

uniform bool
RetroreflectionBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                                    uniform BsdfLobeProperty property,
                                    varying float * uniform dest)
{
    const varying RetroreflectionBsdfLobe * uniform retroLobe =
        (const varying RetroreflectionBsdfLobe * uniform) lobe;

    uniform bool handled = true;

    switch (property)
    {
    case BSDF_LOBE_PROPERTY_ROUGHNESS:
        *dest       = retroLobe->mInputRoughness;
        *(dest + 1) = retroLobe->mInputRoughness;
        break;
    case BSDF_LOBE_PROPERTY_NORMAL:
        *dest       = retroLobe->mN.x;
        *(dest + 1) = retroLobe->mN.y;
        *(dest + 2) = retroLobe->mN.z;
        break;
    default:
        handled = false;
        break;
    }

    return handled;
}

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(RetroreflectionBsdfLobe);

