// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfUnder.ispc

#include "BsdfUnder.isph"
#include <moonray/rendering/shading/ispc/bsdf/BsdfSlice.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>

varying Color
UnderBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                   const varying BsdfSlice &slice,
                   const Vec3f &wi,
                   varying float * uniform pdf)
{
    const varying UnderBsdfLobe * uniform under =
        (const varying UnderBsdfLobe * uniform) lobe;

    // We only want to apply the effects of the UnderBsdfLobe for
    // the lanes where isLaneActive(under->mUnderLobeMask) == true.
    // For the inactive lanes, many of the members (thickness, attenuationColor, etc)
    // are likely uninitialized.
    if (!isActive(under->mFresnel) ||
        !isLaneActive(under->mUnderLobeMask)) {
        // just evaluate under lobe
        return BsdfLobe_eval(under->mUnder, slice, wi, pdf);
    }

    // Note: we assume this lobe has been setup with a OneMinus*Fresnel
    // We use cosThetaWo to evaluate the fresnel term, as an approximation
    // of what hDotWi would be for the "over" lobe.
    float cosThetaWo = (!BsdfLobe_getFresnel(lobe)  ?  1.0f  :
                        max(dot(under->mN, slice.mWo), 0.0f));

    float cosThetaWi = max(dot(under->mN, wi), 0.0f);

    // Forward to the under lobe and apply fresnel
    Color result =  BsdfLobe_eval(under->mUnder,
                                  slice,
                                  wi,
                                  pdf);

    result = result * BsdfLobe_computeScaleAndFresnel(lobe,
                                                      cosThetaWo);
    // Absorption
    const Color ct = computeTransmission(under->mThickness,
                                         under->mAttenuationColor,
                                         under->mAttenuationWeight,
                                         cosThetaWo,
                                         cosThetaWi);

    result = result * ct;
    return result;

}

varying Color
UnderBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                     const varying BsdfSlice &slice,
                     varying float r1, varying float r2,
                     varying Vec3f &wi, varying float &pdf)
{
    const varying UnderBsdfLobe * uniform under =
        (const varying UnderBsdfLobe * uniform) lobe;

    // We only want to apply the effects of the UnderBsdfLobe for
    // the lanes where isLaneActive(under->mUnderLobeMask) == true.
    // For the inactive lanes, many of the members (thickness, attenuationColor, etc)
    // are likely uninitialized.
    if (!isActive(under->mFresnel) ||
        !isLaneActive(under->mUnderLobeMask)) {
        // just sample under lobe
        return BsdfLobe_sample(under->mUnder, slice, r1, r2, wi, pdf);
    }

    // Note: we assume this lobe has been setup with a OneMinus*Fresnel
    // We use cosThetaWo to evaluate the fresnel term, as an approximation
    // of what hDotWi would be for the "over" lobe.
    float cosThetaWo = (!BsdfLobe_getFresnel(lobe)  ?  1.0f  :
                        max(dot(under->mN, slice.mWo), 0.0f));

    // Forward to the under lobe and apply fresnel
    Color result = BsdfLobe_sample(under->mUnder,
                                   slice,
                                   r1, r2,
                                   wi,
                                   pdf);

    float cosThetaWi = max(dot(under->mN, wi), 0.0f);

    result = result * BsdfLobe_computeScaleAndFresnel(lobe,
                                                      cosThetaWo);
    // Absorption
    const Color ct = computeTransmission(under->mThickness,
                                         under->mAttenuationColor,
                                         under->mAttenuationWeight,
                                         cosThetaWo,
                                         cosThetaWi);

    result = result * ct;
    return result;
}

/// @brief albedo function
varying Color
UnderBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                     const varying BsdfSlice &slice)
{
    const varying UnderBsdfLobe * uniform under =
        (const varying UnderBsdfLobe * uniform) lobe;

    // We only want to apply the effects of the UnderBsdfLobe for
    // the lanes where isLaneActive(under->mUnderLobeMask) == true.
    // For the inactive lanes, many of the members (thickness, attenuationColor, etc)
    // are likely uninitialized.
    if (!isActive(under->mFresnel) ||
        !isLaneActive(under->mUnderLobeMask)) {
        // just forward call to the lobe we are wrapping
        return BsdfLobe_albedo(under->mUnder, slice);
    }

    // See eval() for info
    const float cosThetaWo =
        (!BsdfLobe_getFresnel((varying BsdfLobe * uniform) lobe) ?
                 1.0f  :  max(dot(under->mN, slice.mWo), 0.0f));

    // Forward to the under lobe and apply fresnel
    return BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo) *
           BsdfLobe_albedo(under->mUnder, slice);
}

/// @brief differentials function
void
UnderBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                            const varying Vec3f &wo,
                            const varying Vec3f &wi,
                            varying float r1, varying float r2,
                            const varying Vec3f &dNdx,
                            const varying Vec3f &dNdy,
                            varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    const varying UnderBsdfLobe * uniform under =
        (const varying UnderBsdfLobe * uniform) lobe;

    // Forward to the under lobe
    BsdfLobe_differentials(under->mUnder, wo, wi, r1, r2, dNdx, dNdy, dDdx, dDdy);
}

/// @brief getAttribute function
uniform bool
UnderBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                          uniform BsdfLobeProperty property,
                          varying float * uniform dest)
{
    const varying UnderBsdfLobe * uniform under =
        (const varying UnderBsdfLobe * uniform) lobe;

    // THINK: interesting question, handle PROPERTY_NORMAL ourselves?
    return BsdfLobe_getProperty(under->mUnder, property, dest);
}

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(UnderBsdfLobe);

