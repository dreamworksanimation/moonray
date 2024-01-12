// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfUnderClearcoatTransmission.ispc

#include "BsdfUnderClearcoatTransmission.isph"
#include <moonray/rendering/shading/ispc/bsdf/BsdfSlice.isph>
#include <moonray/rendering/shading/ispc/Util.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>

/// @brief calls UnderBsdfLobe_eval and attenuates the result using transmission color
varying Color
UnderClearcoatTransmissionBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                                        const varying BsdfSlice &slice,
                                        const Vec3f &wi,
                                        varying float * uniform pdf)
{
    const varying UnderClearcoatTransmissionBsdfLobe * uniform under =
        (const varying UnderClearcoatTransmissionBsdfLobe * uniform) lobe;

    if (pdf != nullptr)
        *pdf = 0.0f;

    // We only want to apply the effects of the UnderBsdfLobe for
    // the lanes where isLaneActive(under->mUnderLobeMask) == true.
    // For the inactive lanes, many of the members (thickness, attenuationColor, etc)
    // are likely uninitialized.
    if (!isActive(under->mFresnel) ||
        !isLaneActive(under->mUnderLobeMask)) {
        // just evaluate under lobe
        return BsdfLobe_eval(under->mUnder, slice, wi, pdf);
    }

    Color result;
    Vec3f woPrime;
    float cosThetaWo, cosThetaWoPrime;
    if (!computeRefractionDirection(under->mN,
                                    slice.mWo,
                                    under->mNeta,
                                    woPrime,
                                    cosThetaWo,
                                    cosThetaWoPrime))
     {
         // Total Outward Reflection
         return sBlack;
     }

    float cosThetaWi = max(dot(under->mN, wi), 0.0f);

    // Reverse refraction direction
    woPrime = neg(woPrime);
    BsdfSlice underSlice;
    BsdfSlice_init(&underSlice,
                   slice.mNg,
                   woPrime,
                   slice.mFlags,
                   slice.mIncludeCosineTerm,
                   slice.mEntering,
                   BsdfSlice_getShadowTerminatorFix(&slice));

    // evaluate under lobe
    result = BsdfLobe_eval(under->mUnder,
                           underSlice,
                           wi,
                           pdf);

    // portion of light transmitted through air->clearcoat interface approximated using cosThetaWo
    result = result * BsdfLobe_computeScaleAndFresnel(lobe,
                                                      cosThetaWo);

    // Clearcoat Absorption
    const Color ct = computeTransmission(under->mThickness,
                                         under->mAttenuationColor,
                                         under->mAttenuationWeight,
                                         cosThetaWoPrime,
                                         0.0f);

    result = result * ct;

    return result;
}

/// @brief calls UnderBsdfLobe_sample and attenuates the result using transmission color
varying Color
UnderClearcoatTransmissionBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                                          const varying BsdfSlice &slice,
                                          varying float r1, varying float r2,
                                          varying Vec3f &wi,
                                          varying float &pdf)
{
    const varying UnderClearcoatTransmissionBsdfLobe * uniform under =
        (const varying UnderClearcoatTransmissionBsdfLobe * uniform) lobe;

    pdf = 0.0f;

    // We only want to apply the effects of the UnderBsdfLobe for
    // the lanes where isLaneActive(under->mUnderLobeMask) == true.
    // For the inactive lanes, many of the members (thickness, attenuationColor, etc)
    // are likely uninitialized.
    if (!isActive(under->mFresnel) ||
        !isLaneActive(under->mUnderLobeMask)) {
        // just sample under lobe
        return BsdfLobe_sample(under->mUnder, slice, r1, r2, wi, pdf);
    }

    Vec3f woPrime;
    Color result;
    float cosThetaWo, cosThetaWoPrime;

    if (!computeRefractionDirection(under->mN,
                                    slice.mWo,
                                    under->mNeta,
                                    woPrime,
                                    cosThetaWo,
                                    cosThetaWoPrime))
    {
        // Total Outward Reflection
        return sBlack;
    }
    // Reverse refraction direction
    woPrime = neg(woPrime);
    BsdfSlice underSlice;
    BsdfSlice_init(&underSlice,
                   slice.mNg,
                   woPrime,
                   slice.mFlags,
                   slice.mIncludeCosineTerm,
                   slice.mEntering,
                   BsdfSlice_getShadowTerminatorFix(&slice));

    result =  BsdfLobe_sample(under->mUnder,
                              underSlice,
                              r1, r2,
                              wi,
                              pdf);

    float cosThetaWi = max(dot(under->mN, wi), 0.0f);
    // portion of light transmitted through air->clearcoat interface approximated using cosThetaWo
    result = result * BsdfLobe_computeScaleAndFresnel(lobe,
                                                      cosThetaWo);

    // Clearcoat Absorption
    const Color ct = computeTransmission(under->mThickness,
                                         under->mAttenuationColor,
                                         under->mAttenuationWeight,
                                         cosThetaWoPrime,
                                         0.0f);

    result = result * ct;

    return result;
}

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(UnderClearcoatTransmissionBsdfLobe);

