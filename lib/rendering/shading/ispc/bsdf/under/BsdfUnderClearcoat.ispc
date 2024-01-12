// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file BsdfUnderClearcoat.ispc

#include "BsdfUnderClearcoat.isph"
#include <moonray/rendering/shading/ispc/bsdf/BsdfSlice.isph>
#include <moonray/rendering/shading/ispc/Util.isph>

#include <scene_rdl2/common/platform/IspcUtil.isph>

/// @brief calls UnderBsdfLobe_eval and attenuates the result using transmission color
varying Color
UnderClearcoatBsdfLobe_eval(const varying BsdfLobe * uniform lobe,
                            const varying BsdfSlice &slice,
                            const Vec3f &wi,
                            varying float * uniform pdf)
{
    const varying UnderClearcoatBsdfLobe * uniform under =
        (const varying UnderClearcoatBsdfLobe * uniform) lobe;

    // Early Exit
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

    float cosThetaWi, cosThetaWiPrime;
    Vec3f wiPrime;
    if (!computeRefractionDirection(under->mN,
                                    wi,
                                    under->mNeta,
                                    wiPrime,
                                    cosThetaWi,
                                    cosThetaWiPrime))
    {
        // Total Outward Reflection
        return sBlack;
    }
    wiPrime = neg(wiPrime);

    float pdfPrime;
    // evaluate under lobe
    Color result = BsdfLobe_eval(under->mUnder,
                                 underSlice,
                                 wiPrime,
                                 &pdfPrime);

    // portion of light transmitted through air->clearcoat interface approximated using cosThetaWo
    result = result * BsdfLobe_computeScaleAndFresnel(lobe,
                                                      cosThetaWo);

    // Clearcoat Absorption
    const Color ct = computeTransmission(under->mThickness,
                                         under->mAttenuationColor,
                                         under->mAttenuationWeight,
                                         cosThetaWoPrime,
                                         cosThetaWiPrime);

    // At each bounce some portion of light escapes into the 'air', while the rest is reflected
    // inward. The portion that is reflected inward is attenuated by clearcoat absorption, and
    // then again reflected off the under lobe, and this process repeats. We use a geometric
    // series to approximate the result.
    // The series looks like:
    // Ti*UnderBSDF*ct*To + Ti*UnderBSDF*ct^2*Ro*AvgReflectanceUnderLobe*To + Ti*UnderBSDF*ct^3*Ro^2*AvgReflectanceUnderLobe^2*To + ...(so on)
    // This is equivalent to the following sum:
    // Ti*UnderBSDF*ct*To * 1/(1 - ct*Ro*AveReflectanceUnderLobe)
    const Color Fr = Fresnel_eval(under->mExitingFresnel, cosThetaWoPrime);
    const Color Ft = sWhite - Fr;
    const Color albedo = BsdfLobe_albedo(under->mUnder,
                                         underSlice);
    result = (result * ct * Ft) * rcp(max(sWhite - (ct * Fr * albedo), Color_ctor(0.01f)));

    // Now we account for the change in variables because of bending rays.
    // For the derivation, please look in:
    // /work/gshad/moonshine/papers/ClearcoatChangeOfVariables.pdf
    float jacobian = abs(cosThetaWi) / max(abs(cosThetaWiPrime),0.01f);
    if (BsdfLobe_matchesFlag(under->mUnder,
                             BSDF_LOBE_TYPE_GLOSSY)) {
        jacobian *= under->mNeta*under->mNeta;
    }
    if (pdf != nullptr)
        *pdf = pdfPrime * jacobian;

    result = result * jacobian;

    return result;
}

/// @brief calls UnderBsdfLobe_sample and attenuates the result using transmission color
varying Color
UnderClearcoatBsdfLobe_sample(const varying BsdfLobe * uniform lobe,
                              const varying BsdfSlice &slice,
                              varying float r1, varying float r2,
                              varying Vec3f &wi,
                              varying float &pdf)
{
    const varying UnderClearcoatBsdfLobe * uniform under =
        (const varying UnderClearcoatBsdfLobe * uniform) lobe;

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

    // sample the under lobe
    Vec3f wiPrime;
    float pdfPrime;
    Color result =  BsdfLobe_sample(under->mUnder,
                                    underSlice,
                                    r1, r2,
                                    wiPrime,
                                    pdfPrime);

    if (isZero(pdfPrime)) {
        return sBlack;
    }

    float cosThetaWi, cosThetaWiPrime;
    if (!computeRefractionDirection(under->mN,
                                    wiPrime,
                                    rcp(under->mNeta),
                                    wi,
                                    cosThetaWiPrime,
                                    cosThetaWi))
    {
        // Total Internal Reflection
        if (under->mPassThroughTIRWhenSampling) {
            // Assume this ray bounces around inside the clearcoat interface
            // and eventually gets back out  in the same direction.
            // Pass the ray through
            wi = wiPrime;
            cosThetaWi = cosThetaWiPrime;
        }
        else {
            // Discard this sample
            return sBlack;
        }
    }
    else {
        // Reverse the refracted vector to point outward
        wi = neg(wi);
    }

    // portion of light transmitted through air->clearcoat interface approximated using cosThetaWo
    result = result * BsdfLobe_computeScaleAndFresnel(lobe,
                                                      cosThetaWo);

    // Clearcoat Absorption
    const Color ct = computeTransmission(under->mThickness,
                                         under->mAttenuationColor,
                                         under->mAttenuationWeight,
                                         cosThetaWoPrime,
                                         cosThetaWiPrime);

    // At each bounce some portion of light escapes into the 'air', while the rest is reflected
    // inward. The portion that is reflected inward is attenuated by clearcoat absorption, and
    // then again reflected off the under lobe, and this process repeats. We use a geometric
    // series to approximate the result.
    // The series looks like:
    // Ti*UnderBSDF*ct*To + Ti*UnderBSDF*ct^2*Ro*AvgReflectanceUnderLobe*To + Ti*UnderBSDF*ct^3*Ro^2*AvgReflectanceUnderLobe^2*To + ...(so on)
    // This is equivalent to the following sum:
    // Ti*UnderBSDF*ct*To * 1/(1 - ct*Ro*AveReflectanceUnderLobe)
    const Color Fr = Fresnel_eval(under->mExitingFresnel, cosThetaWoPrime);
    const Color Ft = sWhite - Fr;
    const Color albedo = BsdfLobe_albedo(under->mUnder,
                                         underSlice);
    result = (result * ct * Ft) * rcp(max(sWhite - (ct * Fr * albedo), Color_ctor(0.01f)));

    // Now we account for the change in variables because of bending rays.
    // For the derivation, please look in:
    // /work/gshad/moonshine/papers/ClearcoatChangeOfVariables.pdf
    float jacobian = abs(cosThetaWi) / max(abs(cosThetaWiPrime),0.01f);
    if (BsdfLobe_matchesFlag(under->mUnder,
                             BSDF_LOBE_TYPE_GLOSSY)) {
        jacobian *= under->mNeta*under->mNeta;
    }
    pdf = pdfPrime * jacobian;

    result = result * jacobian;

    return result;
}

/// @brief albedo function
varying Color
UnderClearcoatBsdfLobe_albedo(const varying BsdfLobe * uniform lobe,
                     const varying BsdfSlice &slice)
{
    const varying UnderClearcoatBsdfLobe * uniform under =
        (const varying UnderClearcoatBsdfLobe * uniform) lobe;

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
    float cosThetaWo = (!BsdfLobe_getFresnel((varying BsdfLobe * uniform) lobe)  ?  1.0f  :
                        max(dot(under->mN, slice.mWo), 0.0f));

    // Forward to the under lobe and apply fresnel
    return BsdfLobe_computeScaleAndFresnel(lobe, cosThetaWo) *
           BsdfLobe_albedo(under->mUnder, slice);
}

/// @brief differentials function
void
UnderClearcoatBsdfLobe_differentials(const varying BsdfLobe * uniform lobe,
                            const varying Vec3f &wo,
                            const varying Vec3f &wi,
                            varying float r1, varying float r2,
                            const varying Vec3f &dNdx,
                            const varying Vec3f &dNdy,
                            varying Vec3f &dDdx, varying Vec3f &dDdy)
{
    const varying UnderClearcoatBsdfLobe * uniform under =
        (const varying UnderClearcoatBsdfLobe * uniform) lobe;

    // Forward to the under lobe
    BsdfLobe_differentials(under->mUnder, wo, wi, r1, r2, dNdx, dNdy, dDdx, dDdy);
}

/// @brief getAttribute function
uniform bool
UnderClearcoatBsdfLobe_getProperty(const varying BsdfLobe * uniform lobe,
                          uniform BsdfLobeProperty property,
                          varying float * uniform dest)
{
    const varying UnderClearcoatBsdfLobe * uniform under =
        (const varying UnderClearcoatBsdfLobe * uniform) lobe;

    // THINK: interesting question, handle PROPERTY_NORMAL ourselves?
    return BsdfLobe_getProperty(under->mUnder, property, dest);
}

#pragma ignore warning(all)
ISPC_UTIL_EXPORT_STRUCT_TO_HEADER(UnderClearcoatBsdfLobe);

