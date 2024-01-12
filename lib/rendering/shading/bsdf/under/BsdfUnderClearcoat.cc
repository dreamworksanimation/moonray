// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfUnderClearcoat.cc
/// $Id$
///

#include "BsdfUnderClearcoat.h"
#include <moonray/rendering/shading/Util.h>

#include <moonray/rendering/shading/ispc/bsdf/under/BsdfUnderClearcoat_ispc_stubs.h>

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

// BsdfLobe API
Color
UnderClearcoatBsdfLobe::eval(const BsdfSlice &slice,
                             const Vec3f &wi,
                             float *pdf) const
{
    if (pdf != nullptr)
        *pdf = 0.0f;

    Vec3f woPrime;
    float cosThetaWo, cosThetaWoPrime;
    if (!computeRefractionDirection(mN,
                                    slice.getWo(),
                                    mNeta,
                                    woPrime,
                                    cosThetaWo,
                                    cosThetaWoPrime))
    {
        // Total Outward Reflection handled by the clearcoat layer on top
        return scene_rdl2::math::sBlack;
    }
    if (cosThetaWo <= scene_rdl2::math::sEpsilon) return scene_rdl2::math::sBlack;

    // Reverse refraction direction
    woPrime = -woPrime;

    BsdfSlice underSlice(slice.getNg(),
                         woPrime,
                         slice.getIncludeCosineTerm(),
                         slice.getEntering(),
                         slice.getShadowTerminatorFix(),
                         slice.getFlags());

    Vec3f wiPrime;
    float pdfPrime = 0.0f;
    float cosThetaWi, cosThetaWiPrime;
    if (!computeRefractionDirection(mN,
                                    wi,
                                    mNeta,
                                    wiPrime,
                                    cosThetaWi,
                                    cosThetaWiPrime))
    {
        // Total Outward Reflection handled by the clearcoat layer on top
        return scene_rdl2::math::sBlack;
    }

    if (cosThetaWi <= scene_rdl2::math::sEpsilon) return scene_rdl2::math::sBlack;

    // Reverse the refraction vector to face outward
    wiPrime = -wiPrime;
    Color result = mUnder->eval(underSlice, wiPrime, &pdfPrime);

    // portion of light transmitted through air->clearcoat interface approximated using cosThetaWo
    result *= computeScaleAndFresnel(cosThetaWo);

    // Clearcoat Absorption
    const Color ct = computeTransmission(cosThetaWoPrime, cosThetaWiPrime);

    // The light that bounces off the under lobe does not entirely exit the clearcoat immediately.
    // Here we model multiple 'bounces' within the clearcoat layer itself.
    // At each bounce some portion of light escapes into the 'air', while the rest is reflected
    // inward. The portion that is reflected inward is attenuated by clearcoat absorption, and
    // then again reflected off the under lobe, and this process repeats. We use a geometric
    // series to approximate the result.
    // The series looks like:
    // Ti*UnderBSDF*ct*To + Ti*UnderBSDF*ct^2*Ro*AvgReflectanceUnderLobe*To + Ti*UnderBSDF*ct^3*Ro^2*AvgReflectanceUnderLobe^2*To + ...(so on)
    // This is equivalent to the following sum:
    // Ti*UnderBSDF*ct*To * 1/(1 - ct*Ro*AveReflectanceUnderLobe)
    const Color Fr = mExitingFresnel->eval(cosThetaWoPrime);
    const Color Ft = scene_rdl2::math::sWhite - Fr;
    const Color albedo = mUnder->albedo(underSlice);
    result = (result * ct * Ft) * rcp(maxColor(scene_rdl2::math::sWhite - (ct * Fr * albedo), 0.01f));

    // Now we account for the change in variables because of bending rays.
    // For the derivation, please look in:
    // /work/gshad/moonshine/papers/ClearcoatChangeOfVariables.pdf
    float jacobian = scene_rdl2::math::abs(cosThetaWi) / scene_rdl2::math::max(scene_rdl2::math::abs(cosThetaWiPrime),0.01f);
    if (mUnder->matchesFlag(BsdfLobe::GLOSSY)) {
        jacobian *= mNeta*mNeta;
    }
    if (pdf != nullptr)
        *pdf = pdfPrime * jacobian;

    result *= jacobian;

    return result;
}

Color
UnderClearcoatBsdfLobe::sample(const BsdfSlice &slice,
                               float r1, float r2,
                               Vec3f &wi,
                               float &pdf) const
{
    pdf = 0.0f;
    Vec3f woPrime;
    Color result;
    float cosThetaWo, cosThetaWoPrime;
    if (!computeRefractionDirection(mN,
                                    slice.getWo(),
                                    mNeta,
                                    woPrime,
                                    cosThetaWo,
                                    cosThetaWoPrime))
    {
        // Total Outward Reflection
        return scene_rdl2::math::sBlack;
    }
    // Reverse refraction direction
    woPrime = -woPrime;
    BsdfSlice underSlice(slice.getNg(),
                         woPrime,
                         slice.getIncludeCosineTerm(),
                         slice.getEntering(),
                         slice.getShadowTerminatorFix(),
                         slice.getFlags());

    // sample the under lobe
    Vec3f wiPrime;
    float pdfPrime;
    result = mUnder->sample(underSlice, r1, r2, wiPrime, pdfPrime);

    if (scene_rdl2::math::isZero(pdfPrime)) {
        return scene_rdl2::math::sBlack;
    }

    float cosThetaWi, cosThetaWiPrime;
    if (!computeRefractionDirection(mN,
                                    wiPrime,
                                    scene_rdl2::math::rcp(mNeta),
                                    wi,
                                    cosThetaWiPrime,
                                    cosThetaWi))
    {
        // Total Internal Reflection
        if (mPassThroughTIRWhenSampling) {
            // Assume this ray bounces around inside the clearcoat interface
            // and eventually gets back out  in the same direction.
            // Pass the ray through
            wi = wiPrime;
            cosThetaWi = cosThetaWiPrime;
        }
        else {
            // Discard this sample
            return scene_rdl2::math::sBlack;
        }
    }
    else {
        // Reverse the refracted vector to point outward
        wi = -wi;
    }

    // portion of light transmitted through air->clearcoat interface approximated using cosThetaWo
    result *= computeScaleAndFresnel(cosThetaWo);

    // Clearcoat Absorption
    const Color ct = computeTransmission(cosThetaWoPrime, cosThetaWiPrime);

    // The light that bounces off the under lobe does not entirely exit the clearcoat immediately.
    // Here we model multiple 'bounces' within the clearcoat layer itself.
    // At each bounce some portion of light escapes into the 'air', while the rest is reflected
    // inward. The portion that is reflected inward is attenuated by clearcoat absorption, and
    // then again reflected off the under lobe, and this process repeats. We use a geometric
    // series to approximate the result.
    // The series looks like:
    // Ti*UnderBSDF*ct*To + Ti*UnderBSDF*ct^2*Ro*AvgReflectanceUnderLobe*To + Ti*UnderBSDF*ct^3*Ro^2*AvgReflectanceUnderLobe^2*To + ...(so on)
    // This is equivalent to the following sum:
    // Ti*UnderBSDF*ct*To * 1/(1 - ct*Ro*AveReflectanceUnderLobe)
    const Color Fr = mExitingFresnel->eval(cosThetaWoPrime);
    const Color Ft = scene_rdl2::math::sWhite - Fr;
    const Color albedo = mUnder->albedo(underSlice);
    result = (result * ct * Ft) * rcp(maxColor(scene_rdl2::math::sWhite - (ct * Fr * albedo), 0.01f));

    // Now we account for the change in variables because of bending rays.
    // For the derivation, please look in:
    // /work/gshad/moonshine/papers/ClearcoatChangeOfVariables.pdf
    float jacobian = scene_rdl2::math::abs(cosThetaWi) / scene_rdl2::math::max(scene_rdl2::math::abs(cosThetaWiPrime),0.01f);
    if (mUnder->matchesFlag(BsdfLobe::GLOSSY)) {
        jacobian *= mNeta*mNeta;
    }
    pdf = pdfPrime * jacobian;

    result *= jacobian;

    return result;
}

//----------------------------------------------------------------------------
ISPC_UTIL_TYPEDEF_STRUCT(UnderClearcoatBsdfLobe, UnderClearcoatBsdfLobev);

} // namespace shading
} // namespace moonray

