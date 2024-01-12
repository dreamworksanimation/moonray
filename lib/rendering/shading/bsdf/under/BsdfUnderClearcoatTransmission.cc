// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file BsdfUnderClearcoat.cc
/// $Id$
///

#include "BsdfUnderClearcoatTransmission.h"
#include <moonray/rendering/shading/Util.h>

#include <moonray/rendering/shading/ispc/bsdf/under/BsdfUnderClearcoatTransmission_ispc_stubs.h>

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

// UnderClearcoatTransmissionBsdfLobe API
Color
UnderClearcoatTransmissionBsdfLobe::eval(const BsdfSlice &slice,
                                         const Vec3f &wi,
                                         float *pdf) const
{
    if (pdf != nullptr)
        *pdf = 0.0f;

    Color result;
    Vec3f woPrime;
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

    result = mUnder->eval(underSlice, wi, pdf);

    result *= computeScaleAndFresnel(dot(mN, slice.getWo()));

    // Clearcoat Absorption
    const Color ct = computeTransmission(cosThetaWoPrime,
                                         0.0f);

    result = result * ct;

    return result;
}

Color
UnderClearcoatTransmissionBsdfLobe::sample(const BsdfSlice &slice,
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

    result = mUnder->sample(underSlice, r1, r2, wi, pdf);

    result *= computeScaleAndFresnel(cosThetaWo);

    // Clearcoat Absorption
    const Color ct = computeTransmission(cosThetaWoPrime,
                                         0.0f);

    result = result * ct;

    return result;
}

//----------------------------------------------------------------------------
//ISPC_UTIL_TYPEDEF_STRUCT(UnderClearcoatTransmissionBsdfLobe, UnderClearcoatTransmissionBsdfLobev);

} // namespace shading
} // namespace moonray

