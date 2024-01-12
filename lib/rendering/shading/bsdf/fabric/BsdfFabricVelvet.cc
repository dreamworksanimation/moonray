// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file FabricVelvetBsdfLobe.cc
/// $Id$
///

#include "BsdfFabric.h"
#include "VelvetAlbedo.h"

#include <moonray/rendering/shading/ispc/bsdf/fabric/BsdfFabricVelvet_ispc_stubs.h>

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

//----------------------------------------------------------------------------
// Constructor / Destructor
FabricVelvetBsdfLobe::FabricVelvetBsdfLobe(const Vec3f &N,
                                           const float roughness,
                                           const Color& color) :
                                               BsdfLobe(Type(REFLECTION | GLOSSY),
                                                        DifferentialFlags(IGNORES_INCOMING_DIFFERENTIALS),
                                                        /* isSpherical = */ false,
                                                        PROPERTY_COLOR | PROPERTY_PBR_VALIDITY |
                                                        PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
                                                        mFrame(N),
                                               mColor(color)
{

    mRoughness = clamp(roughness, 0.05f, 1.0f);

    mSpecularExponent = 1.0f / mRoughness;
    const float* nTable = reinterpret_cast<const float*>(ispc::BsdfFabricVelvet_normalizationTable());
    const int index = static_cast<int>(scene_rdl2::math::floor(mRoughness*ispc::VELVET_NORMALIZATION_TABLE_SIZE - 1));
    mNormalizationFactor = 1.0f / nTable[index];

    // Derive a directional differential scale that varies according to width
    mdDFactor = sdDFactorMin + mRoughness * sdDFactorSlope;
}

Color
FabricVelvetBsdfLobe::eval(const BsdfSlice &slice,
                           const Vec3f &wi,
                           float *pdf) const
{
    if (pdf != NULL) {
        *pdf = 0.0f;
    }

    const Vec3f N = mFrame.getN();

    const float cosThetaWi = scene_rdl2::math::dot(N, wi);
    if (cosThetaWi <= scene_rdl2::math::sEpsilon) return scene_rdl2::math::sBlack;

    const float cosThetaWo = scene_rdl2::math::dot(N, slice.getWo());
    if (cosThetaWo <= sEpsilon) return scene_rdl2::math::sBlack;

    Vec3f H;
    if (!computeNormalizedHalfVector(slice.getWo(), wi, H)) {
        return sBlack;
    }

    // This is the Velvet BRDF for grazing angle highlights
    // BRDF = sin(N, H)^exponent
    const float cosNDotH = clamp(scene_rdl2::math::dot(N, H), -.99f, .99f);
    const float sinSquared = (1.0f - cosNDotH*cosNDotH);
    const float sinNDotH   = scene_rdl2::math::sqrt(sinSquared);

    float velvetFactor = mNormalizationFactor * scene_rdl2::math::pow(sinNDotH, mSpecularExponent);

    // MOONSHINE-921
    // Add an extra cosineThetaWi term to soften the shadow terminator
    // Note - this breaks bidirectionality but artists prefer this look.
    velvetFactor *= cosThetaWi;

    if (pdf != NULL) {
        *pdf = cosThetaWi * scene_rdl2::math::sOneOverPi;
    }

    const Color f = computeScaleAndFresnel(cosThetaWo) *
                    velvetFactor * mColor *
                    (slice.getIncludeCosineTerm() ? cosThetaWi : 1.0f);

    return f;
}

Color
FabricVelvetBsdfLobe::sample(const BsdfSlice &slice,
                       float r1, float r2,
                       Vec3f &wi,
                       float &pdf) const
{
    // cosine-weighted hemisphere sampling
    wi = mFrame.localToGlobal(sampleLocalHemisphereCosine(r1, r2));
    return eval(slice, wi, &pdf);
}

//----------------------------------------------------------------------------
Color
FabricVelvetBsdfLobe::albedo(const BsdfSlice &slice) const
{
    float cosThetaWo = scene_rdl2::math::max(dot(mFrame.getN(), slice.getWo()), 0.0f);
    return (computeScaleAndFresnel(cosThetaWo) * mColor *
            VelvetAlbedo::at(cosThetaWo, mRoughness));
}

// no surface curvature required (similar to the hair lobe derivatives)
void
FabricVelvetBsdfLobe::differentials(const Vec3f &wo, const Vec3f &wi,
                                    float r1, float r2,
                                    const Vec3f &dNdx, const Vec3f &dNdy,
                                    Vec3f &dDdx, Vec3f &dDdy) const
{
    // It's complex to capture the full derivative. Instead we use the
    // derivative of a mirror reflection about the H vector, and scale the
    // length of the directional derivative proportionally with roughness.
    Vec3f H;
    if (!computeNormalizedHalfVector(wo, wi, H)) {
        H = mFrame.getN();
    }
    computeReflectionDirectionDifferentialNoCurvature(wo, wi, H, dDdx, dDdy);

    dDdx *= mdDFactor;
    dDdy *= mdDFactor;
}


} // namespace shading
} // namespace moonray

