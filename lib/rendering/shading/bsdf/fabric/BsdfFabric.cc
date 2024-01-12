// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file FabricBsdfLobe.cc
/// $Id$
///

#include "BsdfFabric.h"

namespace moonray {
namespace shading {

using namespace scene_rdl2::math;

//----------------------------------------------------------------------------
// Constructor / Destructor
FabricBsdfLobe::FabricBsdfLobe(const Vec3f &N,
                               const Vec3f &T,
                               const Vec3f &threadDirection,
                               const float threadElevation,
                               const float roughness,
                               const Color& color) :
                                       BsdfLobe(Type(REFLECTION | GLOSSY),
                                                DifferentialFlags(IGNORES_INCOMING_DIFFERENTIALS),
                                                false,
                                                PROPERTY_COLOR | PROPERTY_PBR_VALIDITY |
                                                PROPERTY_NORMAL | PROPERTY_ROUGHNESS),
                                       mFrame(N, T),
                                       mSpecularExponent(1),
                                       mNormalizationFactor(1),
                                       mColor(color)
{
    // Clamp to a minimum roughness for fabric
    mRoughness = scene_rdl2::math::clamp(roughness, 0.05f, 1.0f);

    // Thread Direction
    calculateThreadDirection(threadDirection,
                             threadElevation);

    // Derive a directional differential scale that varies according to width
    // TODO Figure out if the hair shader defaults work well or we need new ones for fabric
    mdDFactor = sdDFactorMin + mRoughness * sdDFactorSlope;
}

void
FabricBsdfLobe::differentials(const Vec3f &wo, const Vec3f &wi,
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
    dDdx = normalize(dDdx) * mdDFactor;
    dDdy = normalize(dDdy) * mdDFactor;
    // TODO calculate the right factors for fabric
    // These are taken from the hair lobe
}

Color
FabricBsdfLobe::albedo(const BsdfSlice &slice) const
{
    // TODO is there a more correct albedo than the cosine-weight?
    float cosThetaWo = scene_rdl2::math::max(dot(mFrame.getN(), slice.getWo()), 0.0f);
    return computeScaleAndFresnel(cosThetaWo) * mColor;
}

bool
FabricBsdfLobe::getProperty(Property property,
                            float *dest) const
{
    bool handled = true;
    switch (property)
    {
    case PROPERTY_COLOR:
    {
        *dest       = mColor[0];
        *(dest + 1) = mColor[1];
        *(dest + 2) = mColor[2];
        break;
    }
    case PROPERTY_PBR_VALIDITY:
    {
        // Same as Albedo PBR Validity
        scene_rdl2::math::Color res = computeAlbedoPbrValidity(mColor);
        *dest       = res.r;
        *(dest + 1) = res.g;
        *(dest + 2) = res.b;
        break;
    }
    case PROPERTY_ROUGHNESS:
    {
        *dest       = mRoughness;
        *(dest + 1) = mRoughness;
        break;
    }
    case PROPERTY_NORMAL:
        {
            const Vec3f &N = mFrame.getN();
            *dest       = N.x;
            *(dest + 1) = N.y;
            *(dest + 2) = N.z;
        }
        break;
    default:
        handled = BsdfLobe::getProperty(property, dest);
        break;
    }
    return handled;
}

// Initialize the thread direction vector for fabric
// 'thread elevation' is used to rotate the fabric thread vertically (useful for silk)
void
FabricBsdfLobe::calculateThreadDirection(const Vec3f& threadDirection,
                                         float threadElevation)
{
    // Rotate the thread direction in the X,Z (T, N) plane
    Vec3f newDir;
    float sinTheta, cosTheta;
    scene_rdl2::math::sincos(scene_rdl2::math::deg2rad(threadElevation), &sinTheta, &cosTheta);
    newDir.x = threadDirection.x * cosTheta - threadDirection.z * sinTheta;
    newDir.y = threadDirection.y;
    newDir.z = threadDirection.x * sinTheta + threadDirection.z * cosTheta;

    mThreadDirection = normalize(mFrame.localToGlobal(newDir));

    // Center The frame along the thread direction now
    if (isOne(scene_rdl2::math::dot(mFrame.getN(), mThreadDirection))) {
        // If the two are colinear, just use the normal
        mFrame = ReferenceFrame(mFrame.getN());
    } else {
        mFrame = ReferenceFrame(mFrame.getN(), mThreadDirection);
    }
}

} // namespace shading
} // namespace moonray

