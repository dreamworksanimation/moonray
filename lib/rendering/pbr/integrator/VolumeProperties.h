// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VolumeProperties.h
///

#pragma once

#include <moonray/rendering/pbr/Types.h>

#include <scene_rdl2/common/math/Color.h>

namespace moonray {
namespace pbr {

/// This class is the building block for decoupled ray marching. A per ray
/// array of VolumeProperties can help us do quick transmittance lookup and
/// scattering sampling. It does introduces some bias because it tries to
/// approximate heterogeneous transmittance with discretized 1D step function
class VolumeProperties
{
public:
    VolumeProperties() = default;

    VolumeProperties(const scene_rdl2::math::Color& sigmaT, const scene_rdl2::math::Color& sigmaS,
                     const scene_rdl2::math::Color& sigmaTh, const scene_rdl2::math::Color& sigmaSh,
                     const scene_rdl2::math::Color& transmittance, const scene_rdl2::math::Color& transmittanceH,
                     float surfaceOpacityThreshold,
                     float g, float tStart, float delta,
                     int assignmentId):
        mSigmaT(sigmaT), mSigmaS(sigmaS),
        mSigmaTh(sigmaTh), mSigmaSh(sigmaSh),
        mTransmittance(transmittance), mTransmittanceH(transmittanceH),
        mSurfaceOpacityThreshold(surfaceOpacityThreshold),
        mG(g), mTStart(tStart), mDelta(delta), mAssignmentId(assignmentId) {}

    scene_rdl2::math::Color mSigmaT;   // extinction coefficient
    scene_rdl2::math::Color mSigmaS;   // scattering coefficient
    scene_rdl2::math::Color mSigmaTh;  // cutouts (holdouts) extinction coefficient
    scene_rdl2::math::Color mSigmaSh;  // cutouts (holdouts) scattering coefficient
    scene_rdl2::math::Color mTransmittance;   // transmittance due to regular extinction
    scene_rdl2::math::Color mTransmittanceH;  // transmittance due to cutouts (holdouts)
    float mSurfaceOpacityThreshold;    // the total accumulated opacity that's considered to be a "surface"
    float mG;
    float mTStart;
    float mDelta;
    int mAssignmentId;
};

} // namespace pbr
} // namespace moonray

