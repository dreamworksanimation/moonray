// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file AmorphousVolume.h
///

#pragma once
#ifndef AMORPHOUSVOLUME_H
#define AMORPHOUSVOLUME_H

#include "VdbVolume.h"

#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/Xform.h>

namespace moonray {
namespace geom {


/// @class AmorphousVolume
class AmorphousVolume : public VdbVolume
{
public:
    AmorphousVolume(const VdbInitData& vdbInitData,
            const MotionBlurParams& motionBlurParams,
            LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    ~AmorphousVolume();

    size_type getMemory() const override;

    size_type getMotionSamplesCount() const override;

    /// set the primitive name
    /// @param name the name of this primitive
    void setName(const std::string& name) override;

    /// get the primitive name
    /// @return name the name of this primitive
    const std::string& getName() const override;

    /// set the interpolation mode for voxel sampling
    /// @param interpolation the interpolation mode
    void setInterpolation(Interpolation interpolation) override;

    /// set the velocity scale for motion blur
    void setVelocityScale(float velocityScale) override;

    /// set the velocity sample rate for motion blur
    void setVelocitySampleRate(float velocitySampleRate) override;

    /// set the emission sample rate for emission illumination
    void setEmissionSampleRate(float emissionSampleRate) override;

private:
    /// @remark For renderer internal use, procedural should never call this
    /// @internal
    internal::Primitive* getPrimitiveImpl() override;

    /// @remark For renderer internal use, procedural should never call this
    /// @internal
    void transformPrimitive(const MotionBlurParams& motionBlurParams,
                            const shading::XformSamples& prim2render) override;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace geom
} // namespace moonray

#endif // AMORPHOUSVOLUME_H
