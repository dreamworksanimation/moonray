// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VdbVolume.h
///

#pragma once
#ifndef VDBVOLUME_H
#define VDBVOLUME_H

#include <moonray/rendering/geom/LayerAssignmentId.h>
#include <moonray/rendering/geom/Primitive.h>

#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/Xform.h>

namespace moonray {
namespace geom {

class MotionBlurParams;

/// @class VdbVolume
class VdbVolume : public moonray::geom::Primitive
{
public:

    enum class Interpolation
    {
        POINT,
        BOX,
        QUADRATIC
    };

    struct VdbInitData
    {
        std::string mVdbFilePath;
        std::string mDensityGridName;
        std::string mEmissionGridName;
        std::string mVelocityGridName;
    };

    VdbVolume(const VdbInitData& vdbInitData,
            const MotionBlurParams& motionBlurParams,
            LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    // Default constructor allows derived classes to skip
    // creation of mImpl
    VdbVolume();

    ~VdbVolume();

    void accept(PrimitiveVisitor& v) override;

    virtual size_type getMemory() const override;

    virtual size_type getMotionSamplesCount() const override;

    /// set the primitive name
    /// @param name the name of this primitive
    virtual void setName(const std::string& name);

    /// get the primitive name
    /// @return name the name of this primitive
    virtual const std::string& getName() const;

    /// set the interpolation mode for voxel sampling
    /// @param interpolation the interpolation mode
    virtual void setInterpolation(Interpolation interpolation);

    /// set the velocity scale for motion blur
    virtual void setVelocityScale(float velocityScale);

    /// set the velocity sample rate for motion blur
    virtual void setVelocitySampleRate(float velocitySampleRate);

    /// set the emission sample rate for emission illumination
    virtual void setEmissionSampleRate(float emissionSampleRate);

private:
    /// @remark For renderer internal use, procedural should never call this
    /// @internal
    virtual internal::Primitive* getPrimitiveImpl() override;

    /// @remark For renderer internal use, procedural should never call this
    /// @internal
    virtual void transformPrimitive(
            const MotionBlurParams& motionBlurParams,
            const shading::XformSamples& prim2render) override;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace geom
} // namespace moonray

#endif // VDBVOLUME_H
