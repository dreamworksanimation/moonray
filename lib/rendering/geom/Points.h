// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Points.h
/// $Id$
///
#pragma once
#ifndef POINTS_H
#define POINTS_H

#include <moonray/rendering/geom/LayerAssignmentId.h>
#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/VertexBuffer.h>

#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/geom/internal/InterleavedTraits.h>

namespace moonray {
namespace geom {


/// @class Points
/// @brief Primitive that composed by a set of points that is intended
///        to be used for particles effects
class Points : public moonray::geom::Primitive
{
public:
    typedef moonray::geom::VertexBuffer<Vec3f, InterleavedTraits> VertexBuffer;
    typedef std::vector<float> RadiusBuffer;

    Points(VertexBuffer&& position, RadiusBuffer&& radius,
            LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    ~Points();

    virtual void accept(PrimitiveVisitor& v) override;

    virtual size_type getMemory() const override;

    virtual size_type getMotionSamplesCount() const override;

    /// set the primitive name
    /// @param name the name of this primitive
    void setName(const std::string& name);

    void setCurvedMotionBlurSampleCount(int count);

    /// get the primitive name
    /// @return name the name of this primitive
    const std::string& getName() const;

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

#endif // POINTS_H
