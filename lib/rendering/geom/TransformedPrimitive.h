// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file TransformedPrimitive.h
/// $Id$
///

#pragma once
#ifndef TRANSFORMEDPRIMITIVE_H
#define TRANSFORMEDPRIMITIVE_H

#include <moonray/rendering/geom/Primitive.h>

#include <moonray/rendering/bvh/shading/Xform.h>

namespace moonray {
namespace geom {

class PrimitiveVisitor;

/// @class TransformedPrimitive
/// @brief Primitive that contains a transform and a primitive,
///        which serves similar purpose of pivot point in MAYA
class TransformedPrimitive : public moonray::geom::Primitive
{
public:
    TransformedPrimitive(const shading::XformSamples& xform,
            std::unique_ptr<Primitive> primitive);

    ~TransformedPrimitive();

    virtual void accept(PrimitiveVisitor& v) override;

    virtual size_type getMemory() const override;

    shading::XformSamples getXformSamples() const;

    virtual size_type getMotionSamplesCount() const override;

    const std::unique_ptr<Primitive>& getPrimitive() const;

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

#endif // TRANSFORMEDPRIMITIVE_H
