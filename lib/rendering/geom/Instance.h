// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Instance.h
/// $Id$
///
#pragma once
#ifndef INSTANCE_H
#define INSTANCE_H

#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/SharedPrimitive.h>

#include <moonray/rendering/bvh/shading/Xform.h>

namespace moonray {
namespace geom {

/// @class Instance
/// @brief Primitive that contains a transform and a reference to
///     shared primitive, which can be used for rendering massive amount
///     of primitives with shared geometry data
class Instance : public moonray::geom::Primitive
{
public:
    Instance(const shading::XformSamples& xform,
            std::shared_ptr<SharedPrimitive> reference,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    ~Instance();

    virtual void accept(PrimitiveVisitor& v) override;

    virtual size_type getMemory() const override;

    virtual size_type getMotionSamplesCount() const override;

    const std::shared_ptr<SharedPrimitive>& getReference() const;

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
    class Impl;
    std::unique_ptr<Impl> mImpl;
};


} // namespace geom
} // namespace moonray

#endif // INSTANCE_H
