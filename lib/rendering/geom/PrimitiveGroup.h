// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PrimitiveGroup.h
/// $Id$
///

#pragma once

#include <moonray/rendering/geom/Primitive.h>

#include <moonray/rendering/bvh/shading/Xform.h>

namespace moonray {
namespace geom {

class PrimitiveVisitor;

/// @class PrimitiveGroup
/// @Primitive that group a set of primitives
class PrimitiveGroup : public moonray::geom::Primitive
{
public:
    PrimitiveGroup();

    ~PrimitiveGroup();

    virtual void accept(PrimitiveVisitor& v) override;

    virtual size_type getMemory() const override;

    virtual size_type getMotionSamplesCount() const override;

    size_type getPrimitivesCount() const;

    void addPrimitive(std::unique_ptr<Primitive> primitive);

    void reserve(size_type n);

    void clear();

    void forEachPrimitive(PrimitiveVisitor& visitor,
            bool parallel = true);

    void forEachStatic(PrimitiveVisitor& visitor,
            bool parallel = true);

    void forEachDynamic(PrimitiveVisitor& visitor,
            bool parallel = true);

    void forEachDeformable(PrimitiveVisitor& visitor,
            bool parallel = true);

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


