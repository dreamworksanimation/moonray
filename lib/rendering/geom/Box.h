// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Box.h
/// $Id$
///
#pragma once

#include <moonray/rendering/geom/LayerAssignmentId.h>
#include <moonray/rendering/geom/Primitive.h>

#include <moonray/rendering/bvh/shading/Xform.h>

namespace moonray {
namespace geom {

/// @class Box
/// @brief quadric Box
class Box : public moonray::geom::Primitive
{
public:
    Box(float length, float width, float height, LayerAssignmentId&& layerAssignmentId,
            shading::PrimitiveAttributeTable&& primitiveAttributeTable);

    ~Box();

    virtual void accept(PrimitiveVisitor& v) override;

    virtual size_type getMemory() const override;

    virtual size_type getMotionSamplesCount() const override;

    /// set the primitive name
    /// @param name the name of this primitive
    void setName(const std::string& name);

    /// get the primitive name
    /// @return name the name of this primitive
    const std::string& getName() const;

    // set whether the primitive is single sided
    void setIsSingleSided(bool isSingleSided);

    // get whether the primitive is single sided
    bool getIsSingleSided() const;

    // set whether normals are reversed
    void setIsNormalReversed(bool isNormalReversed);

    // get whether normals are reversed
    bool getIsNormalReversed() const;

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


