// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file SharedPrimitive.h
/// $Id$
///

#pragma once

#include <moonray/rendering/geom/Primitive.h>

namespace moonray {
namespace geom {


class SharedPrimitive
{
    // expose private method setBVHScene/getBVHScene
    // for internal renderer use
    friend class internal::PrimitivePrivateAccess;

public:
    SharedPrimitive(std::unique_ptr<Primitive>&& primitive);

    ~SharedPrimitive();

    const std::unique_ptr<Primitive>& getPrimitive() const;

    void setHasSurfaceAssignment(bool hasSurfaceAssignments);
    bool getHasSurfaceAssignment() const;

    void setHasVolumeAssignment(bool hasVolumeAssignments);
    bool getHasVolumeAssignment() const;

private:
    /// @remark For renderer internal use, procedural should never call this
    void setBVHScene(void* bvhScene);
    /// @remark For renderer internal use, procedural should never call this
    void* getBVHScene();

private:
    struct Impl;

    std::unique_ptr<Impl> mImpl;
};

} // namespace geom
} // namespace moonray


