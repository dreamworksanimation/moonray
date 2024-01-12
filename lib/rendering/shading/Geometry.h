// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Geometry.h

#pragma once

#include <scene_rdl2/scene/rdl2/Geometry.h>

namespace moonray {
namespace shading {

/**
 * Extension to rdl2::Geometry
 */
class Geometry: public scene_rdl2::rdl2::SceneObject::Extension
{
public:
    explicit Geometry(const scene_rdl2::rdl2::SceneObject &owner);

    // The geometry label id used in material aov expressions
    void setGeomLabelId(int labelId);
    int getGeomLabelId() const { return mGeomLabelId; }

private:
    int mGeomLabelId;
};

} // namespace shading
} // namespace moonray

