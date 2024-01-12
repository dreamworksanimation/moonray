// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file Geometry.cc

#include "Geometry.h"

namespace moonray {
namespace shading {

Geometry::Geometry(const scene_rdl2::rdl2::SceneObject &owner):
    mGeomLabelId(-1)
{
}

void
Geometry::setGeomLabelId(int labelId)
{
    mGeomLabelId = labelId;
}

} // namespace shading
} // namespace moonray

