// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <moonray/rendering/shading/ispc/OpenVdbUtil_ispc_stubs.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <string>

namespace moonray {
namespace shading {

bool isOpenVdbGeometry(const scene_rdl2::rdl2::Geometry* geom);

scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::String> getModelAttributeKey(const scene_rdl2::rdl2::Geometry* geom);

} // namespace shading
} // namespace moonray

