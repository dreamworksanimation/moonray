// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file Shading.h
/// $Id$
///

#pragma once

#include <moonray/rendering/shading/EvalAttribute.h>

#include <moonray/rendering/bvh/shading/Log.h>
#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/shading/ispc/EvalAttribute_ispc_stubs.h>

namespace moonray {
namespace shading {

typedef std::pair<const scene_rdl2::rdl2::Light *, double> LightContrib;
typedef std::vector<LightContrib> LightContribArray;

enum WrapType {
    Black = 0,  // Black outside [0..1]
    Clamp,      // Clamp outside [0..1]
    Periodic,   // Periodic mode 1
    Mirror,     // Mirror the image
    Default     // Use the default found in the file
};

} // end namespace shading
} // end namespace moonray

