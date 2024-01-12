// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file ShadingUtil.cc
/// $Id$
///

#include "ShadingUtil.h"

#include <moonray/rendering/bvh/shading/AttributeKey.h>
#include <moonray/rendering/bvh/shading/ShadingTLState.h>
#include <scene_rdl2/render/util/Arena.h>

namespace moonray {
namespace shading {

scene_rdl2::alloc::Arena*
getArena(shading::TLState *tls) {
    return tls->mArena;
}

} // namespace shading
} // namespace moonray

