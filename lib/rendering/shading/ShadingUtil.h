// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file ShadingUtil.h
/// $Id$
///

#pragma once

namespace scene_rdl2 {

namespace alloc { class Arena; }
}

namespace moonray {

namespace shading { template <typename T> class TypedAttributeKey; }

namespace shading {

class TLState;


scene_rdl2::alloc::Arena* getArena(shading::TLState *tls);

} // namespace shading
} // namespace moonray

