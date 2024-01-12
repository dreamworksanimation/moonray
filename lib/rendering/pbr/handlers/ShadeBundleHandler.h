// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <moonray/rendering/mcrt_common/Bundle.h>

namespace moonray {

namespace mcrt_common { class ThreadLocalState; }
namespace shading { struct SortedRayState; }

namespace pbr {

// This function is responsible generating differential geometry, evalating
// shade points and routing BSDFs through to the integrator.
void shadeBundleHandler(mcrt_common::ThreadLocalState *tls, unsigned numEntries,
                        shading::SortedRayState *entries, void *userData);

} // namespace pbr
} // namespace moonray


