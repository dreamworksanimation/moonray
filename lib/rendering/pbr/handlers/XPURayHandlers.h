// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RayHandlers.h"
#include <moonray/rendering/rt/gpu/GPURay.h>
#include <tbb/spin_mutex.h>

namespace moonray {
namespace pbr {

// Unlocks the mutex when we're done with the GPU
void xpuOcclusionQueryBundleHandlerGPU(mcrt_common::ThreadLocalState *tls,
                                       unsigned numRays,
                                       BundledOcclRay *rays,
                                       const rt::GPURay *gpuRays,
                                       tbb::spin_mutex& mutex);

} // namespace pbr
} // namespace moonray
