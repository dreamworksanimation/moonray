// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RayHandlers.h"
#include <moonray/rendering/rt/gpu/GPURay.h>
#include <atomic>

namespace moonray {
namespace pbr {

// Unlocks the mutex when we're done with the GPU
void xpuRayBundleHandler(mcrt_common::ThreadLocalState *tls,
                         unsigned numRayStates,
                         RayState **rayStates,
                         const rt::GPURay *gpuRays,
                         std::atomic<int>& threadsUsingGPU);

// Unlocks the mutex when we're done with the GPU
void xpuOcclusionQueryBundleHandler(mcrt_common::ThreadLocalState *tls,
                                    unsigned numRays,
                                    BundledOcclRay *rays,
                                    const rt::GPURay *gpuRays,
                                    std::atomic<int>& threadsUsingGPU);

} // namespace pbr
} // namespace moonray
