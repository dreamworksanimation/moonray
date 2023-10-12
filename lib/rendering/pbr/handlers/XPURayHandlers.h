// Copyright 2023 DreamWorks Animation LLC
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

// This is called when the GPU is busy and can't process any more rays.
// It just invokes the regular vector mode handler code.
// userData is params needed by embree when performing intersections on the CPU, see
// occlusionQueryBundleHandler()
void xpuOcclusionQueryBundleHandlerCPU(mcrt_common::ThreadLocalState *tls,
                                       unsigned numRays,
                                       const BundledOcclRay *rays,
                                       void *userData);

} // namespace pbr
} // namespace moonray

