// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../GPURay.h"

#ifndef __APPLE__
#include <optix.h>
#endif

namespace moonray {
namespace rt {

// Global parameters passed to the GPU

struct MetalGPUParams
{
#ifdef __APPLE__
    // UMA device maps the CPU rays into the GPU - avoiding a copy
    uint64_t mCPURays;
    unsigned int mCPURayStride;
#else
    OptixTraversableHandle mAccel;
#endif

    // The input rays we are passing to the GPU for occlusion testing
    unsigned int mNumRays;
    moonray::rt::GPURay __gpu_device__* mRaysBuf;

    // The output results buffer
    unsigned char __gpu_device__* mIsOccludedBuf;
};

} // namespace rt
} // namespace moonray

