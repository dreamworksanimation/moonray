// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../GPURay.h"

#include <optix.h>

namespace moonray {
namespace rt {

// Global parameters passed to the GPU

struct OptixGPUParams
{
    OptixTraversableHandle mAccel;

    // The input rays we are passing to the GPU
    unsigned mNumRays;
    moonray::rt::GPURay* mRaysBuf;

    // One of these output buffers will be nullptr -
    //  we can distinguish intersect() vs occluded() in the CUDA code
    GPURayIsect* mIsectBuf;        // The output results buffer for intersect()
    unsigned char* mIsOccludedBuf; // The output results buffer for occluded()
};

} // namespace rt
} // namespace moonray
