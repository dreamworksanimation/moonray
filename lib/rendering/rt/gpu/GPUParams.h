// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "GPUOcclusionRay.h"

#include <optix.h>

namespace moonray {
namespace rt {

// Global parameters passed to the GPU

struct GPUParams
{
    OptixTraversableHandle mAccel;

    // The input rays we are passing to the GPU for occlusion testing
    unsigned mNumRays;
    moonray::rt::GPUOcclusionRay* mRaysBuf;

    // The output results buffer
    unsigned char* mIsOccludedBuf;
};

} // namespace rt
} // namespace moonray

