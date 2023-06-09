// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace moonray {
namespace rt {

// The ray that is created in the XPU queue handler and sent to the GPU.
// Note that we can't use any of the CUDA float3 types here because that would
// pull CUDA headers into large parts of Moonray, and we can't use the regular
// Moonray math::Vec3 types because that would pull those headers into the GPU
// code.  So, we are limited to the built-in C++ types.  Luckily we are only
// passing a small amount of simple data.

struct GPUOcclusionRay
{
    float mOriginX, mOriginY, mOriginZ;
    float mDirX, mDirY, mDirZ;
    float mMinT;
    float mMaxT;
    float mTime;
    unsigned long long mLightId;
};

} // namespace rt
} // namespace moonray

