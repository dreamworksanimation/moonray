// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef __METAL__
#include <stdint.h>
#endif

namespace moonray {
namespace rt {

// The ray that is created in the XPU queue handler and sent to the GPU.
// Note that we can't use any of the CUDA float3 types here because that would
// pull CUDA headers into large parts of Moonray, and we can't use the regular
// Moonray math::Vec3 types because that would pull those headers into the GPU
// code.  So, we are limited to the built-in C++ types.

struct GPURay
{
#ifndef __APPLE__
    // Apple uses UMA so it gets most of the data for the GPURay directly from
    // the queued BundledOcclRays that are in UMA memory.  Thus, the GPURay only
    // has a few members that aren't present on the BundledOcclRay.
    float mOriginX, mOriginY, mOriginZ;
    float mDirX, mDirY, mDirZ;
    float mMinT;
    float mMaxT;
    float mTime;
#endif
    int mMask;
    int mShadowReceiverId;
    uint64_t mLightId;
};

// Used for intersect() queries but not occluded()
struct GPURayIsect
{
    float mTFar; // intersection distance
    float mNgX, mNgY, mNgZ; // geometry normal
    float mU, mV; // barycentric coords

    unsigned int mEmbreeGeomID;
    unsigned int mPrimID;
    intptr_t mEmbreeUserData;

    unsigned int mInstance0IdOrLight;
    unsigned int mInstance1Id;
    unsigned int mInstance2Id;
    unsigned int mInstance3Id;
    float mL2R[4][3]; // layout matches Mat43
};

} // namespace rt
} // namespace moonray
