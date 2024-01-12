// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VolumeTransition.h
///

#pragma once

namespace moonray {
namespace geom {
namespace internal {

/// VolumeTransition marks a transition event that ray "enters" or "leaves"
/// a particular volume region. During volume integration stage, we update
/// VolumeRegions along the ray based on the series of VolumeTransition
/// ray encountered
struct VolumeTransition
{
    VolumeTransition() {}

    VolumeTransition(const Primitive* primitive, float tIntersect, int volumeId, bool isEntry, float *tRenderSpace = nullptr) :
        mT(tIntersect), mVolumeId(volumeId), mIsEntry(isEntry), mIsExit(!isEntry), mPrimitive(primitive)
    {
        if (tRenderSpace) {
            mTRenderSpace[0] = tRenderSpace[0];
            mTRenderSpace[1] = tRenderSpace[1];
        } else {
            mTRenderSpace[0] = -1.0f;
            mTRenderSpace[1] = -1.0f;
        }
    }

    float mT; // danger: This MUST be the first member or these won't sort properly! (scene_rdl2::util::smartSort32)
    int mVolumeId;
    bool mIsEntry;
    bool mIsExit;
    const Primitive* mPrimitive;

    // Render space distance [0]=t0, [1]=t1.
    // Only valid when volume transmittance ray toward to the light which is inside ShadowSet
    // light list with mEnter=true condition. Otherwise, this value is {-1.0, -1.0}
    // This value is used to detect origin volumeId for ShadowSet control.
    float mTRenderSpace[2];

    finline bool isEntry() const { return mIsEntry; }
    finline bool isExit()  const { return mIsExit;  }
};

inline bool operator==(const VolumeTransition& a, const VolumeTransition& b)
{
    return a.mPrimitive == b.mPrimitive &&
           a.mT        == b.mT        &&
           a.mVolumeId == b.mVolumeId &&
           a.mIsEntry  == b.mIsEntry  &&
           a.mIsExit   == b.mIsExit;
}

} // namespace internal
} // namespace geom
} // namespace moonray

