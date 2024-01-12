// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file ShadowLinking.h
/// $Id$
///

#pragma once

#include <scene_rdl2/scene/rdl2/Light.h>
#include <unordered_set>

namespace moonray {
namespace geom {
namespace internal {

// A utility class for excluding some occlusions. The actual skipping of occlusions occurs in the
// Embree intersection filter skipOcclusionFilter().
// 
// The additional occlusion control it provides for the associated geometry primitive is:
//  - whether it will cast a shadow from specific lights
//  - whether it will cast a shadow onto specific receivers
// 
// This feature is provided purely for artistic control and is obviously not physically correct.

class ShadowLinking
{
public:
    ShadowLinking() {}
    ShadowLinking(bool complementReceiverSet) : mIsComplemented(complementReceiverSet) {}

    void reset()
    {
        mLightSet.clear();
        mReceiverSet.clear();
    }

    bool canCastShadow(const scene_rdl2::rdl2::Light* light) const
    {
        return mLightSet.find(light) == mLightSet.end();
    }

    void addLight(const scene_rdl2::rdl2::Light* light)
    {
        mLightSet.insert(light);
    }

    const std::unordered_set<const scene_rdl2::rdl2::Light*>& getLights() const
    {
        return mLightSet;
    }

    bool canReceiveShadow(int receiverId) const
    {
        if (mIsComplemented) {
            return mReceiverSet.find(receiverId) != mReceiverSet.end();
        } else {
            return mReceiverSet.find(receiverId) == mReceiverSet.end();
        }
    }

    void addReceiver(int receiverID)
    {
        mReceiverSet.insert(receiverID);
    }

    const std::unordered_set<int>& getReceivers() const
    {
        return mReceiverSet;
    }

    bool getIsComplemented() const
    {
        return mIsComplemented;
    }

private:
    std::unordered_set<const scene_rdl2::rdl2::Light*> mLightSet;
    std::unordered_set<int> mReceiverSet;
    bool mIsComplemented;
};

} // namespace internal
} // namespace geom
} // namespace moonray

