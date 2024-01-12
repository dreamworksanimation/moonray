// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VolumeSampleInfo.h
///

#pragma once

#include <scene_rdl2/scene/rdl2/VolumeShader.h>

namespace moonray {
namespace geom {
namespace internal {

/// This class provides information to sample field value for one particular
/// combination of volume/ray/primitive. It is filled out during volume BVH
/// traversal stage and stored in ThreadLocalStorage so that all the volume
/// sample/integration work for one particular ray can reuse the same info
class VolumeSampleInfo
{
public:
    void initialize(const scene_rdl2::rdl2::VolumeShader* volumeShader,
                    const Vec3f& sampleRayOrg, const Vec3f& sampleRayDir,
                    float featureSize, bool canCastShadow, bool isVDB) {
        mVolumeShader = volumeShader;
        mSampleRayOrg = sampleRayOrg;
        mSampleRayDir = sampleRayDir;
        mFeatureSize = featureSize;
        mCanCastShadow = canCastShadow;
        mIsVDB = isVDB;
    }

    const scene_rdl2::rdl2::VolumeShader* getShader() const
    {
        return mVolumeShader;
    }

    float getFeatureSize() const
    {
        return mFeatureSize;
    }

    unsigned int getProperties() const
    {
        return mVolumeShader->getProperties();
    }

    Vec3f getSamplePosition(float t) const
    {
        return mSampleRayOrg + t * mSampleRayDir;
    }

    // Is volume sample homogenous?
    bool isHomogenous() const
    {
        // VDB contains heterogeneous grid data
        if (mIsVDB) {
            return false;
        }

        return mVolumeShader->isHomogenous();
    }

    bool canCastShadow() const
    {
        return mCanCastShadow;
    }

private:
    const scene_rdl2::rdl2::VolumeShader* mVolumeShader;
    Vec3f mSampleRayOrg;
    Vec3f mSampleRayDir;
    float mFeatureSize;
    bool mCanCastShadow;
    bool mIsVDB;
};

} // namespace internal
} // namespace geom
} // namespace moonray

