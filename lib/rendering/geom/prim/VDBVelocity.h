// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file VDBVelocity.h
///
#pragma once

#include <moonray/rendering/geom/prim/GeomTLState.h>

#include <moonray/rendering/geom/Types.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/tools/VelocityFields.h>

namespace moonray {
namespace geom {
namespace internal {

// VDB Velocity fields are used to apply motion blur to volumes.

class VDBVelocity
{
public:
    VDBVelocity() : mVelocityGrid(nullptr),
                    mIsMotionBlurOn(false),
                    mTShutterOpen(0.f),
                    mTShutterRange(0.f)
    {}

    void setVelocityGrid(openvdb::VectorGrid::ConstPtr velocityGrid,
                         const std::vector<int>& volumeIds)
    {
        mVolumeIdCount = volumeIds.size();
        for (uint32_t samplerId = 0; samplerId < mVolumeIdCount; ++samplerId) {
            mVolumeIdToSamplerId[volumeIds[samplerId]] = samplerId;
        }
        mVelocityGrid = velocityGrid;
        unsigned samplerCount = mVolumeIdCount * mcrt_common::getNumTBBThreads();
        mVelocitySamplers.reserve(samplerCount);
        for (unsigned i = 0; i < samplerCount; ++i) {
            mVelocitySamplers.emplace_back(*mVelocityGrid);
        }
        mIsMotionBlurOn = true;
    }

    void setShutterValues(float tShutterOpen, float tShutterRange)
    {
        mTShutterOpen = tShutterOpen;
        mTShutterRange = tShutterRange;
    }

    void getShutterOpenAndClose(float& shutterOpen, float& shutterClose)
    {
        shutterOpen = mTShutterOpen;
        shutterClose = mTShutterOpen + mTShutterRange;
    }

    // return a vdb position for grid value access, the input position
    // may get advected by velocity in motion blur case
    openvdb::Vec3d getEvalPosition(mcrt_common::ThreadLocalState *tls,
                                   uint32_t volumeId,
                                   const scene_rdl2::math::Vec3f& pSample,
                                   float time) const
    {
        if (mIsMotionBlurOn) {
            tls->mGeomTls->mStatistics.incCounter(STATS_VELOCITY_GRID_SAMPLES);
            uint32_t threadIdx = tls->mThreadIdx;
            // Use backward advection to retrieve velocity v at time t,
            // and apply v on position p with backward advection again to
            // retrieve the final resolved position
            // For detailed reference see:
            // "Eulerian Motion Blur"
            // EGWNP2007 Doyub Kim and Hyeong-Seok Ko
            // equation (6) and (7)
            openvdb::Vec3d p(pSample[0], pSample[1], pSample[2]);
            unsigned samplerIdx = threadIdx * mVolumeIdCount + (mVolumeIdToSamplerId.at(volumeId));
            openvdb::Vec3s velocity = mVelocitySamplers[samplerIdx].sample(p);
            float dt = mTShutterOpen + time * mTShutterRange;
            velocity = mVelocitySamplers[samplerIdx].sample(p - velocity * dt);
            return p - velocity * dt;
        } else {
            return openvdb::Vec3d(pSample[0], pSample[1], pSample[2]);
        }
    }

    size_t getMemory()
    {
        size_t mem = 0;
        if (mVelocityGrid) {
            mem += mVelocityGrid->memUsage();
        }

        mem += scene_rdl2::util::getVectorElementsMemory(mVelocitySamplers);

        return mem;
    }

private:
    openvdb::VectorGrid::ConstPtr mVelocityGrid;
    std::vector<openvdb::tools::VelocitySampler<openvdb::VectorGrid>> mVelocitySamplers;
    bool mIsMotionBlurOn;
    float mTShutterOpen, mTShutterRange;
    std::unordered_map<uint32_t, uint32_t> mVolumeIdToSamplerId;
    unsigned mVolumeIdCount;
};

} // namespace internal
} // namespace geom
} // namespace moonray


