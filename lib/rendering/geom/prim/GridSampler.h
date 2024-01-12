// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file GridSampler.h
///

#pragma once

#include "GeomTLState.h"

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/common/math/Color.h>

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/tools/Interpolation.h>

namespace moonray {
namespace geom {
namespace internal {

class GridSamplerEvalData
{
public:
    GridSamplerEvalData(pbr::TLState *pbrTls,
                        const uint32_t volumeId,
                        const Vec3f& p) :
        mPbrTls(pbrTls),
        mVolumeId(volumeId),
        mP(p) {}

    pbr::TLState *mPbrTls;
    const uint32_t mVolumeId;
    const Vec3f& mP;
};

enum class Interpolation
{
    POINT,
    BOX,
    QUADRATIC
};

template<typename GridType>
struct GridSampler
{
    typedef typename GridType::ValueType ValueT;
    typedef typename GridType::ConstAccessor ConstAccessor;
    typedef openvdb::tools::GridSampler<ConstAccessor, openvdb::tools::PointSampler> PointSampler;
    typedef openvdb::tools::GridSampler<ConstAccessor, openvdb::tools::BoxSampler> BoxSampler;
    typedef openvdb::tools::GridSampler<ConstAccessor, openvdb::tools::QuadraticSampler> QuadraticSampler;

    GridSampler():
        mAccessor(nullptr),
        mPointSampler(nullptr),
        mBoxSampler(nullptr),
        mQuadraticSampler(nullptr)
    {
    }

    GridSampler(const GridType& grid):
        mAccessor(grid.getConstAccessor()),
        mPointSampler(mAccessor, grid.transform()),
        mBoxSampler(mAccessor, grid.transform()),
        mQuadraticSampler(mAccessor, grid.transform())
    {
    }

    ValueT evalPoint(const openvdb::Vec3d& p) const
    {
        return mPointSampler.wsSample(p);
    }

    ValueT evalBox(const openvdb::Vec3d& p) const
    {
        return mBoxSampler.wsSample(p);
    }

    ValueT evalQuadratic(const openvdb::Vec3d& p) const
    {
        return mQuadraticSampler.wsSample(p);
    }

    ConstAccessor mAccessor;
    PointSampler mPointSampler;
    BoxSampler mBoxSampler;
    QuadraticSampler mQuadraticSampler;
};

template<typename GridType>
struct VDBSampler
{
    typedef typename GridType::ValueType ValueT;
    typedef typename GridType::ConstPtr GridConstPtr;

    VDBSampler(): mGrid(nullptr), mIsValid(false), mVolumeIdCount(0u)
    {
    }

    void initialize(const GridConstPtr& grid,
                    const std::vector<int>& volumeIds,
                    StatCounters statsCounterType)
    {
        mGrid = grid;
        mIsValid = (mGrid != nullptr) && !mGrid->empty();
        if (mIsValid) {
            mVolumeIdCount = volumeIds.size();
            for (uint32_t samplerId = 0; samplerId < mVolumeIdCount; ++samplerId) {
                mVolumeIdToSamplerId[volumeIds[samplerId]] = samplerId;
            }
            unsigned mSamplerCount = mVolumeIdCount * mcrt_common::getNumTBBThreads();
            mSamplers.reserve(mSamplerCount);
            for (size_t i = 0; i < mSamplerCount; ++i) {
                mSamplers.emplace_back(*mGrid);
            }
        }

        mStatsCounterType = statsCounterType;
    }

    ValueT eval(mcrt_common::ThreadLocalState* tls,
                uint32_t volumeId,
                const openvdb::Vec3d& p,
                Interpolation mode) const
    {
        if (mIsValid) {
            tls->mGeomTls->mStatistics.incCounter(mStatsCounterType);
            uint32_t threadIdx = tls->mThreadIdx;
            unsigned samplerIdx = threadIdx * mVolumeIdCount + (mVolumeIdToSamplerId.at(volumeId));
            switch (mode) {
            case Interpolation::POINT:
                return mSamplers[samplerIdx].evalPoint(p);
                break;
            case Interpolation::BOX:
                return mSamplers[samplerIdx].evalBox(p);
                break;
            case Interpolation::QUADRATIC:
                return mSamplers[samplerIdx].evalQuadratic(p);
                break;
            default:
                return mSamplers[samplerIdx].evalPoint(p);
                break;
            }
        }

        return ValueT(0.0f);
    }

    size_t getMemory() const
    {
        // We do not need to call mGrid->memUsage() because the grid itself
        // is shared with other members of the Primitive class, and we
        // count those members instead.

        return scene_rdl2::util::getVectorElementsMemory(mSamplers);
    }

    GridConstPtr mGrid;
    bool mIsValid;
    std::vector<GridSampler<GridType>> mSamplers;
    std::unordered_map<uint32_t, uint32_t> mVolumeIdToSamplerId;
    unsigned mVolumeIdCount;
    StatCounters mStatsCounterType;
};

} // namespace internal
} // namespace geom
} // namespace moonray

