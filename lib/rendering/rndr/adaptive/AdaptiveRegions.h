// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "ActivePixelMask.h"
#include "AdaptiveRegionTree.h"
#include "OverlappingRegions.h"
#include "UpdateSentinel.h"

#include <moonray/common/mcrt_util/MutexPool2D.h>
#include <moonray/rendering/mcrt_common/ProfileAccumulatorHandles.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/common/fb_util/FbTypes.h>
#include <scene_rdl2/common/fb_util/Tiler.h>
#include <scene_rdl2/render/util/AtomicFloat.h>
#include <scene_rdl2/render/util/ReaderWriterMutex.h>

#include <array>
#include <atomic>

namespace moonray {
namespace rndr {

// The AdaptiveRegionTree has no notion of threading. All threading considerations are pushed to this class.
// The AdaptiveRegions, as the name may imply, contains multiple AdaptiveRegionTrees for portions of screen space.
//
// On a high level, this allows one thread to update the AdaptiveRegionTree for a region while the other threads are off
// rendering other sections of the image. The AdaptiveRegions keeps track of how many tiles have been rendered in
// an area. Once all of an area's tiles have been visited, the last thread updates the AdaptiveRegionTree. This scheme
// works best when rendering with the Morton curve tile order, so that other threads aren't waiting on the quad to be
// updated and continue rendering.
class AdaptiveRegions
{
    // When rendering with a Morton curve, 2x2 regions gives us mostly
    // contention-free updates without a lot of overhead, until we near the end
    // of rendering and we spend more time updating than rendering.  This is
    // now greater than that for better load-balancing, and if the tile order
    // is something like image-center out, having smaller regions may allow
    // inner regions to be updated while the outer regions continue to render.
    // It may also make sense to increase this value when doing vectorized
    // rendering, where frame buffer results aren't as deterministic.
    static constexpr int sDesiredNRegionsPerDimension = 4;
    static constexpr int sMaxNRegions                 = sDesiredNRegionsPerDimension*sDesiredNRegionsPerDimension;
    static constexpr int sTileWidth                   = COARSE_TILE_SIZE;
    static constexpr int sTileHeight                  = COARSE_TILE_SIZE;
    static constexpr int sRegionOverlap               = COARSE_TILE_SIZE;

    using Regions      = OverlappingRegions<sRegionOverlap, sTileWidth, sTileHeight, sDesiredNRegionsPerDimension>;
    using MutexType    = scene_rdl2::util::ReaderWriterMutex;
    using ReadLock     = scene_rdl2::util::ReadLock;
    using IndexArray   = typename Regions::IndexArray;

    static int tilesHorizontal(int width)
    {
        return roundUpDivision(width, sTileWidth);
    }

    static int tilesVertical(int height)
    {
        return roundUpDivision(height, sTileHeight);
    }

public:
    AdaptiveRegions() = default;

    using VisitedArray = std::array<bool, sMaxNRegions>;

    void init(scene_rdl2::math::BBox2i renderBounds, float targetError, bool vectorized);

    inline void disableAdjustUpdateTiming();
    inline void enableAdjustUpdateTiming(const std::vector<unsigned> &adaptiveIterationPixSampleIdTbl);

    void update(const scene_rdl2::math::BBox2i& tile,
                const scene_rdl2::fb_util::Tiler& tiler,
                const scene_rdl2::fb_util::RenderBuffer& renderBuf,
                const scene_rdl2::fb_util::FloatBuffer& numSamplesBuf,
                const scene_rdl2::fb_util::RenderBuffer& renderBufOdd,
                mcrt_common::ThreadLocalState* tls,
                const unsigned endSampleIdx);
    void updateAll(const scene_rdl2::fb_util::Tiler& tiler,
                   const scene_rdl2::fb_util::RenderBuffer& renderBuf,
                   const scene_rdl2::fb_util::FloatBuffer& numSamplesBuf,
                   const scene_rdl2::fb_util::RenderBuffer& renderBufOdd);

    ActivePixelMask getSampleArea(const scene_rdl2::math::BBox2i& tile, mcrt_common::ThreadLocalState* tls) const;
    float getError() const;
    bool done() const;

private:
    // Align the ints on cache line sizes to avoid false sharing.
    struct alignas(CACHE_LINE_SIZE) RegionTileCount
    {
        std::atomic<int> value;
    };

    // This value can be pretty arbitrary. We'll use 2^sLogMutexCount mutexes. The larger this is, the less contention
    // on the reads, but the more overhead on the writes.
    static constexpr int sLogMutexCount = 8;
    using MutexPool = MutexPool2D<sLogMutexCount, MutexType>;

    bool mVectorized;
    Regions mRegions;
    RegionTileCount mRegionTileCount[sMaxNRegions];
    std::atomic<float> mRegionError[sMaxNRegions];      // This is cached on update so we don't have to lock.
    std::atomic<bool> mDone[sMaxNRegions];              // This is cached on update so we don't have to lock.
    mutable MutexPool mMutexPool[sMaxNRegions];
    AdaptiveRegionTree mTrees[sMaxNRegions];
    int mNumTiles[sMaxNRegions];

    UpdateSentinel mUpdateSentinel; // adjust adaptiveTreeUpdate timing logic related code

    unsigned mAdaptiveTreeUpdateCounter[sMaxNRegions]; // for debug purpose. count adaptive tree update is very useful

    void savePixelErrorsByPPM(const int adaptiveRegionTreeId) const; // for debug
    void saveRenderBufferByPPM(const int adaptiveRegionTreeId,
                               const scene_rdl2::fb_util::Tiler &tiler,
                               const scene_rdl2::fb_util::RenderBuffer &renderBuf, const bool odd) const; // for debug
    void saveFloatBufferByPPM(const int adaptiveRegionTreeId,
                              const scene_rdl2::fb_util::Tiler &tiler,
                              const scene_rdl2::fb_util::FloatBuffer &floatBuf) const; // for debug
};

inline void
AdaptiveRegions::disableAdjustUpdateTiming()
{
    mUpdateSentinel.disableAdjustUpdateTiming();
}

inline void
AdaptiveRegions::enableAdjustUpdateTiming(const std::vector<unsigned> &adaptiveIterationPixSampleIdTbl)
{
    mUpdateSentinel.enableAdjustUpdateTiming(adaptiveIterationPixSampleIdTbl);
}

inline ActivePixelMask AdaptiveRegions::getSampleArea(const scene_rdl2::math::BBox2i& tile, mcrt_common::ThreadLocalState* tls) const
{
    EXCL_ACCUMULATOR_PROFILE(tls, EXCL_QUERY_ADAPTIVE_TREE);
    const int idx = mRegions.getRegionIndex(tile);

    // It's possible that another thread is updating the tree, so we need to lock on our read.
    // Divide by the tile dimensions so that neighboring tiles access the mutex pool with different indices.
    ReadLock lock(mMutexPool[idx].getMutex(tile.lower[0]/sTileWidth, tile.lower[1]/sTileHeight));
    return mTrees[idx].getSampleArea(tile);
}

inline float AdaptiveRegions::getError() const
{
    float error = 0.0f;
    for (auto& e : mRegionError) {
        error = scene_rdl2::math::max(e.load(std::memory_order_relaxed), error);
    }
    std::atomic_thread_fence(std::memory_order_release);
    return error;
}

inline bool AdaptiveRegions::done() const
{
    bool done = true;
    for (auto& e : mDone) {
        if (e.load(std::memory_order_relaxed) == false) {
            done = false;
            break;
        }
    }
    std::atomic_thread_fence(std::memory_order_release);
    return done;
}

} // namespace rndr
} // namespace moonray

