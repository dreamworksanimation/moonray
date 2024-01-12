// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "AdaptiveRegions.h"
#include <moonray/rendering/rndr/RenderDriver.h>

#include <moonray/rendering/mcrt_common/ProfileAccumulatorHandles.h>
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/shading/Material.h>

#include <algorithm>
#include <iomanip>
#include <limits>

namespace moonray {
namespace rndr {
constexpr int AdaptiveRegions::sDesiredNRegionsPerDimension;
constexpr int AdaptiveRegions::sMaxNRegions;
constexpr int AdaptiveRegions::sTileWidth;
constexpr int AdaptiveRegions::sTileHeight;
constexpr int AdaptiveRegions::sRegionOverlap;

namespace {
inline void doFlush(mcrt_common::ThreadLocalState* tls)
{
    MNRY_ASSERT(tls);
    auto driver = getRenderDriver();
    MNRY_ASSERT(driver);

    unsigned flushed;
    do {
        flushed = shading::Material::flushNonEmptyShadeQueue(tls);
        flushed += driver->flushXPUQueues(tls, &tls->mArena);
        flushed += tls->mPbrTls->flushLocalQueues();
    } while (flushed);
}
} // anonymous namespace

void AdaptiveRegions::init(scene_rdl2::math::BBox2i renderBounds, float targetError, bool vectorized)
{
    mVectorized = vectorized;
    mRegions.init(renderBounds);
    for (int i = 0; i < mRegions.getNumRegions(); ++i) {
        const scene_rdl2::math::BBox2i bounds = mRegions.getOverlappingRegionBounds(i);

        mTrees[i] = AdaptiveRegionTree(bounds, targetError);
        mRegionError[i] = std::numeric_limits<float>::max();
        mDone[i] = false;
        mRegionTileCount[i].value = mNumTiles[i] = tilesHorizontal(extents(bounds, 0)) *
                                                   tilesVertical(extents(bounds, 1));
        mAdaptiveTreeUpdateCounter[i] = 0;
    }
    disableAdjustUpdateTiming();
}

void AdaptiveRegions::update(const scene_rdl2::math::BBox2i& tile,
                             const scene_rdl2::fb_util::Tiler& tiler,
                             const scene_rdl2::fb_util::RenderBuffer& renderBuf,
                             const scene_rdl2::fb_util::FloatBuffer& numSamplesBuf,
                             const scene_rdl2::fb_util::RenderBuffer& renderBufOdd,
                             mcrt_common::ThreadLocalState* tls,
                             const unsigned endSampleIdx)
{
    // This function must be called for EVERY tile, even if the tile ends up doing no rendering work. If this isn't
    // adhered to, the load balancing will become off.

    // Our regions overlap to help reduce seams. getOverlappedRegions returns an array of (potentially redundant)
    // valid region indices for this tile, and adds a -1 at the end of valid indices. Instead of sorting the array
    // and removing redundant elements, we keep track of which regions we have already visited and skip those.
    //
    // Sorting a nine-element array, even with a hard-coded sorting network, and then running std::unique on it is
    // much less efficient than just going through the array once and keeping track of what we've seen.
    //
    // The overlapped regions do not incur more rendering costs, it simply expands are area of interest when
    // calculating error. When a tile is rendered, it is only checked against its primary region (getSampleArea).

    IndexArray indices;
    mRegions.getOverlappedRegions(tile, indices);

    VisitedArray visited = { false };
    MNRY_ASSERT(std::all_of(visited.cbegin(), visited.cend(), [](bool b) { return b == false; }));

    // Keep track of what region this thread was last in for queue flushing purposes.
    static thread_local int localRegionID = -1;
    for (int idx : indices) {
        MNRY_ASSERT(idx < sMaxNRegions);
        if (idx < 0) {
            break;
        } else if (visited[idx]) {
            continue;
        }
        visited[idx] = true;

        if (mNumTiles[idx] == 0) {
            // This can happen when, for example, we're doing debug pixels and we have a screen-space quad with no
            // area.
            continue;
        }

        // Decrement the number of tiles that have been rendered in this quad. If we're the last thread, we update
        // the AdaptiveRegionTree.
        if (--mRegionTileCount[idx].value == 0) {
            if (mVectorized) {
                doFlush(tls);
            }

            EXCL_ACCUMULATOR_PROFILE(tls, EXCL_EXCL_LOCK_ADAPTIVE_TREE);
            mMutexPool[idx].exclusiveLockAll();
            EXCL_ACCUMULATOR_PROFILE(tls, EXCL_BUILD_ADAPTIVE_TREE);

            // The memory ordering specifies how non-atomic memory is accessed around an atomic read or write.
            // Relaxed memory order specifies neither ordering nor synchronization, but we don't need to do that in
            // these cases, as the lock above ensures proper ordering and synchronization. If we put more
            // constrained memory ordering, we're simply paying the price twice.
            int qpc = mRegionTileCount[idx].value.load(std::memory_order_relaxed);
            //PRINT3(idx, qpc, mNumTiles[idx]);
            while (qpc <= 0) {
                // More threads may have come through between first read of mRegionTileCount and the lock, causing
                // the value to go negative, so we have to increment until positive to account for these possible
                // accesses. This means that fewer than mNumTiles[idx] are rendered by threads on the next pass,
                // which is a form of automatic load balancing.
                qpc += mNumTiles[idx];
            }
            mRegionTileCount[idx].value.store(qpc, std::memory_order_relaxed);

            if (mUpdateSentinel.shouldUpdate(endSampleIdx)) {
                // useful debug dump functions for input of adaptive tree update
                // saveRenderBufferByPPM(idx, tiler, renderBuf, false);
                // saveRenderBufferByPPM(idx, tiler, renderBufOdd, true);
                // saveFloatBufferByPPM(idx, tiler, numSamplesBuf);

                // We have to update adaptive tree
                const float error = mTrees[idx].update(tiler, renderBuf, numSamplesBuf, renderBufOdd);
                mRegionError[idx].store(error, std::memory_order_relaxed);
                mDone[idx].store(mTrees[idx].done(), std::memory_order_relaxed);

                // savePixelErrorsByPPM(idx); // useful debug dump all pixelError info to the disk as image
                ++mAdaptiveTreeUpdateCounter[idx];
            }
            mMutexPool[idx].unlockAll();
        } else if (mVectorized && idx != localRegionID) {
            // We don't want to flush too often, so we do it once per thread per adaptive region by keeping track of the
            // last region we were in. This falls down a bit if we restart the render (we may miss a flush).
            doFlush(tls);
#pragma warning push
#pragma warning disable 1711
            // warning #1711: assignment to statically allocated variable
            localRegionID = idx;
#pragma warning pop
        }
    }
}

void
AdaptiveRegions::updateAll(const scene_rdl2::fb_util::Tiler& tiler,
                           const scene_rdl2::fb_util::RenderBuffer& renderBuf,
                           const scene_rdl2::fb_util::FloatBuffer& numSamplesBuf,
                           const scene_rdl2::fb_util::RenderBuffer& renderBufOdd)
//
// This function is only called at initialization stage (i.e. before start MCRT) of resume render
// by single thread
//
{
    for (int idx = 0; idx < sMaxNRegions; ++idx) { // Can we do by MT?
        const float error = mTrees[idx].update(tiler, renderBuf, numSamplesBuf, renderBufOdd);
        mRegionError[idx].store(error, std::memory_order_relaxed);

        // savePixelErrorsByPPM(idx); // useful debug dump all pixelError info to the disk as image
        ++mAdaptiveTreeUpdateCounter[idx];
        mDone[idx].store(mTrees[idx].done(), std::memory_order_relaxed);
    }
}

void
AdaptiveRegions::savePixelErrorsByPPM(const int adaptiveRegionTreeId) const
//
// For debugging purpose function.
// Save pixelError information to the disk with some message.
//
{
    std::cerr << "AdaptiveRegions.cc savePixelErrorsByPPM() q:" << adaptiveRegionTreeId
              << " counter:" << mAdaptiveTreeUpdateCounter[adaptiveRegionTreeId] << std::endl;

    std::ostringstream ostr;
    ostr << "pixError_region" << adaptiveRegionTreeId << "_"
         << std::setw(2) << std::setfill('0')
         << mAdaptiveTreeUpdateCounter[adaptiveRegionTreeId] << ".ppm";
    std::string filename = ostr.str();
    if (mTrees[adaptiveRegionTreeId].savePixelErrorsByPPM(filename)) {
        std::cerr << "pixError PPM output filename:" << filename << " done" << std::endl;
    }
}

void
AdaptiveRegions::saveRenderBufferByPPM(const int adaptiveRegionTreeId,
                                       const scene_rdl2::fb_util::Tiler &tiler,
                                       const scene_rdl2::fb_util::RenderBuffer &renderBuf,
                                       const bool odd) const
{
    std::ostringstream ostr;
    ostr << "renderBuff_" << ((odd)? "odd": "all") << "_region" << adaptiveRegionTreeId << "_"
         << std::setw(2) << std::setfill('0')
         << mAdaptiveTreeUpdateCounter[adaptiveRegionTreeId] << ".ppm";
    std::string filename = ostr.str();
    if (mTrees[adaptiveRegionTreeId].saveRenderBufferByPPM(filename, tiler, renderBuf)) {
        std::cerr << "renderBuf PPM output filename:" << filename << " done" << std::endl;
    }
}

void
AdaptiveRegions::saveFloatBufferByPPM(const int adaptiveRegionTreeId,
                                      const scene_rdl2::fb_util::Tiler &tiler,
                                      const scene_rdl2::fb_util::FloatBuffer &floatBuf) const
{
    std::ostringstream ostr;
    ostr << "floatBuff_region" << adaptiveRegionTreeId << "_"
         << std::setw(2) << std::setfill('0')
         << mAdaptiveTreeUpdateCounter[adaptiveRegionTreeId] << ".ppm";
    std::string filename = ostr.str();
    if (mTrees[adaptiveRegionTreeId].saveFloatBufferByPPM(filename, tiler, floatBuf)) {
        std::cerr << "floatBuf PPM output filename:" << filename << " done" << std::endl;
    }
}

} // namespace rndr
} // namespace moonray

