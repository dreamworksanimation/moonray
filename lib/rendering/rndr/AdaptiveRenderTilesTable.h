// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

#include "AdaptiveRenderTileInfo.h"

#include <scene_rdl2/common/platform/Platform.h> // finline
#include <scene_rdl2/render/util/AtomicFloat.h>
#include <atomic>
#include <vector>

// This directive is used for runtime debug dump message output if adaptive tile condition is changing
//#define DEBUG_MSG_TILE_CONDITION_UPDATE

namespace scene_rdl2 {

namespace math {
class Viewport;
}
namespace fb_util {
class Tile;
}
}

namespace moonray {

namespace rndr {

namespace detail
{
    constexpr unsigned tileAlignment(unsigned value)
    {
        return (value + 7) & ~7;
    }

    constexpr unsigned tilesInDimension(unsigned length)
    {
        return tileAlignment(length) / 8;
    }
}

class Film;

class AdaptiveRenderTilesTable
{
public:
    AdaptiveRenderTilesTable(const unsigned width, const unsigned height, const unsigned maxSamplesPerPixel) :
        mWidth(width),
        mHeight(height),
        mMaxSamplesPerPixel(maxSamplesPerPixel),
        mNumTilesAtMinimum(detail::tilesInDimension(width) * detail::tilesInDimension(height)),
        mTargetError(1.0f),
        mTiles(detail::tilesInDimension(width) * detail::tilesInDimension(height)),
        mCompletedSamples(0),
        mCurrentError(std::numeric_limits<float>::max()),
        mDebug(false),
        mDebugPixX(0),
        mDebugPixY(0),
        mDebugTileId(0),
        mDebugPixId(0)
    {}

    finline void reset();
    unsigned reset(const Film &film,
                   const std::vector<scene_rdl2::fb_util::Tile> &tiles,
                   const unsigned minSamplesPerPixel,
                   const unsigned maxSamplesPerPixel,
                   const float targetAdaptiveError,
                   const scene_rdl2::math::Viewport &viewport); // for reset resume render : return tile start sampleId

    finline void setTargetError(const float error) { mTargetError = error; }
    finline void setCurrentError(const float error) { mCurrentError = error; }
    finline void setTileUniform(const unsigned tileIdx, const unsigned addedSamples);
    finline void setTileAdaptive(const unsigned tileIdx, const unsigned addedSamples);
    finline void setTileUpdate(const unsigned tileIdx, const unsigned addedSamples);
    finline void setTileCompleteAdaptiveStage(const unsigned tileIdx);
    finline bool getMinimumDone() const noexcept;

    AdaptiveRenderTileInfo &getTile(const unsigned tileIdx) { return mTiles[tileIdx]; }

    finline float getCompletedFraction(bool activeRendering,
                                       size_t *completedSamples = nullptr,
                                       size_t *total = nullptr) const;

    // useful debug functions
    std::string show(const std::string &hd, const std::vector<scene_rdl2::fb_util::Tile> &tiles) const;
    std::string showTileInfo(const std::string &hd,
                             const Film &film,
                             const std::vector<scene_rdl2::fb_util::Tile> &tiles, const unsigned tileIdx) const;

    void setDebugPosition(const std::vector<scene_rdl2::fb_util::Tile> &tiles);
    void debugTileInfoDump(const Film &film, const std::vector<scene_rdl2::fb_util::Tile> &tiles, const unsigned tileIdx,
                           const std::string &msg) const;
    finline bool isDebugTile(const unsigned tileIdx) const { return mDebug && tileIdx == mDebugTileId; }
    finline bool isDebugPixById(const unsigned tileIdx, const unsigned pixId) const;
    finline bool isDebugPixByPos(const unsigned px, const unsigned py) const;
    finline bool getDebugPix(unsigned &px, unsigned &py) const;

private:
    finline float getAdaptiveCompletedFraction() const;
    finline float getUniformCompletedFraction() const;

    unsigned resetTileInfo(const Film &film,
                           const size_t tileId,
                           const scene_rdl2::fb_util::Tile &tile,
                           const unsigned minSamplesPerPixel,
                           const unsigned maxSamplesPerPixel,
                           const float targetAdaptiveError,
                           const scene_rdl2::math::Viewport &viewport); // return minimum pixel samples for this tile

    template <typename F> void crawlTilePix(const scene_rdl2::fb_util::Tile &tile, F &&pixFunc) const;
    unsigned findTileMinPixSamples(const Film &film, const scene_rdl2::fb_util::Tile &tile,
                                   const unsigned maxPixSamples, unsigned &totalTileSamples) const;

    template <typename F> void showTilePix(std::ostringstream &ostr, const std::string &hd,
                                           const int itemWidth,
                                           const scene_rdl2::fb_util::Tile &tile, F &&pixShowFunc) const; // for debug dump
    std::string showTileInfo(const std::string &hd,
                             const Film &film, const scene_rdl2::fb_util::Tile &tile) const; // debug dump function

    template <typename F> void crawlAllTilesTopBottomLeftRight(const std::vector<scene_rdl2::fb_util::Tile> &tiles,
                                                               F &&tileFunc) const;
    std::string showTilesTable(const std::string &hd, const std::vector<scene_rdl2::fb_util::Tile> &tiles) const;

    bool isDebugTile(const scene_rdl2::fb_util::Tile &tile) const;

    unsigned mWidth, mHeight;
    unsigned mMaxSamplesPerPixel;
    std::atomic<unsigned> mNumTilesAtMinimum;
    float mTargetError;

    std::vector<AdaptiveRenderTileInfo> mTiles;

    // For large images with large sample counts, it's possible that a 32-bit int overflows.
    std::atomic<std::uint64_t> mCompletedSamples;

    // statistical information
    std::atomic<float> mCurrentError;

    // debug pixel and tile information
    bool mDebug;
    unsigned mDebugPixX, mDebugPixY;
    unsigned mDebugTileId, mDebugPixId;
};

finline void
AdaptiveRenderTilesTable::reset()
{
    for (size_t tileId = 0; tileId < mTiles.size(); ++tileId) {
        mTiles[tileId].reset();
    }
    mCompletedSamples = 0;

    mDebug = false;
    mDebugPixX = 0;
    mDebugPixY = 0;
    mDebugTileId = 0;
    mDebugPixId = 0;

    mCurrentError = std::numeric_limits<float>::max();
}

finline void
AdaptiveRenderTilesTable::setTileUniform(const unsigned tileIdx, const unsigned addedSamples)
{
    mTiles[tileIdx].setCondition(AdaptiveRenderTileInfo::Stage::UNIFORM_STAGE);
    setTileUpdate(tileIdx, addedSamples);
}

finline void
AdaptiveRenderTilesTable::setTileAdaptive(const unsigned tileIdx, const unsigned addedSamples)
{
    if (mTiles[tileIdx].getCondition() == AdaptiveRenderTileInfo::Stage::UNIFORM_STAGE) {
        --mNumTilesAtMinimum;
    }
    mTiles[tileIdx].setCondition(AdaptiveRenderTileInfo::Stage::ADAPTIVE_STAGE);
    setTileUpdate(tileIdx, addedSamples);
}

finline void
AdaptiveRenderTilesTable::setTileUpdate(const unsigned tileIdx, const unsigned addedSamples)
{
    mTiles[tileIdx].update(addedSamples);
    mCompletedSamples += addedSamples;
}

finline void
AdaptiveRenderTilesTable::setTileCompleteAdaptiveStage(const unsigned tileIdx)
{
    if (mTiles[tileIdx].getCondition() == AdaptiveRenderTileInfo::Stage::UNIFORM_STAGE) {
        --mNumTilesAtMinimum;
    }
    mCompletedSamples += mTiles[tileIdx].complete(mMaxSamplesPerPixel);
}

finline bool
AdaptiveRenderTilesTable::getMinimumDone() const noexcept
{
    return mNumTilesAtMinimum == 0;
}

finline float
AdaptiveRenderTilesTable::getAdaptiveCompletedFraction() const
{
    const float r = mTargetError/mCurrentError;
    // We assume that it will take four times as many samples to halve the error.
    return r*r;
}

finline float
AdaptiveRenderTilesTable::getUniformCompletedFraction() const
{
    const unsigned totalMax = mMaxSamplesPerPixel * mWidth * mHeight;
    return static_cast<float>(mCompletedSamples)/static_cast<float>(totalMax);
}

finline float
AdaptiveRenderTilesTable::getCompletedFraction(bool activeRendering,
                                               size_t *completedSamples,
                                               size_t *total) const
{
    // Look at both adaptive and linear time frames, because we may finish based on the max sample count.
    const float adaptiveFrac = getAdaptiveCompletedFraction();
    const float linearFrac = getUniformCompletedFraction();

    if (completedSamples) *completedSamples = mCompletedSamples;
    if (total) *total = mMaxSamplesPerPixel * mWidth * mHeight;

    // It looks weird when it says 100% but it's still rendering.
    float maxFraction = (activeRendering) ? 0.999f : 1.0f;
    return std::min(maxFraction, std::max(adaptiveFrac, linearFrac));
}

finline bool    
AdaptiveRenderTilesTable::isDebugPixById(const unsigned tileIdx, const unsigned pixId) const
{
    return mDebug && tileIdx == mDebugTileId && pixId == mDebugPixId;
}

finline bool
AdaptiveRenderTilesTable::isDebugPixByPos(const unsigned px, const unsigned py) const
{
    return mDebug && mDebugPixX == px && mDebugPixY == py;
}

finline bool
AdaptiveRenderTilesTable::getDebugPix(unsigned &px, unsigned &py) const
{
    if (!mDebug) return false;
    px = mDebugPixX;
    py = mDebugPixY;
    return true;
}

//---------------------------------------------------------------------------------------------------------------

template <typename F>
void
AdaptiveRenderTilesTable::crawlAllTilesTopBottomLeftRight(const std::vector<scene_rdl2::fb_util::Tile> &tiles,
                                                          F &&tileFunc) const
{
    //
    // We want to access all tiles by top to bottom and each line by left to right order.
    // Tile position detail is saved inside tiles and this tile order (i.e. tileId) is generated
    // depend on the tile scheduler type (TileScheduler::Type).
    // We can not easily convert from tile position to tileId unfortunately.
    // In order to do this, we creates sorted TileId (=tile position id) converted to tileId convertion table.
    // Sorted tileId start from left down tile and increased for x direction first. After reached most right
    // tile, back to the most left tile with increate +1 for y direction until all the tile is done.
    // Using this sortedTileId to tileId table, we can easily access all tiles by position based order.
    //
    unsigned tileXTotal = 0;
    unsigned tileYTotal = 0;
    for (const scene_rdl2::fb_util::Tile &tile : tiles) {
        tileXTotal = scene_rdl2::math::max(tileXTotal, tile.mMinX / 8 + 1);
        tileYTotal = scene_rdl2::math::max(tileYTotal, tile.mMinY / 8 + 1);
    }
    unsigned tileTotal = tileXTotal * tileYTotal;

    std::vector<unsigned> sorted2OrigTbl(tileTotal);
    for (size_t tileId = 0; tileId < tiles.size(); ++tileId) {
        const scene_rdl2::fb_util::Tile &currTile = tiles[tileId];
        size_t sortedTileId = (currTile.mMinY / 8) * tileXTotal + (currTile.mMinX / 8);
        sorted2OrigTbl[sortedTileId] = tileId;
    }

    for (int tileYId = (int)tileYTotal - 1; tileYId > 0; --tileYId) {
        for (unsigned tileXId = 0; tileXId < tileXTotal; ++tileXId) {
            unsigned sortedTileId = (unsigned)tileYId * tileXTotal + tileXId;
            unsigned origTileId = sorted2OrigTbl[sortedTileId];
            const scene_rdl2::fb_util::Tile &currTile = tiles[origTileId];
            const AdaptiveRenderTileInfo &currTileInfo = mTiles[origTileId];
            tileFunc(tileXId, tileYId, currTileInfo, currTile);
        }
    }
}

} // namespace rndr
} // namespace moonray

