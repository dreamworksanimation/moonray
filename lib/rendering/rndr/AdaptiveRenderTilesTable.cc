// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "AdaptiveRenderTilesTable.h"
#include "Film.h"

#include <scene_rdl2/common/fb_util/FbTypes.h>
#include <scene_rdl2/common/math/Viewport.h>

#include <sstream>

// Debug dump for resetTileInfo() for re-compute initial adaptive tile condition for resume render
// Debug pixel condition is setup at inside AdaptiveRenderTilesTable::setDebugPosition()
//#define DEBUG_RESETTILEINFO

namespace moonray {
namespace rndr {

unsigned
AdaptiveRenderTilesTable::reset(const Film &film,
                                const std::vector<scene_rdl2::fb_util::Tile> &tiles,
                                const unsigned minSamplesPerPixel,
                                const unsigned maxSamplesPerPixel,
                                const float targetAdaptiveError,
                                const scene_rdl2::math::Viewport &viewport)
//
// return start pixel sample id
//
{
    reset();

    setDebugPosition(tiles); // setup debug pixel position at reset operation for resume render

    unsigned startPixSampleId = maxSamplesPerPixel;
    for (size_t tileId = 0; tileId < tiles.size(); ++tileId) {
        const scene_rdl2::fb_util::Tile &currTile = tiles[tileId];

        unsigned currStartPixSampleId = resetTileInfo(film, tileId, currTile,
                                                      minSamplesPerPixel,
                                                      maxSamplesPerPixel,
                                                      targetAdaptiveError,
                                                      viewport);
        if (currStartPixSampleId < startPixSampleId) startPixSampleId = currStartPixSampleId;
    }

    return startPixSampleId;
}

std::string    
AdaptiveRenderTilesTable::show(const std::string &hd, const std::vector<scene_rdl2::fb_util::Tile> &tiles) const
{
    std::ostringstream ostr;
    ostr << hd << "AdaptiveRenderTilesTable {\n";
    ostr << hd << "  mWidth:" << mWidth << " mHeight:" << mHeight << '\n';
    ostr << hd << "  mMaxSamplesPerPixel:" << mMaxSamplesPerPixel << '\n';
    ostr << showTilesTable(hd + "  ", tiles) << '\n';
    ostr << hd << "}";
    return ostr.str();
}

std::string
AdaptiveRenderTilesTable::showTileInfo(const std::string &hd,
                                       const Film &film, const std::vector<scene_rdl2::fb_util::Tile> &tiles,
                                       const unsigned tileIdx) const
{
    const scene_rdl2::fb_util::Tile &currTile = tiles[tileIdx];

    std::ostringstream ostr;
    ostr << hd << "TileInfo (tileId:" << tileIdx << ") {\n";
    ostr << showTileInfo(hd + "  ", film, currTile) << '\n';
    ostr << mTiles[tileIdx].show(hd + "  ") << '\n';
    ostr << hd << "}";
    return ostr.str();
}

void
AdaptiveRenderTilesTable::setDebugPosition(const std::vector<scene_rdl2::fb_util::Tile> &tiles)
{
    // For activate debugPosition for non resume render, you should check comment about
    // setDebugPosition() at RenderFrame.cc RenderDriver::renderFrame() as well.
    
    mDebug = false;
    /*
    mDebug = true;
    mDebugPixX = 420;
    mDebugPixY = 51;
    */

    if (!mDebug) return;

    std::cerr << ">> AdaptiveRenderTilesTable.cc setDebugPosition()"
              << " debugPix(" << mDebugPixX << ',' << mDebugPixY << ')' << std::endl;

    for (size_t tileId = 0; tileId < tiles.size(); ++tileId) {
        const scene_rdl2::fb_util::Tile &currTile = tiles[tileId];
        if (isDebugTile(currTile)) {
            mDebugTileId = tileId;
            int offX = mDebugPixX - currTile.mMinX;
            int offY = mDebugPixY - currTile.mMinY;
            mDebugPixId = offY * 8 + offX;
            return;             // properly setup all debug info
        }
    }

    mDebug = false; // Could not find active tile -> (mDebugPixX, mDebugPixY) might be outside viewport
}

void
AdaptiveRenderTilesTable::debugTileInfoDump(const Film &film,
                                            const std::vector<scene_rdl2::fb_util::Tile> &tiles,
                                            const unsigned tileIdx,
                                            const std::string &msg) const
{
    if (isDebugTile(tiles[tileIdx])) {
        std::cerr << ">> AdaptiveRenderTilesTable.cc >" << msg << '<'
                  << " debug(Pix(" << mDebugPixX << ',' << mDebugPixY << ')'
                  << " PixId:" << mDebugPixId << ") "
                  << showTileInfo("", film, tiles, tileIdx) << std::endl;
    }
}

//-------------------------------------------------------------------------------------------------------------

unsigned
AdaptiveRenderTilesTable::resetTileInfo(const Film &film,
                                        const size_t tileId,
                                        const scene_rdl2::fb_util::Tile &tile,
                                        const unsigned minSamplesPerPixel,
                                        const unsigned maxSamplesPerPixel,
                                        const float targetAdaptiveError,
                                        const scene_rdl2::math::Viewport &viewport)
//
// return minimum pixel samples for this tile
//
{
    AdaptiveRenderTileInfo &tileInfo = getTile(tileId);
    tileInfo.reset();
    mNumTilesAtMinimum = 0;

#   ifdef DEBUG_RESETTILEINFO
    bool debug = isDebugTile(tile);
    if (debug) {
        std::cerr << ">> AdaptiveRenderTilesTable.cc resetTileInfo() start {\n"
                  << showTileInfo("  ", film, tile) << std::endl;
        std::cerr << "  maxSmplPerPix:" << maxSamplesPerPixel << " minSamplPerPix:" << minSamplesPerPixel
                  << std::endl;
    }
#   endif // end DEBUG_RESETTILEINFO

    //
    // Adaptive stage tileInfo check (see also RenderDriver::updateTileCondition())
    //
    unsigned currTotalTileSamples; // Total samples for this tile
    unsigned currMinPixSamples = findTileMinPixSamples(film, tile,
                                                       maxSamplesPerPixel, currTotalTileSamples);

    if (currMinPixSamples >= maxSamplesPerPixel) {
        // All pixels already have more than max samples -> done this tile
        setTileCompleteAdaptiveStage(tileId);
#       ifdef DEBUG_RESETTILEINFO
        if (debug) {
            std::cerr << "  ADAPTIVE_STAGE cMinPixSmpl:" << currMinPixSamples << " > maxSmplPerPix:"
                      << maxSamplesPerPixel << '\n'
                      << tileInfo.show("  result ") << '\n'
                      << "} resetTileInfo() done" << std::endl;
        }
#       endif // end DEBUG_RESETTILEINFO
        return maxSamplesPerPixel;
    }

    if (currMinPixSamples < minSamplesPerPixel) {
        // We have to do the uniform stage because some of the pixel is below minSamplesPerPixel
        setTileUniform(tileId, currTotalTileSamples);
        ++mNumTilesAtMinimum;
#       ifdef DEBUG_RESETTILEINFO
        if (debug) {
            std::cerr << "  UNIFORM_STAGE cMinPixSmpl:" << currMinPixSamples << " < minSmplPerPix:"
                      << minSamplesPerPixel << '\n'
                      << tileInfo.show("  result ") << '\n'
                      << "} resetTileInfo() done" << std::endl;
        }
#       endif // end DEBUG_RESETTILEINFO
        return currMinPixSamples;
    }

    // If we're not complete, and we're not uniform, we're adaptive.
    setTileAdaptive(tileId, currTotalTileSamples);
#   ifdef DEBUG_RESETTILEINFO
    if (debug) {
        std::cerr << "  ADAPTIVE_STAGE: "
                  << tileInfo.show("  result ") << '\n'
                  << "} resetTileInfo() done" << std::endl;
    }
#   endif // end DEBUG_RESETTILEINFO
    return currMinPixSamples;
}

template <typename F>
void
AdaptiveRenderTilesTable::crawlTilePix(const scene_rdl2::fb_util::Tile &tile, F &&pixFunc) const
{
    for (unsigned py = tile.mMinY; py < tile.mMaxY; ++py) {
        for (unsigned px = tile.mMinX; px < tile.mMaxX; ++px) {
            pixFunc(px, py);
        } // px
    } // py
}

unsigned
AdaptiveRenderTilesTable::findTileMinPixSamples(const Film &film, const scene_rdl2::fb_util::Tile &tile,
                                                const unsigned maxPixSamples,
                                                unsigned &totalTileSamples) const
{
    unsigned minPixSamples = maxPixSamples;
    totalTileSamples = 0;
    crawlTilePix(tile, [&](unsigned px, unsigned py) {
            unsigned currPixSamples = film.getNumRenderBufferPixelSamples(px, py);
            if (currPixSamples < minPixSamples) minPixSamples = currPixSamples;
            totalTileSamples += (currPixSamples > maxPixSamples)? maxPixSamples: currPixSamples;
        });
    return minPixSamples;  // return minimum pixel samples inside tile
}

template <typename F>
void
AdaptiveRenderTilesTable::showTilePix(std::ostringstream &ostr, const std::string &hd, const int itemWidth,
                                      const scene_rdl2::fb_util::Tile &tile, F &&pixShowFunc) const
{
    ostr << hd << " -Y-\n";

    crawlTilePix(tile, [&](unsigned px, unsigned py) {
            unsigned yLocalMax = tile.mMaxY - tile.mMinY - 1;
            unsigned fy = yLocalMax - (py - tile.mMinY) + tile.mMinY; // flip Y inside tile.
            if (px == tile.mMinX) ostr << hd << std::setw(4) << fy << "  " << fy - tile.mMinY << " |";
            pixShowFunc(ostr, px, fy);
            if (px == tile.mMaxX - 1) ostr << '\n';
        });

    ostr << (hd + "        +") << std::setw(8 * (itemWidth + 1) - 1) << std::setfill('-') << '-'
         << std::setfill(' ') << "\n";
    for (int i = 0; i < 8; ++i) {
        ostr << ((i == 0)? (hd + "         "): "") << std::setw(itemWidth) << i
             << ((i != 7)? ' ': '\n');
    }
    for (int i = 0; i < 8; ++i) {
        ostr << ((i == 0)? (hd + "         "): "") << std::setw(itemWidth) << i + tile.mMinX
             << ((i != 7)? " ": "  -X-\n");
    }
}

std::string
AdaptiveRenderTilesTable::showTileInfo(const std::string &hd,
                                       const Film &film, const scene_rdl2::fb_util::Tile &tile) const
{
    std::ostringstream ostr;
    ostr << hd << "tile (" << tile.mMinX << ',' << tile.mMinY << ")-(" << tile.mMaxX << ',' << tile.mMaxY
         << ") {\n"; {
        ostr << hd << "  pixel samples {\n";
        showTilePix(ostr, hd + "    ", 11, tile,
                    [&](std::ostringstream &ostr, unsigned px, unsigned py) {
                        ostr << std::dec << std::setw(11) << (unsigned)film.getWeight(px, py) << ' ';
                    });
        ostr << hd << "  }\n";
    }
    ostr << hd << "}";
    return ostr.str();
}

std::string
AdaptiveRenderTilesTable::showTilesTable(const std::string &hd, const std::vector<scene_rdl2::fb_util::Tile> &tiles) const
{
    unsigned totalUniform = 0;
    unsigned totalAdaptive = 0;
    unsigned totalCompleted = 0;

    std::ostringstream ostr;
    ostr << hd << "mTiles {\n";
    crawlAllTilesTopBottomLeftRight
        (tiles,
         [&](unsigned tileXId, unsigned tileYId,
             const AdaptiveRenderTileInfo &currTileInfo,
             const scene_rdl2::fb_util::Tile &currTile) {
            const unsigned tileTotalX = detail::tilesInDimension(mWidth);
            if ((tileXId % tileTotalX) == 0) ostr << (hd + "  ");

            switch(currTileInfo.getCondition()) {
            case AdaptiveRenderTileInfo::Stage::UNIFORM_STAGE:  ostr << "U"; totalUniform++; break;
            case AdaptiveRenderTileInfo::Stage::ADAPTIVE_STAGE: ostr << "A"; totalAdaptive++; break;
            case AdaptiveRenderTileInfo::Stage::COMPLETED:      ostr << "C"; totalCompleted++; break;
            }
            ostr << ' ';
            if (((tileXId + 1) % tileTotalX) == 0) ostr << '\n';
        });
    ostr << hd << "  totalUniform:" << totalUniform << '\n'
         << hd << "  totalAdaptive:" << totalAdaptive << '\n'
         << hd << "  totalCompleted:" << totalCompleted << '\n';
    ostr << hd << "}";
    return ostr.str();
}

bool
AdaptiveRenderTilesTable::isDebugTile(const scene_rdl2::fb_util::Tile &tile) const
{
    return (mDebug &&
            tile.mMinX <= mDebugPixX && mDebugPixX < tile.mMaxX &&
            tile.mMinY <= mDebugPixY && mDebugPixY < tile.mMaxY);
}

} // namespace rndr
} // namespace moonray

