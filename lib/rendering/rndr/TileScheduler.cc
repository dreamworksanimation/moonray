// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
// - A tile is the finest granularity we can split up rendering within a single pass.
// - Tiles are the perfect format for sending data to a merge node since all the pixels
//   for a single tile are contigous in memory. The message where tiles are sent
//   to a merge note should contain a list of tightly packed tiles. This implies
//   that detiling should be done on the merge node (eventually).
// - No explicit cropping is neccessary since this is handled in the extrapolation passes.
//
#include "TileScheduler.h"
#include "Film.h"
#include <scene_rdl2/render/util/Arena.h>
#include <scene_rdl2/render/util/Random.h>
#include <random>

namespace moonray {
namespace rndr {

namespace {

void
distributeTiles(std::vector<scene_rdl2::fb_util::Tile>& tiles, unsigned renderNodeIdx,
                unsigned numRenderNodes)
{
    const unsigned numTiles = unsigned(tiles.size());
    std::vector<scene_rdl2::fb_util::Tile> temp;
    temp.reserve(numTiles / numRenderNodes + 1u);

    for (unsigned i = renderNodeIdx; i < numTiles; i += numRenderNodes) {
        temp.push_back(tiles[i]);
    }

    tiles.swap(temp);
}

}   // End of anon namespace.

//-----------------------------------------------------------------------------

TileScheduler::TileScheduler(Type type) :
    mRenderNodeIdx(0),
    mNumRenderNodes(1),
    mType(type),
    mTaskDistribType(0)
{
    MNRY_ASSERT(mType < NUM_TILE_SCHEDULER_TYPES);
    mTileIndices.reset();
}

unsigned
TileScheduler::generateTiles(scene_rdl2::alloc::Arena *arena,
                             unsigned width,
                             unsigned height,
                             const scene_rdl2::math::Viewport &viewport,
                             unsigned renderNodeIdx,
                             unsigned numRenderNodes,
                             unsigned taskDistribType)
{
    MNRY_ASSERT(viewport.mMinX <= viewport.mMaxX && viewport.mMinY <= viewport.mMaxY);
    MNRY_ASSERT(viewport.mMaxX < (int)width && viewport.mMaxY < (int)height);
    MNRY_ASSERT(renderNodeIdx < numRenderNodes);

    mRenderNodeIdx = renderNodeIdx;
    mNumRenderNodes = numRenderNodes;

    mTiles.clear();

    SCOPED_MEM(arena);

    unsigned numTilesX = ((viewport.mMaxX) >> 3) - ((viewport.mMinX) >> 3) + 1;
    unsigned numTilesY = ((viewport.mMaxY) >> 3) - ((viewport.mMinY) >> 3) + 1;
    unsigned numTiles = numTilesX * numTilesY;
    MNRY_ASSERT(numTiles);

    mTileIndices.reset(new uint32_t[numTiles]);

    // under multi-machine context, each mcrt computation uses different random seed
    generateTileIndices(arena, numTilesX, numTilesY, mTileIndices.get(), uint32_t(mRenderNodeIdx));

#ifdef DEBUG

    // Validate data:
    uint32_t *tileIndicesCopy = arena->allocArray<uint32_t>(numTiles, CACHE_LINE_SIZE);
    memcpy(tileIndicesCopy, mTileIndices.get(), numTiles * sizeof(uint32_t));
    std::sort(tileIndicesCopy, tileIndicesCopy + numTiles);
    for (unsigned i = 0; i < numTiles; ++i) {
        MNRY_ASSERT(tileIndicesCopy[i] == i);
    }

#endif

    // Generate actual tiles from the tileIndices.
    mTiles.resize(numTiles);

    unsigned baseX = viewport.mMinX & ~0x07;
    unsigned baseY = viewport.mMinY & ~0x07;

    for (unsigned i = 0; i < numTiles; ++i) {

        unsigned tileIdx = mTileIndices[i];
        scene_rdl2::fb_util::Tile &tile = mTiles[tileIdx];

        uint32_t x = i % numTilesX;
        uint32_t y = i / numTilesX;

        tile.mMinX = std::max((x << 3) + baseX, (unsigned)viewport.mMinX);
        tile.mMaxX = std::min((x << 3) + baseX + 8, (unsigned)viewport.mMaxX + 1);

        tile.mMinY = std::max((y << 3) + baseY, (unsigned)viewport.mMinY);
        tile.mMaxY = std::min((y << 3) + baseY + 8, (unsigned)viewport.mMaxY + 1);

        MNRY_ASSERT(tile.mMinX < (unsigned)viewport.mMaxX + 1);
        MNRY_ASSERT(tile.mMaxX > (unsigned)viewport.mMinX);
        MNRY_ASSERT(tile.mMinY < (unsigned)viewport.mMaxY + 1);
        MNRY_ASSERT(tile.mMaxY > (unsigned)viewport.mMinY);
    }

    //
    // Handle distributed rendering by getting each machine to only render
    // a subset of the generated tiles.
    //
    if (numRenderNodes > 1) {
        mTaskDistribType = taskDistribType;
        if (mTaskDistribType == static_cast<unsigned>(scene_rdl2::rdl2::TaskDistributionType::NON_OVERLAPPED_TILE)) {
            // unorverlapped tile distribution
            distributeTiles(mTiles, renderNodeIdx, numRenderNodes);
        }
    }

    return unsigned(mTiles.size());
}

std::unique_ptr<TileScheduler>
TileScheduler::create(TileScheduler::Type type)
{
    std::unique_ptr<TileScheduler> tileScheduler;

    switch (type) {
    case TOP:              tileScheduler.reset(new TopTileScheduler);             break;
    case BOTTOM:           tileScheduler.reset(new BottomTileScheduler);          break;
    case LEFT:             tileScheduler.reset(new LeftTileScheduler);            break;
    case RIGHT:            tileScheduler.reset(new RightTileScheduler);           break;
    case MORTON:           tileScheduler.reset(new MortonTileScheduler);          break;
    case RANDOM:           tileScheduler.reset(new RandomTileScheduler);          break;
    case SPIRAL_SQUARE:    tileScheduler.reset(new SpiralSquareTileScheduler);    break;
    case SPIRAL_RECT:      tileScheduler.reset(new SpiralRectTileScheduler);      break;
    case MORTON_SHIFTFLIP: tileScheduler.reset(new MortonShiftFlipTileScheduler); break;
    default:
        MNRY_ASSERT(0);
    }

    return tileScheduler;
}

//-----------------------------------------------------------------------------

void
TopTileScheduler::generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                      unsigned numTilesX, unsigned numTilesY,
                                      uint32_t *tileIndices, uint32_t seed) const
{
    BottomTileScheduler scheduler;
    scheduler.generateTileIndices(arena, numTilesX, numTilesY, tileIndices, seed);
    std::reverse(tileIndices, tileIndices + (numTilesX * numTilesY));
}

void
BottomTileScheduler::generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                         unsigned numTilesX, unsigned numTilesY,
                                         uint32_t *tileIndices, uint32_t seed) const
{
    unsigned tileIdx = 0;
    for (unsigned y = 0; y < numTilesY; ++y) {
        if (y & 1) {
            // Odd rows go right to left:
            for (int x = numTilesX - 1; x >= 0; --x, ++tileIdx) {
                tileIndices[y * numTilesX + unsigned(x)] = tileIdx;
            }
        } else {
            // Even rows go left to right:
            for (unsigned x = 0; x < numTilesX; ++x, ++tileIdx) {
                tileIndices[y * numTilesX + x] = tileIdx;
            }
        }
    }

    MNRY_ASSERT(tileIdx == numTilesX * numTilesY);
}

void
LeftTileScheduler::generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                       unsigned numTilesX, unsigned numTilesY,
                                       uint32_t *tileIndices, uint32_t seed) const
{
    unsigned tileIdx = 0;
    for (unsigned x = 0; x < numTilesX; ++x) {
        if (x & 1) {
            for (int y = numTilesY - 1; y >= 0; --y, ++tileIdx) {
                tileIndices[unsigned(y) * numTilesX + x] = tileIdx;
            }
        } else {
            for (unsigned y = 0; y < numTilesY; ++y, ++tileIdx) {
                tileIndices[y * numTilesX + x] = tileIdx;
            }
        }
    }

    MNRY_ASSERT(tileIdx == numTilesX * numTilesY);
}

void
RightTileScheduler::generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                        unsigned numTilesX, unsigned numTilesY,
                                        uint32_t *tileIndices, uint32_t seed) const
{
    LeftTileScheduler scheduler;
    scheduler.generateTileIndices(arena, numTilesX, numTilesY, tileIndices, seed);
    std::reverse(tileIndices, tileIndices + (numTilesX * numTilesY));
}

void
mortonTileOrderGen(scene_rdl2::alloc::Arena* arena,
                   unsigned numTilesX,
                   unsigned numTilesY,
                   uint32_t* tileIndices,
                   unsigned shiftX, // tile count
                   unsigned shiftY, // tile count
                   bool flipX,
                   bool flipY)
{
    unsigned totalTiles = numTilesX * numTilesY;

    unsigned currShiftX = shiftX % numTilesX;
    unsigned currShiftY = shiftY % numTilesY;

    unsigned pow2TilesX = scene_rdl2::util::roundUpToPowerOfTwo(numTilesX);
    unsigned pow2TilesY = scene_rdl2::util::roundUpToPowerOfTwo(numTilesY);
    unsigned totalPow2Tiles = pow2TilesX * pow2TilesY;

    SCOPED_MEM(arena);

    unsigned *swizzleRemap = arena->allocArray<unsigned>(totalPow2Tiles);
    memset(swizzleRemap, 0xff, sizeof(unsigned) * totalPow2Tiles);

    unsigned tilesFound = 0;
    for (unsigned tileY = 0; tileY < pow2TilesY; ++tileY) {
        unsigned y;
        if (!flipY) {
            if (tileY >= numTilesY) continue;
            y = (tileY + numTilesY - currShiftY) % numTilesY;
        } else {
            if (tileY < (pow2TilesY - numTilesY)) continue;
            y = ((pow2TilesY - 1 - tileY) + currShiftY) % numTilesY;
        }

        for (unsigned tileX = 0; tileX < pow2TilesX; ++tileX) {
            unsigned x;
            if (!flipX) {
                if (tileX >= numTilesX) continue;
                x = (tileX + numTilesX - currShiftX) % numTilesX;
            } else {
                if (tileX < (pow2TilesX - numTilesX)) continue;
                x = ((pow2TilesX - 1 - tileX) + currShiftX) % numTilesX;
            }
            
            swizzleRemap[scene_rdl2::util::convertCoordToSwizzledIndex(x,
                                                                       y,
                                                                       pow2TilesX,
                                                                       pow2TilesY)] = tilesFound++;
        }
    }

    unsigned tilesPlaced = 0;
    for (unsigned i = 0; i < totalPow2Tiles; ++i) {
        if (swizzleRemap[i] != unsigned(-1)) {
            MNRY_ASSERT(swizzleRemap[i] < totalTiles);
            tileIndices[swizzleRemap[i]] = tilesPlaced++;
            if (tilesPlaced == totalTiles) {
                break;
            }
        }
    }
}

void
MortonTileScheduler::generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                         unsigned numTilesX, unsigned numTilesY,
                                         uint32_t *tileIndices, uint32_t seed) const
//
// Standard Morton tile scheduler
//
{
    mortonTileOrderGen(arena, numTilesX, numTilesY, tileIndices, 0, 0, false, false);
}

void
MortonShiftFlipTileScheduler::generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                                  unsigned numTilesX, unsigned numTilesY,
                                                  uint32_t *tileIndices, uint32_t seed) const
//
// Added flip and tile shift operations to the standard Morton tile scheduler
//
{
    mortonTileOrderGen(arena, numTilesX, numTilesY, tileIndices, mShiftX, mShiftY, mFlipX, mFlipY);
}

void
RandomTileScheduler::generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                         unsigned numTilesX, unsigned numTilesY,
                                         uint32_t *tileIndices, uint32_t seed) const
{
    unsigned totalTiles = numTilesX * numTilesY;

    for (unsigned i = 0; i < totalTiles; ++i) {
        tileIndices[i] = i;
    }

    // Important, take the seed into account so results are deterministic.
    scene_rdl2::util::Random rng(seed);
    std::shuffle(tileIndices, tileIndices + totalTiles, rng);
}

void
SpiralSquareTileScheduler::generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                               unsigned numTilesX, unsigned numTilesY,
                                               uint32_t *tileIndices, uint32_t seed) const
{
    unsigned maxSide = std::max(numTilesX, numTilesY);
    unsigned maxSideSquared = maxSide * maxSide;
    unsigned numTiles = numTilesX * numTilesY;

    SCOPED_MEM(arena);

    unsigned *tempIndices = arena->allocArray<unsigned>(maxSideSquared);

    // Generate a square tiling pattern which surrounds our desired viewport.
    SpiralRectTileScheduler scheduler;
    scheduler.generateTileIndices(arena, maxSide, maxSide, tempIndices, seed);

    // Trivial case, the viewport is square.
    if (numTilesX == numTilesY) {
        MNRY_ASSERT(maxSideSquared == numTiles);
        memcpy(tileIndices, tempIndices, sizeof(unsigned) * numTiles);
        return;
    }

    //
    // Extract out the tiles we're interested in.
    //
    unsigned minX, maxX, minY, maxY;

    if (numTilesX > numTilesY) {
        minX = 0;
        maxX = numTilesX;
        minY = (maxSide - numTilesY) / 2;
        maxY = minY + numTilesY;
    } else {
        minX = (maxSide - numTilesX) / 2;
        maxX = minX + numTilesX;
        minY = 0;
        maxY = numTilesY;
    }

    struct UnsortedTile
    {
        unsigned mCounter;  // Incrementing counter for later index compaction.
        unsigned mOldIndex;
        unsigned mNewIndex;
    };

    UnsortedTile *tiles = arena->allocArray<UnsortedTile>(numTiles);

    unsigned dstIdx = 0;
    for (unsigned tileY = 0; tileY < maxSide; ++tileY) {
        if (tileY >= minY && tileY < maxY) {
            for (unsigned tileX = 0; tileX < maxSide; ++tileX) {
                if (tileX >= minX && tileX < maxX) {
                    tiles[dstIdx].mCounter = dstIdx;
                    tiles[dstIdx].mOldIndex = tempIndices[tileY * maxSide + tileX];
                    ++dstIdx;
                }
            }
        }
    }

    MNRY_ASSERT(dstIdx == numTiles);

    //
    // Compact indices.
    //

    // Sort so that old indices are in ascending order.
    std::sort(tiles, tiles + numTiles, [](const UnsortedTile &a, const UnsortedTile &b) -> bool
    {
        return a.mOldIndex < b.mOldIndex;
    });

    // Patch in new indices.
    for (unsigned i = 0; i < numTiles; ++i) {
        tiles[i].mNewIndex = i;
    }

    // Revert to original ordering.
    std::sort(tiles, tiles + numTiles, [](const UnsortedTile &a, const UnsortedTile &b) -> bool
    {
        return a.mCounter < b.mCounter;
    });

    // Copy indices out to dst array.
    for (unsigned i = 0; i < numTiles; ++i) {
        tileIndices[i] = tiles[i].mNewIndex;
    }
}

void
SpiralRectTileScheduler::generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                             unsigned numTilesX, unsigned numTilesY,
                                             uint32_t *tileIndices, uint32_t seed) const
{
    enum
    {
        LEFT,
        DOWN,
        RIGHT,
        UP,
        NUM_DIRS,
    };

    // Start at an out corner of the screen and work inwards. We'll reverse the ordering
    // later in this function.
    unsigned numTiles = numTilesX * numTilesY;
    unsigned currX = numTilesX - 1;
    unsigned currY = numTilesY - 1;
    unsigned dir = (numTilesX >= numTilesY) ? LEFT : DOWN;

    unsigned currTileIdx = 0;

    // Mark all tiles as need to be filled.
    memset(tileIndices, 0xff, sizeof(uint32_t) * numTiles);

    while (1) {

        unsigned dstIdx = numTilesX * currY + currX;
        MNRY_ASSERT(dstIdx < numTiles);

        if (tileIndices[dstIdx] == 0xffffffff) {
            // Here is where we reverse the ordering so we're starting from the inside
            // as opposed to the outside.
            tileIndices[dstIdx] = numTiles - currTileIdx - 1;
            if (++currTileIdx == numTiles) {
                break;
            }
        }

        //
        // Figure out where we should place next tile.
        //
        switch (dir) {
        case LEFT:
            if (currX == 0 ||
                tileIndices[numTilesX * currY + currX - 1] != 0xffffffff) {
                dir = DOWN;
            } else {
                --currX;
            }
            break;

        case DOWN:
            if (currY == 0 ||
                tileIndices[numTilesX * (currY - 1) + currX] != 0xffffffff) {
                dir = RIGHT;
            } else {
                --currY;
            }
            break;

        case RIGHT:
            if (currX == numTilesX - 1 ||
                tileIndices[numTilesX * currY + currX + 1] != 0xffffffff) {
                dir = UP;
            } else {
                ++currX;
            }
            break;

        case UP:
            if (currY == numTilesY - 1 ||
                tileIndices[numTilesX * (currY + 1) + currX] != 0xffffffff) {
                dir = LEFT;
            } else {
                ++currY;
            }
            break;

        default:
            MNRY_ASSERT(0);
        }
    }
}

//-----------------------------------------------------------------------------

} // namespace rndr
} // namespace moonray


