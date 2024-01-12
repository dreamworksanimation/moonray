// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once
#include <scene_rdl2/common/fb_util/FbTypes.h>
#include <scene_rdl2/scene/rdl2/SceneVariables.h>

#include <memory>
#include <vector>

namespace scene_rdl2 {
namespace alloc { class Arena; }
}

namespace moonray {

namespace rndr {

//-----------------------------------------------------------------------------

class TileScheduler
{
public:
    enum Type
    {
        TOP,                // 0
        BOTTOM,             // 1
        LEFT,               // 2
        RIGHT,              // 3
        MORTON,             // 4
        RANDOM,             // 5
        SPIRAL_SQUARE,      // 6
        SPIRAL_RECT,        // 7
        MORTON_SHIFTFLIP,   // 8
        NUM_TILE_SCHEDULER_TYPES,
    };

    explicit TileScheduler(Type type);
    virtual ~TileScheduler() = default;

    Type getType() const    { return mType; }

    // The width and height parameters should be the unaligned original render
    // buffer extents. Viewport is either the full extents of the buffer or a
    // sub-viewport.
    // The renderNodeIdx and numRenderNodes parameters only need to be set for
    // distributed rendering. Returns the number of tiles generated.
    unsigned generateTiles(scene_rdl2::alloc::Arena *arena,
                           unsigned width,
                           unsigned height,
                           const scene_rdl2::math::Viewport &viewport,
                           unsigned renderNodeIdx = 0,
                           unsigned numRenderNodes = 1,
                           unsigned taskDistribType = 0);

    // Returns true if tiles are split up amongst rendering nodes.
    bool isDistributed() const  { return mNumRenderNodes > 1; }
    unsigned taskDistribType() const { return mTaskDistribType; } // Film::TaskDistribType

    unsigned getRenderNodeIdx() const { return mRenderNodeIdx; }

    // Returns cached minimal set of tiles for viewport passed in.
    const std::vector<scene_rdl2::fb_util::Tile> &getTiles() const   { return mTiles; }

    // Returns the permuted order of the tile indices
    const uint32_t* getTileIndices() const { return mTileIndices.get(); }

    // Factory function.
    static std::unique_ptr<TileScheduler> create(TileScheduler::Type type);

protected:
    // User must fill out (numTilesX * numTilesY) linear tile indices.
    // Index zero is assumed to be the bottom left of the grid.
    // Each cell in the grid should be filled out with the order you want
    // that tile rendered in. So for example, if you put 15 in the bottom left
    // cell, you are saying that you want it to be the 15th tile rendered.
    virtual void generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                     unsigned numTilesX,
                                     unsigned numTilesY,
                                     uint32_t *tileIndices,
                                     uint32_t seed) const = 0;

    unsigned    mRenderNodeIdx;
    unsigned    mNumRenderNodes;
    Type        mType;

    std::vector<scene_rdl2::fb_util::Tile>   mTiles;
    std::unique_ptr<uint32_t[]> mTileIndices;

    unsigned mTaskDistribType;  // Film::TaskDistribType
};

//-----------------------------------------------------------------------------

class TopTileScheduler : public TileScheduler
{
public:
    TopTileScheduler() : TileScheduler(TileScheduler::TOP) {}
    virtual void generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                     unsigned numTilesX,
                                     unsigned numTilesY,
                                     uint32_t *tileIndices,
                                     uint32_t seed) const override;
};

class BottomTileScheduler : public TileScheduler
{
public:
    BottomTileScheduler() : TileScheduler(TileScheduler::BOTTOM) {}
    virtual void generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                     unsigned numTilesX,
                                     unsigned numTilesY,
                                     uint32_t *tileIndices,
                                     uint32_t seed) const override;
};

class LeftTileScheduler : public TileScheduler
{
public:
    LeftTileScheduler() : TileScheduler(TileScheduler::LEFT) {}
    virtual void generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                     unsigned numTilesX,
                                     unsigned numTilesY,
                                     uint32_t *tileIndices,
                                     uint32_t seed) const override;
};

class RightTileScheduler : public TileScheduler
{
public:
    RightTileScheduler() : TileScheduler(TileScheduler::RIGHT) {}
    virtual void generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                     unsigned numTilesX,
                                     unsigned numTilesY,
                                     uint32_t *tileIndices,
                                     uint32_t seed) const override;
};

class MortonTileScheduler : public TileScheduler
{
public:
    MortonTileScheduler() : TileScheduler(TileScheduler::MORTON) {}
    virtual void generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                     unsigned numTilesX,
                                     unsigned numTilesY,
                                     uint32_t *tileIndices,
                                     uint32_t seed) const override;
};

class RandomTileScheduler : public TileScheduler
{
public:
    RandomTileScheduler() : TileScheduler(TileScheduler::RANDOM) {}
    virtual void generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                     unsigned numTilesX,
                                     unsigned numTilesY,
                                     uint32_t *tileIndices,
                                     uint32_t seed) const override;
};

class SpiralSquareTileScheduler : public TileScheduler
{
public:
    SpiralSquareTileScheduler() : TileScheduler(TileScheduler::SPIRAL_SQUARE) {}
    virtual void generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                     unsigned numTilesX,
                                     unsigned numTilesY,
                                     uint32_t *tileIndices,
                                     uint32_t seed) const override;
};


class SpiralRectTileScheduler : public TileScheduler
{
public:
    SpiralRectTileScheduler() : TileScheduler(TileScheduler::SPIRAL_RECT) {}
    virtual void generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                     unsigned numTilesX,
                                     unsigned numTilesY,
                                     uint32_t *tileIndices,
                                     uint32_t seed) const override;
};

class MortonShiftFlipTileScheduler : public TileScheduler
{
public:
    MortonShiftFlipTileScheduler() : TileScheduler(TileScheduler::MORTON_SHIFTFLIP) {}
    virtual void generateTileIndices(scene_rdl2::alloc::Arena *arena,
                                     unsigned numTilesX,
                                     unsigned numTilesY,
                                     uint32_t *tileIndices,
                                     uint32_t seed) const override;

    void set(unsigned shiftX, // tile count
             unsigned shiftY, // tile count
             bool flipX,
             bool flipY)
    {
        mShiftX = shiftX;
        mShiftY = shiftY;
        mFlipX = flipX;
        mFlipY = flipY;
    };

private:
    unsigned mShiftX {0}; // tile count
    unsigned mShiftY {0}; // tile count
    bool mFlipX {false};
    bool mFlipY {false};
};

//-----------------------------------------------------------------------------

} // namespace rndr
} // namespace moonray


