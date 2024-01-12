// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/fb_util/FbTypes.h>
#include <scene_rdl2/common/fb_util/Tiler.h>
#include <scene_rdl2/common/math/BBox.h>

#include <algorithm>
#include <array>
#include <limits>

namespace moonray {
namespace rndr {
constexpr int roundUpDivision(int x, int y) noexcept
{
    return 1 + ((x - 1) / y);
}

inline int roundUp(int numToRound, int multiple) noexcept
{
    if (multiple == 0) {
        return numToRound;
    }

    const int remainder = numToRound % multiple;
    if (remainder == 0) {
        return numToRound;
    }

    return numToRound + multiple - remainder;
}

/// @class OverlappingRegions
///
/// \tparam sRegionOverlap How many pixels the regions overlap
/// \tparam sTileWidth How wide a tile is
/// \tparam sTileHeight How high a tile is
/// \tparam sDesiredNRegionsPerDimension The desired number of regions per dimension. E.g. "2" will attempt to subdivide
/// the area into 2*2 = 4 regions.
///
/// OverlappingRegions divides an integral region into at most sDesiredNRegionsPerDimension^2 sub regions.
/// These regions will overlap by sRegionOverlap.
///
/// This class is used by adaptive sampling. We divide into regions to reduce thread contention, and we overlap the
/// regions to help eliminate seams caused by this subdivision.
template <int sRegionOverlap, int sTileWidth, int sTileHeight, int sDesiredNRegionsPerDimension>
class OverlappingRegions
{
public:
    static_assert(sRegionOverlap % sTileWidth == 0, "Overlap must be a multiple of tile width");
    static_assert(sRegionOverlap % sTileHeight == 0, "Overlap must be a multiple of tile height");

    using IndexArray = std::array<int, 9>; // 9 to check all neighboring grid cells

    OverlappingRegions() = default;

    explicit OverlappingRegions(scene_rdl2::math::BBox2i bounds)
    : mRegionWidth(0)
    , mRegionHeight(0)
    , mNumRegionsPerDimension(0)
    , mOverallBounds{}
    {
        init(bounds);
    }

    // Expect bounds to be inclusive-exclusive in min and max [min, max).
    void init(scene_rdl2::math::BBox2i bounds)
    {
        const int width  = extents(bounds, 0);
        const int height = extents(bounds, 1);

        MNRY_ASSERT(width > 0);
        MNRY_ASSERT(height > 0);

        //   +----------------+----------------+
        //   |                |                |
        //   +------------------------------+  |
        //   |                |             |  |
        //   |                |             |  |
        //   |                |             |  |
        //   +---------------------------------+
        //   |                |             |  |
        //   |                |             |  |
        //   |                |             |  |
        //   |                |             |  |
        //   |                |             |  |
        //   +----------------+-------------+--+
        // 0, 0
        // Inner box is the render bounds
        // Outer box is the regions we create

        // Since we want our regions to overlap, the smallest region we want
        // should be a tile + our overlap.
        constexpr int minRegionWidth  = sTileWidth  + sRegionOverlap;
        constexpr int minRegionHeight = sTileHeight + sRegionOverlap;

        if (minRegionWidth >= width || minRegionHeight >= height) {
            mRegionWidth  = width;
            mRegionHeight = height;
            mNumRegionsPerDimension = 1;
        } else {
            // We keep the regions numbers the same in each dimension, no matter the image ratio.
            // This makes indexing easier.
            int regionsPerDimension = std::min(width/minRegionWidth, height/minRegionHeight);
            regionsPerDimension = std::min(sDesiredNRegionsPerDimension, regionsPerDimension);
            regionsPerDimension = std::max(1, regionsPerDimension);

            mRegionWidth  = roundUpDivision(width, regionsPerDimension);
            mRegionHeight = roundUpDivision(height, regionsPerDimension);

            // Round the regions up to tile sizes. This makes the overlap calculation easier, and makes tile queries for
            // a single region unambiguous.
            mRegionWidth = roundUp(mRegionWidth, sTileWidth);
            mRegionHeight = roundUp(mRegionHeight, sTileHeight);
            mNumRegionsPerDimension = regionsPerDimension;
        }

        mOverallBounds = std::move(bounds);
    }

    int getNumRegions() const
    {
        return mNumRegionsPerDimension*mNumRegionsPerDimension;
    }

    /// Gets the bounding box (with overlap) for a region at a specific index.
    /// \param idx The index of the bounding box in [0, getNumRegions).
    /// \return The overlapping bounding box for a region at idx.
    scene_rdl2::math::BBox2i getOverlappingRegionBounds(int idx) const
    {
        MNRY_ASSERT(idx >= 0);
        MNRY_ASSERT(idx < getNumRegions());

        int idxX;
        int idxY;
        getRegionCoords(idx, mNumRegionsPerDimension, idxX, idxY);

        const scene_rdl2::math::Vec2i lower{idxX * mRegionWidth, idxY * mRegionHeight};
        const scene_rdl2::math::Vec2i upper = lower + scene_rdl2::math::Vec2i{mRegionWidth, mRegionHeight};

        scene_rdl2::math::BBox2i bounds{mOverallBounds.lower + lower, mOverallBounds.lower + upper};
        bounds = enlarge(bounds, scene_rdl2::math::Vec2i(sRegionOverlap, sRegionOverlap));
        return intersect(bounds, mOverallBounds);
    }

    /// Get the bounds of the entire region (i.e. the union of all sub-regions)
    /// \return The bounds of the entire region represented by the instance of this class
    const scene_rdl2::math::BBox2i& getOverallBounds() const
    {
        return mOverallBounds;
    }

    /// Get the (single) region index in which a point falls (ignoring region overlap)
    /// \return The internal index for the region in which p falls
    int getRegionIndex(scene_rdl2::math::Vec2i p) const
    {
        p -= mOverallBounds.lower;
        const int x = p[0]/mRegionWidth;
        const int y = p[1]/mRegionHeight;
        return y * mNumRegionsPerDimension + x;
    }

    /// Get the (single) region index in which a tile falls (ignoring region overlap)
    /// This implies that the region bounds are aligned to tile multiples
    /// \return The internal index for the region in which the tile falls
    int getRegionIndex(const scene_rdl2::math::BBox2i& tile) const
    {
        return getRegionIndex(tile.lower);
    }

    /// Get a list of all regions into which tile lies (including region overlap)
    /// \param tile The tile to check
    /// \param indices The return value. This array may have redundant entries. The entries are valid up until a
    /// negative value is encountered.
    void getOverlappedRegions(const scene_rdl2::math::BBox2i& tile, IndexArray& indices) const
    {
        const scene_rdl2::math::Vec2i& p = tile.lower;

        int idx = 0;
        for (int y : {-sRegionOverlap, 0, sRegionOverlap}) {
            for (int x : {-sRegionOverlap, 0, sRegionOverlap}) {
                const scene_rdl2::math::Vec2i toCheck = p + scene_rdl2::math::Vec2i{x, y};
                if (conjointExclusive(mOverallBounds, toCheck)) {
                    indices[idx++] = getRegionIndex(toCheck);
                }
            }
        }
        // Mark the end of the array.
        if (idx < 9) indices[idx] = -1;
    }

private:
    /// Convert a one-dimensional sub-region index into x and y array values (for the lower point of the sub-region)
    /// \param idx The sub-region index
    /// \param numRegionsPerDimension The number of sub-regions per dimension (same in both dimensions)
    /// \param x Return value of sub-region's lower x value
    /// \param y Return value of sub-region's lower y value
    static void getRegionCoords(int idx, int numRegionsPerDimension, int& x, int& y) noexcept
    {
        x = idx%numRegionsPerDimension;
        y = idx/numRegionsPerDimension;
    }

    int mRegionWidth;
    int mRegionHeight;
    int mNumRegionsPerDimension;
    scene_rdl2::math::BBox2i mOverallBounds;
};

} // namespace rndr
} // namespace moonray


