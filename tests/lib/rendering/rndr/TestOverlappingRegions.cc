// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "TestOverlappingRegions.h"
#include <moonray/rendering/rndr/adaptive/OverlappingRegions.h>

#include <iostream>

namespace moonray {
namespace rndr {
namespace unittest {

namespace {
/// This test mimics what happens in AdaptiveRegions, where a tile informs the region that it is done rendering.
/// AdaptiveRegions keeps a count of the number of tiles in each region, and when all of the tiles in the region have
/// been updated, the corresponding tree is locked and updated. This code mimics counting the number of tiles that pass
/// through a region.
template <int sRegionOverlap, int sTileWidth, int sTileHeight, int sDesiredRegionsPerDimension>
void
count(const OverlappingRegions<sRegionOverlap, sTileWidth, sTileHeight, sDesiredRegionsPerDimension>& regions, int line)
{
    using Regions = OverlappingRegions<sRegionOverlap, sTileWidth, sTileHeight, sDesiredRegionsPerDimension>;
    using IndexArray = typename Regions::IndexArray;

    std::cout << "Count test line: " << line << '\n';

    const int numRegions = regions.getNumRegions();
    std::vector<int> tileCount(numRegions, 0);

    for (int i = 0; i < numRegions; ++i) {
        const scene_rdl2::math::BBox2i regionBounds = regions.getOverlappingRegionBounds(i);
        const int tilesHorizontal = roundUpDivision(extents(regionBounds, 0), sTileWidth);
        const int tilesVertical   = roundUpDivision(extents(regionBounds, 1), sTileHeight);

        tileCount[i] = tilesHorizontal * tilesVertical;
    }

    const scene_rdl2::math::BBox2i& bounds = regions.getOverallBounds();

    for (int y = bounds.lower[1]; y < bounds.upper[1]; y += sTileHeight) {
        for (int x = bounds.lower[0]; x < bounds.upper[0]; x += sTileWidth) {
            const scene_rdl2::math::BBox2i tile{scene_rdl2::math::Vec2i{x, y}, scene_rdl2::math::Vec2i{x + sTileWidth, y + sTileHeight}};
            IndexArray indexArray;
            std::vector<bool> visited(numRegions, false);

            regions.getOverlappedRegions(tile, indexArray);

            for (std::size_t i = 0; i < indexArray.size() && indexArray[i] >= 0; ++i) {
                const int idx = indexArray[i];
                if (visited[idx]) {
                    continue;
                }
                visited[idx] = true;
                --tileCount[idx];
            }
        }
    }
    for (int i = 0; i < numRegions; ++i) {
        CPPUNIT_ASSERT(tileCount[i] == 0);
    }
}
} // anonymous namespace

#define OVERLAPPING_REGION_TEST(overlap, tilewidth, tileheight, regionsPerDimension, minX, minY, maxX, maxY) \
    count(OverlappingRegions<overlap, tilewidth, tileheight, regionsPerDimension>(scene_rdl2::math::BBox2i{scene_rdl2::math::Vec2i{minX, minY}, scene_rdl2::math::Vec2i{maxX, maxY}}), __LINE__)

void
TestOverlappingRegions::testRegions()
{
    using namespace scene_rdl2::math;

    OVERLAPPING_REGION_TEST( 0, 8, 8, 2,  0,  0, 640, 480);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 2,  0,  0, 640, 480);
    OVERLAPPING_REGION_TEST(16, 8, 8, 2,  0,  0, 640, 480);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 2, 16, 32, 640, 480);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 2, 16, 32, 640, 480);
    OVERLAPPING_REGION_TEST(16, 8, 8, 2, 16, 32, 640, 480);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 2, 11, 13, 631, 648);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 2, 11, 13, 631, 648);
    OVERLAPPING_REGION_TEST(16, 8, 8, 2, 11, 13, 631, 648);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 2,  0,  0,   9,   9);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 2,  0,  0,  17,  17);
    OVERLAPPING_REGION_TEST(16, 8, 8, 2,  0,  0,   9,   9);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 2,  0,  0, 639, 479);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 2,  0,  0, 639, 479);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 2,  0,  0, 641, 481);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 2,  0,  0, 641, 481);
    OVERLAPPING_REGION_TEST(16, 8, 8, 2,  0,  0, 641, 481);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 2,  0,  0, 642, 482);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 2,  0,  0, 642, 482);
    OVERLAPPING_REGION_TEST(16, 8, 8, 2,  0,  0, 642, 482);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 2, 11, 13, 641, 481);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 2, 11, 13, 641, 481);
    OVERLAPPING_REGION_TEST(16, 8, 8, 2, 11, 13, 641, 481);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 2, 11, 13, 642, 482);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 2, 11, 13, 642, 482);
    OVERLAPPING_REGION_TEST(16, 8, 8, 2, 11, 13, 642, 482);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 2, 11, 13,  12,  14);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 2, 11, 13,  12,  14);
    OVERLAPPING_REGION_TEST(16, 8, 8, 2, 11, 13,  12,  14);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 2, 11, 13,  13,  15);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 2, 11, 13,  13,  15);
    OVERLAPPING_REGION_TEST(16, 8, 8, 2, 11, 13,  13,  15);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 2, 11, 13, 11 + 8, 13 + 8);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 2, 11, 13, 11 + 8, 13 + 8);
    OVERLAPPING_REGION_TEST(16, 8, 8, 2, 11, 13, 11 + 8, 13 + 8);

    OVERLAPPING_REGION_TEST( 0, 8, 8, 5,  0,  0, 640, 480);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 5,  0,  0, 640, 480);
    OVERLAPPING_REGION_TEST(16, 8, 8, 5,  0,  0, 640, 480);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 5, 16, 32, 640, 480);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 5, 16, 32, 640, 480);
    OVERLAPPING_REGION_TEST(16, 8, 8, 5, 16, 32, 640, 480);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 5, 11, 13, 631, 648);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 5, 11, 13, 631, 648);
    OVERLAPPING_REGION_TEST(16, 8, 8, 5, 11, 13, 631, 648);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 5,  0,  0,   9,   9);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 5,  0,  0,  17,  17);
    OVERLAPPING_REGION_TEST(16, 8, 8, 5,  0,  0,   9,   9);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 5,  0,  0, 639, 479);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 5,  0,  0, 639, 479);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 5,  0,  0, 641, 481);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 5,  0,  0, 641, 481);
    OVERLAPPING_REGION_TEST(16, 8, 8, 5,  0,  0, 641, 481);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 5,  0,  0, 642, 482);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 5,  0,  0, 642, 482);
    OVERLAPPING_REGION_TEST(16, 8, 8, 5,  0,  0, 642, 482);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 5, 11, 13, 641, 481);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 5, 11, 13, 641, 481);
    OVERLAPPING_REGION_TEST(16, 8, 8, 5, 11, 13, 641, 481);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 5, 11, 13, 642, 482);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 5, 11, 13, 642, 482);
    OVERLAPPING_REGION_TEST(16, 8, 8, 5, 11, 13, 642, 482);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 5, 11, 13,  12,  14);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 5, 11, 13,  12,  14);
    OVERLAPPING_REGION_TEST(16, 8, 8, 5, 11, 13,  12,  14);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 5, 11, 13,  13,  15);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 5, 11, 13,  13,  15);
    OVERLAPPING_REGION_TEST(16, 8, 8, 5, 11, 13,  13,  15);
    OVERLAPPING_REGION_TEST( 0, 8, 8, 5, 11, 13, 11 + 8, 13 + 8);
    OVERLAPPING_REGION_TEST( 8, 8, 8, 5, 11, 13, 11 + 8, 13 + 8);
    OVERLAPPING_REGION_TEST(16, 8, 8, 5, 11, 13, 11 + 8, 13 + 8);
}

} // namespace unittest
} // namespace rndr
} // namespace moonray

