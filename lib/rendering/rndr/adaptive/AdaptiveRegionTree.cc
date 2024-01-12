// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "AdaptiveRegionTree.h"

#include <scene_rdl2/common/fb_util/FbTypes.h>
#include <scene_rdl2/common/fb_util/Tiler.h>
#include <scene_rdl2/common/math/BBox.h>
#include <scene_rdl2/common/math/BBox2iIterator.h>
#include <scene_rdl2/common/math/Math.h>

#include <atomic>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <functional>

namespace moonray {
namespace rndr {

// Using the golden ratio for the minimum node size. Because we divide in real-space, a value of 1 probably still
// ensures that we end up sampling in neighboring tiles to reduce premature convergence issues. Also, being the most
// irrational number, this may lead to fewer tiling artifacts.
//static constexpr float sGoldenRatio = 1.61803398874989f;
static constexpr float sMinNodeSize = 1.27201964951407f; // sqrt(sGoldenRatio)
static constexpr float sMaxNodeSize = 16.0f;
static constexpr float sNoSplit = std::numeric_limits<float>::quiet_NaN();

inline bool noSplit(float v)
{
    return std::isnan(v);
}

static float findSplitLocationHelper(float location, float length)
{
    if (length * location >= sMinNodeSize) {
        // Our split is within minimum node size.
        return location;
    } else if (length >= sMinNodeSize * 2) {
        // Our split isn't within minimum node size: move split location.
        return sMinNodeSize/length;
    } else {
        // We can't split and still be within minimum node size.
        return sNoSplit;
    }
}

float AdaptiveRegionTree::findSplitLocation(float location, float length)
{
    if (location <= 0.5f) {
        return findSplitLocationHelper(location, length);
    } else {
        return 1.0f - findSplitLocationHelper(1.0f - location, length);
    }
}

int AdaptiveRegionTree::maxNodes(const scene_rdl2::math::BBox2i& bounds)
{
    const int w = extents(bounds, 0);
    const int h = extents(bounds, 1);

    if (w == 0 || h == 0) {
        return 0;
    }
    // Our leaf nodes are non-overlapping. This is the maximum number of leaf nodes we can have.
    const int nLeafNodes = static_cast<int>(scene_rdl2::math::ceil((w/sMinNodeSize) * (h/sMinNodeSize)));

    // A perfect binary tree has this many nodes given this many leaf nodes.
    return 2*nLeafNodes - 1;
}

/// @return Average error of leaf nodes.
float AdaptiveRegionTree::update(const scene_rdl2::fb_util::Tiler& tiler,
                                 const scene_rdl2::fb_util::RenderBuffer& renderBuf,
                                 const scene_rdl2::fb_util::FloatBuffer& numSamplesBuf,
                                 const scene_rdl2::fb_util::RenderBuffer& renderBufOdd)
{
    // On update, we rebuild the entire tree so that it gets rebalanced.
    mRoot.mChildren = nullptr;
    mRoot.mStatus = Node::Status::unconverged;
    mNodePool.clear();

    // mIntegerRootBounds should be closed-open (inclusive-exclusive).
    for (auto pit = begin(mIntegerRootBounds); pit != end(mIntegerRootBounds); ++pit) {
        const scene_rdl2::math::Vec2i& p = *pit;
        const scene_rdl2::math::Vec2i pRelative = p - mIntegerRootBounds.lower;
        float pixelError = getPixelError(p[0], p[1], tiler, renderBuf, numSamplesBuf, renderBufOdd);
        if (std::isinf(pixelError)) {
            return pixelError;
        }
        mPixelErrors(pRelative[0], pRelative[1]) = pixelError;
    }
    const float error = updateImpl(mRoot);
    return error;
}

/// @return Average error of leaf nodes.
float AdaptiveRegionTree::updateImpl(Node& node)
{
    using AdaptiveNS::Axis;

    // This threshold comes from the Dammertz paper.
    constexpr float sSplitThreshold = 256.0f;

    AdaptiveNS::MemoryPoolRAII<float> accumulatedErrorPoolClearRAII(mAccumulatedErrorPool);
    AdaptiveNS::PoolAllocator<float> accumlatedErrorAllocator(mAccumulatedErrorPool);

    const auto ext = size(node.mBounds);
    const float width = ext[0];
    const float height = ext[1];

    MNRY_ASSERT(volume(node.mBounds) > 0);
    const Axis axis = (width > height) ? Axis::horizontal : Axis::vertical;
    const float lengthAlongSplit = std::max(width, height);

    const scene_rdl2::math::BBox2i integerNodeBounds = roundLarger(node.mBounds);

    // _accumulatedError is the partial sum of the pixel errors and the last element is the sum of all pixel errors
    // in the vertical or horizontal axis.
    auto accumulatedError = (axis == Axis::horizontal) ?
                                  AdaptiveNS::orientedAccumulatedPixelError<0>(mIntegerRootBounds,
                                                                               integerNodeBounds,
                                                                               mPixelErrors,
                                                                               accumlatedErrorAllocator) :
                                  AdaptiveNS::orientedAccumulatedPixelError<1>(mIntegerRootBounds,
                                                                               integerNodeBounds,
                                                                               mPixelErrors,
                                                                               accumlatedErrorAllocator);

    const float vr = volume(node.mBounds);
    const float error = calculateAreaError(accumulatedError.back(), vr);

    if (error < mTargetError && lengthAlongSplit < sMaxNodeSize) {
        node.mStatus = Node::Status::complete;
        return error;
    } else if (error < mTargetError*sSplitThreshold && lengthAlongSplit > sMinNodeSize*2) { // We can split in half
        const float offset = findSplitLocation(accumulatedError, lengthAlongSplit);
        MNRY_ASSERT(!noSplit(offset));
        if (unlikely(noSplit(offset))) {
            return error;
        }

        // This is just a sanity check to make sure that our offset is reasonable. We only look at a very-relaxed
        // .9 of the node size for precision errors.
        MNRY_ASSERT(lengthAlongSplit * offset >= sMinNodeSize * 0.9f);
        MNRY_ASSERT(lengthAlongSplit * (1.0f - offset) >= sMinNodeSize * 0.9f);
        MNRY_ASSERT(offset >= 0.0f);
        MNRY_ASSERT(offset <= 1.0f);

        // Now that we've divided the error, we no longer need the value nor the memory. Free up for recursion.
        // Since accumulatedError is just a container of floats, we don't _strictly_ have to clear it, but let's be
        // proper so that we don't call the destructor on memory that's no longer ours.
        accumulatedError.clear();
        mAccumulatedErrorPool.clear();
        switch (axis) {
            case Axis::horizontal:
                splitHorizontal(mNodePool, node, offset);
                break;
            case Axis::vertical:
                splitVertical(mNodePool, node, offset);
                break;
        }
        MNRY_ASSERT(hasChildren(node));
        const float errorLeft  = updateImpl(node.getLeftChild());
        const float errorRight = updateImpl(node.getRightChild());
        if (node.getLeftChild().mStatus == Node::Status::complete &&
            node.getRightChild().mStatus == Node::Status::complete) {
            node.mStatus = Node::Status::complete;
        }
        // Return the mean error of our children.
        return scene_rdl2::math::max(errorLeft, errorRight);
    }
    return error;
}

ActivePixelMask AdaptiveRegionTree::getSampleAreaImpl(const scene_rdl2::math::BBox2i& tile, const Node& node) const
{
    auto mask = ActivePixelMask::none();
    if (node.mStatus == Node::Status::complete) {
        return mask;
    }
    const auto toCheck = toFloat(tile);
    const auto isect = intersect(node.mBounds, toCheck);
    if (isect.empty() || volume(isect) <= 0) {
        return mask;
    }

    if (hasChildren(node)) {
        mask = getSampleAreaImpl(tile, node.getLeftChild()) | getSampleAreaImpl(tile, node.getRightChild());
    } else {
        const auto region = roundLarger(isect);
        for (auto p: region) {
            const scene_rdl2::math::Vec2i pTileRelative = p - tile.lower;
            mask.set(pTileRelative[0], pTileRelative[1]);
        }
    }

    // This mimics legacy behavior where we were performing a union on axis-aligned bounding boxes.
    mask.fillGaps();
    return mask;
}

void AdaptiveRegionTree::svg(std::ostream& outs) const
{
    const char* header = R"(<?xml version="1.0" encoding="UTF-8" ?>)";
    const char* footer = R"(</svg>)";

    outs << header << '\n';
    const int width = extents(mIntegerRootBounds, 0);
    const int height = extents(mIntegerRootBounds, 1);
    //  viewBox="-70.5 -70.5 391 391"
    outs << R"(<svg width=")" << width << R"(" height=")" << height << R"(" viewBox=")" << mIntegerRootBounds.lower[0] << ' ' << mIntegerRootBounds.lower[1] << ' ' << width << ' ' << height << ' ' << R"(" xmlns="http://www.w3.org/2000/svg">)";
    svgImpl(outs, mRoot);
    outs << footer << '\n';
}

void AdaptiveRegionTree::svgImpl(std::ostream& outs, const Node& node) const
{
    if (hasChildren(node)) {
        svgImpl(outs, node.getLeftChild());
        svgImpl(outs, node.getRightChild());
    } else {
        const auto width = static_cast<int>(std::round(extents(node.mBounds, 0)));
        const auto height = static_cast<int>(std::round(extents(node.mBounds, 1)));
        const auto x = static_cast<int>(std::round(node.mBounds.lower[0]));
        const int y = mIntegerRootBounds.lower[1] +
                      mIntegerRootBounds.upper[1] -
                      static_cast<int>(std::round(node.mBounds.lower[1]));

        // e.g.
        //<rect x="25" y="25" width="200" height="200" fill="lime" stroke-width="4" stroke="pink" />
        outs << R"(<rect x=")" << x     << R"(" y=")" << y
             << R"(" width=")" << width << R"(" height=")" << height
             << R"(" fill="white" stroke-width="2" stroke="black" />)" << '\n';
    }
}

bool AdaptiveRegionTree::savePixelErrorsByPPM(const std::string& filename) const
//
// For debugging purpose
// save pixel error valaue into disk by PPM (RGB) format w/ channel value resolution as 0~4095.
//   R : pixel error clipped by 0.0~1.0 range goes to value of 0 ~ 4095.
//   G : pixel is covered by adaptive tree node (4095) or not (0)
//   B : pixel is converged (4095) or not-converged (0)
//
{
    constexpr int valReso = 4096;
    auto toIntPixError = [&](const float v) -> int {
        auto clamp = [](const float v) -> float { return ((v < 0.0f)? 0.0f: ((v > 1.0f)? 1.0f: v)); };
        return (int)(clamp(v) * (float)(valReso - 1) + 0.5f);
    };
    auto toIntPixCondition = [&](const bool flag) -> int { return (flag)? 0: valReso - 1; };
    auto getPixCondition = [&](const int pixCondition, bool& doneFlag, bool& nodeCoveredFlag) {
        // 3 2 1 0
        //     | |
        //     | +-- pixel covered by Node
        //     +---- Done pixel
        nodeCoveredFlag = (pixCondition & 0x1)? true: false;
        doneFlag = (pixCondition & 0x2)? true: false;
    };

    std::ofstream ofs(filename);
    if (!ofs) return false;     // can not create file.

    scene_rdl2::util::Array2D<int> pixCondition = calcPixelCondition();
    const auto& pixError = mPixelErrors;

    ofs << "P3\n" << pixError.getWidth() << ' ' << pixError.getHeight() << '\n' << (valReso - 1) << '\n';
    for (int v = pixError.getHeight() - 1; v >= 0; --v) {
        for (int u = 0; u < pixError.getWidth(); ++u) {
            bool nodeCoveredFlag, doneFlag;
            getPixCondition(pixCondition(u, v), doneFlag, nodeCoveredFlag);
            ofs << toIntPixError(pixError(u, v)) << ' '
                << toIntPixCondition(nodeCoveredFlag) << ' '
                << toIntPixCondition(doneFlag) << ' ';
        }
    }

    ofs.close();
    return true;
}

bool AdaptiveRegionTree::saveRenderBufferByPPM(const std::string& filename,
                                               const scene_rdl2::fb_util::Tiler& tiler,
                                               const scene_rdl2::fb_util::RenderBuffer& renderBuf) const
//
// For debugging purpose
// Save renderBuf's RGB (clipped 0.0~1.0) to PPM format w/ value resolution as 0~4095.
//
{
    constexpr int valReso = 4096;
    auto toIntFCol = [&](const float v) -> int {
        auto clamp = [](const float v) -> float { return ((v < 0.0f)? 0.0f: ((v > 1.0f)? 1.0f: v)); };
        return (int)(clamp(v) * (float)(valReso - 1) + 0.5f);
    };

    std::ofstream ofs(filename);
    if (!ofs) return false;

    int xMin = mIntegerRootBounds.lower[0];
    int yMin = mIntegerRootBounds.lower[1];
    int xMax = mIntegerRootBounds.upper[0] - 1;
    int yMax = mIntegerRootBounds.upper[1] - 1;
    int xSize = xMax - xMin + 1;
    int ySize = yMax - yMin + 1;

    ofs << "P3\n" << xSize << ' ' << ySize << '\n' << (valReso - 1) << '\n';
    for (int y = yMax; y >= yMin; --y) {
        for (int x = xMin; x <= xMax; ++x) {
            unsigned px, py;
            tiler.linearToTiledCoords((unsigned)x, (unsigned)y, &px, &py);
            const scene_rdl2::fb_util::RenderColor& c = renderBuf.getPixel(px, py);
            ofs << toIntFCol(c[0]) << ' ' << toIntFCol(c[1]) << ' ' << toIntFCol(c[2]) << ' ';
        }
    }

    ofs.close();
    return true;
}

bool AdaptiveRegionTree::saveFloatBufferByPPM(const std::string& filename,
                                              const scene_rdl2::fb_util::Tiler& tiler,
                                              const scene_rdl2::fb_util::FloatBuffer& floatBuf) const
//
// For debugging purpose (mainly weight/numSample value tracking purpose)
// Save floatBuf's val (clipped 0.0~100.0) to PPM format w/ value resolution as 0~4095.
//
{
    constexpr int valReso = 4096;
    auto toIntFCol = [&](const float v) -> int {
        auto clamp = [](const float v) -> float { return ((v < 0.0f)? 0.0f: ((v > 1.0f)? 1.0f: v)); };
        return (int)(clamp(v / 100.0f) * (float)(valReso - 1) + 0.5f);
    };

    std::ofstream ofs(filename);
    if (!ofs) return false;

    int xMin = mIntegerRootBounds.lower[0];
    int yMin = mIntegerRootBounds.lower[1];
    int xMax = mIntegerRootBounds.upper[0] - 1;
    int yMax = mIntegerRootBounds.upper[1] - 1;
    int xSize = xMax - xMin + 1;
    int ySize = yMax - yMin + 1;

    ofs << "P3\n" << xSize << ' ' << ySize << '\n' << (valReso - 1) << '\n';
    for (int y = (int)yMax; y >= yMin; --y) {
        for (int x = xMin; x <= xMax; ++x) {
            unsigned px, py;
            tiler.linearToTiledCoords((unsigned)x, (unsigned)y, &px, &py);
            const float& c = floatBuf.getPixel(px, py);
            ofs << toIntFCol(c) << ' ' << toIntFCol(c) << ' ' << toIntFCol(c) << ' ';
        }
    }

    ofs.close();
    return true;
}

scene_rdl2::util::Array2D<int> AdaptiveRegionTree::calcPixelCondition() const
//
// For debugging purpose
// Compute pixel condition array based on adaptive tree information
//
{
    auto calcPixConditionVal = [&](const bool donePixel) -> int {
        // 3 2 1 0
        //     | |
        //     | +-- pixel covered by Node
        //     +---- Done pixel
        int flag = 0x1;
        if (donePixel) flag |= 0x2;
        return flag;
    };
    auto fillRegionByVal = [&](scene_rdl2::util::Array2D<int>& array, const scene_rdl2::math::BBox2f& bbox, const int val) {
        for (int v = (int)bbox.lower[1]; v <= (int)bbox.upper[1]; ++v) {
            for (int u = (int)bbox.lower[0]; u <= (int)bbox.upper[0]; ++u) {
                int uu = u - mIntegerRootBounds.lower[0];
                int vv = v - mIntegerRootBounds.lower[1];
                if (!(array(uu, vv) & 0x1)) {
                    array(uu, vv) = val; // very 1st time to set
                } else {
                    if (array(uu, vv) & 0x2) {
                        array(uu, vv) = val; // only update if this pixel is done-condition
                    }
                }
            }
        }
    };
    auto fillRegionByCondition = [&](scene_rdl2::util::Array2D<int>& array, const scene_rdl2::math::BBox2f& bbox, const bool done) {
        fillRegionByVal(array, bbox, calcPixConditionVal(done));
    };
    std::function<void(scene_rdl2::util::Array2D<int>&, const Node&)> fillNode =
        [&](scene_rdl2::util::Array2D<int>& array, const Node& node) {
            if (node.mStatus == Node::Status::complete) {
                fillRegionByCondition(array, node.mBounds, true);
            } else {
                if (node.mChildren) {
                    fillNode(array, node.getLeftChild());
                    fillNode(array, node.getRightChild());
                } else {
                    fillRegionByCondition(array, node.mBounds, false);
                }
            }
        };

    scene_rdl2::util::Array2D<int> pixCondition((int)extents(mRoot.mBounds, 0) + 1, (int)extents(mRoot.mBounds, 1) + 1);
    fillRegionByVal(pixCondition, mRoot.mBounds, 0x0);
    fillNode(pixCondition, mRoot);
    return pixCondition;
}

} // namespace rndr
} // namespace moonray

