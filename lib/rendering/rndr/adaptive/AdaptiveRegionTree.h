// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "ActivePixelMask.h"
#include <moonray/common/mcrt_util/Atomic.h>

#include <scene_rdl2/common/fb_util/FbTypes.h>
#include <scene_rdl2/common/fb_util/Tiler.h>
#include <scene_rdl2/common/math/BBox.h>
#include <scene_rdl2/common/math/BBox2iIterator.h>
#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/Math.h>
#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/render/util/Array2D.h>
#include <scene_rdl2/render/util/TypedStaticallySizedMemoryPool.h>

#include <algorithm>
#include <atomic>
#include <iosfwd>
#include <iterator>
#include <memory>
#include <numeric>

// Use the error as specified in the Dammertz paper, as opposed to our own modified version.
#define USE_DAMMERTZ_PAPER_ERROR 0

namespace moonray {
namespace rndr {

namespace AdaptiveNS
{
template <typename T>
using MemoryPool = scene_rdl2::alloc::TypedStaticallySizedMemoryPool<T>;

template <typename T>
using PoolAllocator = scene_rdl2::alloc::TypedStaticallySizedPoolAllocator<T>;

template <typename T>
using MemoryPoolRAII = scene_rdl2::alloc::TypedStaticalySizedMemoryPoolRAII<T>;

__forceinline float luma(const scene_rdl2::math::Vec4f& a) noexcept { return 0.299f*a[0] + 0.587f*a[1] + 0.114f*a[2]; }
__forceinline bool isnan(const scene_rdl2::math::Vec4f& a) { return std::isnan(a[0]) || std::isnan(a[1]) || std::isnan(a[2]) || std::isnan(a[3]); }

//
// This particular heuristic is a variation on the error metric proposed in the paper
// "A Hierarchical Automatic Stopping Condition for Monte Carlo Global Illumination",
// by Dammertz et al.
//
inline float
estimatePixelErrorInternal(unsigned px,
                           unsigned py,
                           const scene_rdl2::fb_util::Tiler& tiler,
                           const scene_rdl2::fb_util::RenderBuffer& renderBuf,
                           const scene_rdl2::fb_util::FloatBuffer& numSamplesBuf,
                           const scene_rdl2::fb_util::RenderBuffer& renderBufOdd)
{
    tiler.linearToTiledCoords(px, py, &px, &py);

    const float* const sampleCountPointer = &(numSamplesBuf.getPixel(px, py));
    const float totalSamples = util::atomicLoad(sampleCountPointer, std::memory_order_relaxed);
    const float numOddSamples = scene_rdl2::math::floor(totalSamples * 0.5f);

    const float numEvenSamples = totalSamples - numOddSamples;

    if (numEvenSamples <= 0.0f || numOddSamples <= 0.0f) {
        return std::numeric_limits<float>::infinity();
    }

    const float* const totalColorPointer = &(renderBuf.getPixel(px, py)[0]);
    const float* const oddColorPointer   = &(renderBufOdd.getPixel(px, py)[0]);
    alignas(util::kDoubleQuadWordAtomicAlignment) scene_rdl2::math::Vec4f totalColor;
    util::atomicLoadFloat4(&totalColor[0], totalColorPointer);
    alignas(util::kDoubleQuadWordAtomicAlignment) scene_rdl2::math::Vec4f oddColor;
    util::atomicLoadFloat4(&oddColor[0], oddColorPointer);

    if (isnan(totalColor) || isnan(oddColor)) {
        return 0.0f;
    }

    //
    // We want to compute:
    //
    //            even_color          oddColor
    //   diff = ---------------  -  -------------
    //          numEvenSamples      numOddSamples
    //
    // where even_color = totalColor - oddColor
    //
    // We can refactor this expression remove one of the divides.
    //
    const scene_rdl2::math::Vec4f diff = (totalColor * numOddSamples - oddColor * totalSamples) /
                                         (numOddSamples * numEvenSamples);
    const scene_rdl2::math::Vec4f absdiff = abs(diff);
    const float lumDiff = luma(absdiff);
    const float lumAvg = luma(totalColor / totalSamples);

    const float alphaScore = absdiff[3];

    // Radiance is pure black. Return the difference in the alpha to see if there's still work to do.
    if (lumAvg <= 0.0f) {
        return alphaScore;
    }

    // Normalize result and use sqrt to approximate eye's response to linear light
    // (i.e. fake a pseudo gamma curve).
    //
    // Alpha and color are both considered for error evaluation.  Alpha should
    // be in [0, 1], whereas the color luma is in non-negative real
    // space. Even though they are in different scales, this seems to work
    // fairly well to make sure we resolve edges.
    return (lumDiff * scene_rdl2::math::rsqrt(lumAvg)) + alphaScore;
}

/// @function orientedAccumulatedPixelError
/// The partial sum of errors in a direction is used for finding a split location when subdividing a tree node.
/// @return This returns a partial sum of sums along an axis in _region_
//
// If our data looks like this:
//
//  4   2   5   2   6
//  2   6   2   2   7
//  2   6   3   3   2
//  4   6   4   3   1
//
// The vertical marginal sum looks like this:
//
// 12  20  14  10  16
//
// And the partial sum of that looks like this:
// 12  32  46  56  72
// This is used to find our split position.
template <int axis>
std::vector<float, AdaptiveNS::PoolAllocator<float>>
orientedAccumulatedPixelError(const scene_rdl2::math::BBox2i& baseRegion,
                              const scene_rdl2::math::BBox2i& region,
                              const scene_rdl2::util::Array2D<float>& pixelErrors,
                              AdaptiveNS::PoolAllocator<float>& allocator)
{
    const int length = extents(region, axis);

    // We're using an allocator that has a fixed size: we don't want to let the std::vector manage its memory by growing
    // past its capacity: we may run out of space as it has to move all of its elements later in the array.
    std::vector<float, AdaptiveNS::PoolAllocator<float>> marginalErrors(length, 0.0f, allocator);

    // This doesn't work on ICC 15 for some reason. It works on other compilers. :-/
    //for (auto p : region) {

    for (auto pit = begin(region); pit != end(region); ++pit) {
        const scene_rdl2::math::Vec2i& p = *pit;
        const scene_rdl2::math::Vec2i pBaseRelative = p - baseRegion.lower;
        const int idx = p[axis] - region.lower[axis];
        const float pixelError = pixelErrors(pBaseRelative[0], pBaseRelative[1]);
        marginalErrors[idx] += pixelError;
    }

    std::partial_sum(marginalErrors.begin(), marginalErrors.end(), marginalErrors.begin());
    return marginalErrors;
}

enum class Axis
{
    horizontal,
    vertical
};

} // namespace AdaptiveNS

class AdaptiveRegionTree
{
    struct Node
    {
        enum class Status : std::uint8_t
        {
            unconverged,
            complete
        };

        Node()                       = default;
        Node(const Node&)            = delete;
        Node(Node&&)                 = default;
        Node& operator=(const Node&) = delete;
        Node& operator=(Node&&)      = default;

        Node& getLeftChild() noexcept
        {
            MNRY_ASSERT(mChildren);
            return mChildren[0];
        }

        Node& getRightChild() noexcept
        {
            MNRY_ASSERT(mChildren);
            return mChildren[1];
        }

        const Node& getLeftChild() const noexcept
        {
            MNRY_ASSERT(mChildren);
            return mChildren[0];
        }

        const Node& getRightChild() const noexcept
        {
            MNRY_ASSERT(mChildren);
            return mChildren[1];
        }

        // TODO: We could save space by having a(n external) top-level bounding box, and a split axis and offset
        // for each node, creating the bounding box as we traverse the tree. The axis could be stored in the sign
        // bit of the offset.
        scene_rdl2::math::BBox2f mBounds;
        Node* mChildren{nullptr};
        Status mStatus{Status::unconverged};
    };

    static bool hasChildren(const Node& n) noexcept
    {
        return n.mChildren != nullptr;
    }

    static scene_rdl2::math::BBox2i roundLarger(const scene_rdl2::math::BBox2f& b)
    {
        const int minX = static_cast<int>(scene_rdl2::math::floor(b.lower[0]));
        const int minY = static_cast<int>(scene_rdl2::math::floor(b.lower[1]));
        const int maxX = static_cast<int>(scene_rdl2::math::ceil(b.upper[0]));
        const int maxY = static_cast<int>(scene_rdl2::math::ceil(b.upper[1]));

        return scene_rdl2::math::BBox2i{scene_rdl2::math::Vec2i{minX, minY}, scene_rdl2::math::Vec2i{maxX, maxY}};
    }

    static scene_rdl2::math::BBox2f toFloat(const scene_rdl2::math::BBox2i& b)
    {
        return scene_rdl2::math::BBox2f{
            scene_rdl2::math::Vec2f{static_cast<float>(b.lower[0]), static_cast<float>(b.lower[1])},
            scene_rdl2::math::Vec2f{static_cast<float>(b.upper[0]), static_cast<float>(b.upper[1])}
        };
    }

    /// @return A fraction along the continuous space approximated by accumulatedError that splits the error in half.
    template <typename Allocator>
    static float findMidpointSplitLocation(const std::vector<float, Allocator>& accumulatedError)
    {
        MNRY_ASSERT(!accumulatedError.empty());

        // Precondition: accumulated error is a partial sum (implying it's weakly increasing).

        // The last value will be the total sum, because the accumulated error is a partial sum.
        const float half = accumulatedError.back() * 0.5f;

        // Binary search for the mid-point value.
        const auto lbiter = std::upper_bound(accumulatedError.cbegin(), accumulatedError.cend(), half);

        int i;
        if (accumulatedError.back() == 0.f)
            i = accumulatedError.size() / 2;
        else
            i = std::distance(accumulatedError.cbegin(), lbiter);

        const float lower  = accumulatedError[i - 1];
        const float higher = accumulatedError[i];

        // The rough offset gives us the index to a value that's close to the middle.
        const float roughOffset = static_cast<float>(i);

        // Further refine the rough offset between the lower and higher values.
        const float delta = higher - lower;
        const float deltaOffset = (half - lower) / delta;
        const float fraction = (roughOffset + deltaOffset) / accumulatedError.size();
        return fraction;
    }

    static float findSplitLocation(float location, float length);

    template <typename Allocator>
    inline static float findSplitLocation(const std::vector<float, Allocator>& accumulatedError, float length)
    {
        const float location = findMidpointSplitLocation(accumulatedError);
        return findSplitLocation(location, length);
    }

    static int maxNodes(const scene_rdl2::math::BBox2i& bounds);

public:
    AdaptiveRegionTree()
    : mRoot()
    , mIntegerRootBounds()
    , mTargetError(0)
    , mPixelErrors()
    , mNodePool(0)
    , mAccumulatedErrorPool(0)
    {
    }

    AdaptiveRegionTree(const scene_rdl2::math::BBox2i& bounds, float targetError)
    : mRoot()
    , mIntegerRootBounds(bounds)
    , mTargetError(targetError)
    , mPixelErrors(extents(bounds, 0), extents(bounds, 1))
    , mNodePool(maxNodes(bounds))
    , mAccumulatedErrorPool(std::max(extents(bounds, 0), extents(bounds, 1)))
    {
        mRoot.mBounds = toFloat(bounds);
    }

    AdaptiveRegionTree(const AdaptiveRegionTree&)            = delete;
    AdaptiveRegionTree(AdaptiveRegionTree&&)                 = default;
    AdaptiveRegionTree& operator=(const AdaptiveRegionTree&) = delete;
    AdaptiveRegionTree& operator=(AdaptiveRegionTree&&)      = default;

    ActivePixelMask getSampleArea(const scene_rdl2::math::BBox2i& tile) const
    {
        return getSampleAreaImpl(tile, mRoot);
    }

    /// @return Max error of leaf nodes.
    float update(const scene_rdl2::fb_util::Tiler& tiler,
                 const scene_rdl2::fb_util::RenderBuffer& renderBuf,
                 const scene_rdl2::fb_util::FloatBuffer& numSamplesBuf,
                 const scene_rdl2::fb_util::RenderBuffer& renderBufOdd);

    bool done() const noexcept { return mRoot.mStatus == Node::Status::complete; }

    void svg(std::ostream& outs) const;

    bool savePixelErrorsByPPM(const std::string& filename) const; // for debug
    bool saveRenderBufferByPPM(const std::string& filename,
                               const scene_rdl2::fb_util::Tiler& tiler,
                               const scene_rdl2::fb_util::RenderBuffer& renderBuf) const; // for debug
    bool saveFloatBufferByPPM(const std::string& filename,
                              const scene_rdl2::fb_util::Tiler& tiler,
                              const scene_rdl2::fb_util::FloatBuffer& floatBuf) const; // for debug

private:
    ActivePixelMask getSampleAreaImpl(const scene_rdl2::math::BBox2i& tile, const Node& node) const;

    static float calculateAreaError(float error, float nodeArea)
    {
        // We used to pass in image area based on the actual...get this, image area. However, this makes the results
        // inconsistent when doing sub-regions, so now we just hard-code an image area.

        // These values come from previous settings of a_vp_xmin, a_vp_xmax, a_vp_ymin, a_vp_ymax (image resolution).
        // We divide by four, because at the time of this change, we had four regions for adaptive sampling, and we want
        // this to be relatively consistent with renders before that change.
        constexpr float xmin =  -24.0f;
        constexpr float xmax = 1943.0f;
        constexpr float ymin =    0.0f;
        constexpr float ymax =  815.0f;
        constexpr float imageArea = ((ymax - ymin) * (xmax - xmin)) / 4.0f;

        // This comes out of the Dammertz  paper, but we re-write the math a bit.
        // const float r = std::sqrt(nodeArea/static_cast<float>(imageArea));
        // const float e = r/nodeArea * error;

        // The above math can be re-written as this, assuming the values are positive.
        const float e = error * scene_rdl2::math::rsqrt(nodeArea*imageArea);
        return e;
    }

    float getPixelError(unsigned px,
                        unsigned py,
                        const scene_rdl2::fb_util::Tiler& tiler,
                        const scene_rdl2::fb_util::RenderBuffer& renderBuf,
                        const scene_rdl2::fb_util::FloatBuffer& numSamplesBuf,
                        const scene_rdl2::fb_util::RenderBuffer& renderBufOdd) const
    {
        return AdaptiveNS::estimatePixelErrorInternal(px, py, tiler, renderBuf, numSamplesBuf, renderBufOdd);
    }

    /// @return Average error of leaf nodes.
    float updateImpl(Node& node);

    static void splitHorizontal(AdaptiveNS::MemoryPool<Node>& pool, Node& node, float offset)
    {
        node.mChildren = new (pool) Node[2];

        //           P2  upper
        //  +---------+---+
        //  |         |   |
        //  |         |   |
        //  |         |   |
        //  +---------+---+
        // lower      P1

        const float deltaX = extents(node.mBounds, 0);
        const scene_rdl2::math::Vec2f P1(node.mBounds.lower[0] + offset * deltaX, node.mBounds.lower[1]);
        const scene_rdl2::math::Vec2f P2(node.mBounds.lower[0] + offset * deltaX, node.mBounds.upper[1]);

        node.getLeftChild().mBounds  = scene_rdl2::math::BBox2f(node.mBounds.lower, P2);
        node.getRightChild().mBounds = scene_rdl2::math::BBox2f(P1, node.mBounds.upper);
    }

    static void splitVertical(AdaptiveNS::MemoryPool<Node>& pool, Node& node, float offset)
    {
        node.mChildren = new (pool) Node[2];

        //                  upper
        //     +-------------+
        //     |             |
        //     |             |
        //     |             |
        //  P1 +-------------+ P2
        //     |             |
        //     |             |
        //     |             |
        //     |             |
        //     |             |
        //     +-------------+
        //    lower

        const float deltaY = extents(node.mBounds, 1);
        const scene_rdl2::math::Vec2f P1(node.mBounds.lower[0], node.mBounds.lower[1] + offset * deltaY);
        const scene_rdl2::math::Vec2f P2(node.mBounds.upper[0], node.mBounds.lower[1] + offset * deltaY);

        node.getLeftChild().mBounds  = scene_rdl2::math::BBox2f(node.mBounds.lower, P2);
        node.getRightChild().mBounds = scene_rdl2::math::BBox2f(P1, node.mBounds.upper);
    }

    void svgImpl(std::ostream& outs, const Node& node) const;
    scene_rdl2::util::Array2D<int> calcPixelCondition() const; // for debug

    Node mRoot;
    scene_rdl2::math::BBox2i mIntegerRootBounds; // We can get the bounds from the root node, but they're in floats.
    float mTargetError;
    scene_rdl2::util::Array2D<float> mPixelErrors;
    AdaptiveNS::MemoryPool<Node> mNodePool;
    AdaptiveNS::MemoryPool<float> mAccumulatedErrorPool;
};

} // namespace rndr
} // namespace moonray

