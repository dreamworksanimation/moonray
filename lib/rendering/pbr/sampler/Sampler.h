// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "PixelFilter.h"
#include "PixelScramble.h"
#include "Sample.h"
#include "SamplingAlgorithms.h"
#include "SamplingConstants.h"
#include "SequenceID.h"
#include "SlidingWindowCache.h"

#include <moonray/rendering/pbr/sampler/Sampling_ispc_stubs.h>
#include <scene_rdl2/render/util/BitUtils.h>

#include <array>
#include <numeric>
#include <type_traits>

#include <cstdint>

namespace moonray {
namespace pbr {

namespace detail {

// A raw pointer. Return yourself!
template <typename T>
finline T toPointer(T t, std::true_type /* Is pointer == true */)
{
    return t;
}

// Some sort of container (std::array, std::vector). Return your underlying data!
template <typename T>
finline typename T::pointer toPointer(T& t, std::false_type /* Is pointer == false */)
{
    return t.data();
}

// TODO C++14: we should be able to remove this ugly decltype.
template <typename T>
finline auto toPointer(T& t) -> decltype(detail::toPointer(t, typename std::is_pointer<T>::type()))
{
    return detail::toPointer(t, typename std::is_pointer<T>::type());
}
}

// "Correlated Multi-Jittered Sampling"
// Andrew Kensler
// Pixar Technical Memo 13-01, 2013
finline float randfloat(utype i, utype p)
{
    i ^= p;
    i ^= i >> 17;
    i ^= i >> 10;    i *= 0xb36534e5;
    i ^= i >> 12;
    i ^= i >> 21;    i *= 0x93fc4795;
    i ^= 0xdf6e307f;
    i ^= i >> 17;    i *= 1 | p >> 18;
    return scene_rdl2::util::bitsToFloat(i);
}

template <typename ArrayType, std::size_t dimension>
struct SampleDispatch;

template <typename ArrayType>
struct SampleDispatch<ArrayType, 1>
{
    static finline void fill(const SequenceID& seqid, utype n, ArrayType ret[1], utype nsamples)
    {
        MNRY_ASSERT(ret[0].size() >= nsamples);
        float* p = detail::toPointer(ret[0]);
        MNRY_ASSERT(scene_rdl2::alloc::isAligned(p, kSIMDAlignment));

#if defined(USE_PARTITIONED_1D)
        partitioned1D(seqid, n, p, nsamples);
#elif defined(USE_RANDOM_1D)
        random1D(seqid, n, p, nsamples);
#else
#error No 1D integrator sampler defined
#endif
    }

    static finline void fill(const SequenceID& seqid, utype n, ArrayType ret[1], utype nsamples, utype /*totalsamples*/)
    {
        MNRY_ASSERT(ret[0].size() >= nsamples);
        SampleDispatch<ArrayType, 1>::fill(seqid, n, ret, nsamples);
    }
};

template <typename ArrayType>
struct SampleDispatch<ArrayType, 2>
{
    static finline void fill(const SequenceID& seqid, utype n, ArrayType ret[2], utype nsamples)
    {
        MNRY_ASSERT(ret[0].size() >= nsamples);
        MNRY_ASSERT(ret[1].size() >= nsamples);
#if defined(USE_PARTITIONED_2D)
        float* p[2] = { detail::toPointer(ret[0]), detail::toPointer(ret[1]) };
        MNRY_ASSERT(scene_rdl2::alloc::isAligned(p[0], kSIMDAlignment));
        MNRY_ASSERT(scene_rdl2::alloc::isAligned(p[1], kSIMDAlignment));
        partitioned2D(seqid, n, p, nsamples);
#elif defined(USE_RANDOM_2D)
        float* p[2] = { detail::toPointer(ret[0]), detail::toPointer(ret[1]) };
        MNRY_ASSERT(scene_rdl2::alloc::isAligned(p[0], kSIMDAlignment));
        MNRY_ASSERT(scene_rdl2::alloc::isAligned(p[1], kSIMDAlignment));
        random2D(seqid, n, p, nsamples);
#else
#error No 2D integrator sampler defined
#endif
    }

    static finline void fill(const SequenceID& seqid, utype n, ArrayType ret[2], utype nsamples, utype totalsamples)
    {
        MNRY_ASSERT(ret[0].size() >= nsamples);
        MNRY_ASSERT(ret[1].size() >= nsamples);
        if (totalsamples == 1) {
            float* p[2] = { detail::toPointer(ret[0]), detail::toPointer(ret[1]) };
            // The sequence id determines the scrambling of this sequence or set, but we get the hash through a completely
            // randomly-chosen seed so that we don't accidentally line up with other fill calls.
            p[0][0] = randfloat(0, seqid.getHash(0xf2b143c3));
            p[1][0] = randfloat(1, seqid.getHash(0x1d3cb933));
        } else {
            float* p[2] = { detail::toPointer(ret[0]), detail::toPointer(ret[1]) };
            MNRY_ASSERT(scene_rdl2::alloc::isAligned(p[0], kSIMDAlignment));
            MNRY_ASSERT(scene_rdl2::alloc::isAligned(p[1], kSIMDAlignment));
            // The sequence id determines the scrambling of this sequence or set, but we get the hash through a completely
            // randomly-chosen seed so that we don't accidentally line up with other fill calls.
            const int scramble[1] = { static_cast<int>(seqid.getHash(0x2904c44e)) };
            ispc::PBR_correlatedMultiJitter2D(nsamples, p[0], p[1], totalsamples, n, scramble[0]);
        }
    }
};

template <typename ArrayType>
struct SampleDispatch<ArrayType, 3>
{
    // cppcheck-suppress arrayIndexOutOfBounds // cppcheck thinks this is still an array of two from above
    static finline void fill(const SequenceID& seqid, utype n, ArrayType ret[3], utype nsamples)
    {
        MNRY_ASSERT(ret[0].size() >= nsamples);
        MNRY_ASSERT(ret[1].size() >= nsamples);
        MNRY_ASSERT(ret[2].size() >= nsamples);
        // TODO: 3D sample points
        SampleDispatch<ArrayType, 2>::fill(seqid, n, ret + 0, nsamples);
        SampleDispatch<ArrayType, 1>::fill(seqid, n, ret + 2, nsamples);
    }

    // cppcheck-suppress arrayIndexOutOfBounds // cppcheck thinks this is still an array of two from above
    static finline void fill(const SequenceID& seqid, utype n, ArrayType ret[3], utype nsamples, utype totalsamples)
    {
        MNRY_ASSERT(ret[0].size() >= nsamples);
        MNRY_ASSERT(ret[1].size() >= nsamples);
        MNRY_ASSERT(ret[2].size() >= nsamples);
        // TODO: 3D sample points
        SampleDispatch<ArrayType, 3>::fill(seqid, n, ret, nsamples);
    }
};

/* The non-grid mode pixel sampler. */
inline void fillPixelSamples(utype pixelWideScramble, int x, int y, int t, utype n, float* valsx, float* valsy)
{
    MNRY_ASSERT(scene_rdl2::alloc::isAligned(valsx, kSIMDAlignment));
    MNRY_ASSERT(scene_rdl2::alloc::isAligned(valsy, kSIMDAlignment));

#if defined(USE_PARTITIONED_PIXEL)
    partitionedPixel(pixelWideScramble, x, y, t, n, valsx, valsy);
#elif defined(USE_RANDOM_PIXEL)
    randomPixel(pixelWideScramble, x, y, t, n, valsx, valsy);
#else
#error No pixel sampler defined
#endif
}

/* Simple jitter within the pixel.  Assumes a 1x1 subpixel grid, which
 * is the case when "pixel samples" = 1.  The subpixel
 * grid resolution is lowered when the number of pixel samples is less than 64
 * so that all of the subpixels are still populated.
 */
finline void fillPixelSamples1x1(utype pixelWideScramble, int x, int y, int t, utype n,
                                 float* valsx, float* valsy)
{

    MNRY_ASSERT(scene_rdl2::alloc::isAligned(valsx, kSIMDAlignment));
    MNRY_ASSERT(scene_rdl2::alloc::isAligned(valsy, kSIMDAlignment));

    moonray::util::StatelessRandomEngine reng(pixelWideScramble * oddCheck(0x5b4497e3));

    // 1x1 grid samples, jittered
    for (utype i = 0; i < kSIMDSize; ++i) {
        const auto jitter = reng.asFloat(n + i);
        valsx[i] = jitter[0];
        valsy[i] = jitter[1];
    }
}

/*
 * Jittered sampling on a 2x2 subpixel grid.  Note the order2x2 array:
 * this specifies the order of subpixel sampling, which is basically a shuffled
 * order.  Each pixel uses its pixelWideScramble to index into this order so that
 * adjacent pixels sample subpixels in a different order.  For "pixel samples" = 2
 * this doesn't really matter as each subpixel is still sampled once, BUT for the
 * "pixel samples" = 3 case where we have 9 samples to put into 4 subpixels,
 * it becomes important that the distribution of these 9 samples is different
 * for each pixel to keep the sampling as uniform as possible.
 *
 * E.g. 9 samples case:
 * pixel 0 subpixel order: 0, 2, 3, 1, 0, 2, 3, 1, 0
 * pixel 1 subpixel order: 2, 3, 1, 0, 2, 3, 1, 0, 2
 * pixel 2 subpixel order: 3, 1, 0, 2, 3, 1, 0, 2, 3
 * Note that subpixel 3 has been sampled 3 times for pixel 0, but 2 times for the
 * other pixels.  Shuffling the order like this avoids the problem that subpixel
 * 0 *always* has 3 samples but the other subpixels only have 2... that would be
 * an uneven sampling distribution.  Instead, the "3 samples" subpixel gets moved
 * around between pixels.  This is also important when we have "pixel samples" > 8
 * which means that the 8x8 subpixel grid has more than 1 sample per subpixel, but
 * not uniformly so.  E.g. "pixel samples" = 9 is 9*9=81 samples, which cannot be
 * uniformly distributed in an 8x8 grid.
 *
 * Why is there no 3x3 subpixels case where we wouldn't have this problem?
 * There is no way to easily reduce an 8x8 subpixel grid to 3x3.  We can
 * only reduce 8x8 to 4x4, 2x2, or 1x1.
 */

const unsigned char order2x2[4] = {
    0, 2, 3, 1
};

finline void fillPixelSamples2x2(utype pixelWideScramble, int x, int y, int t, utype n,
                                 float* valsx, float* valsy)
{
    MNRY_ASSERT(scene_rdl2::alloc::isAligned(valsx, kSIMDAlignment));
    MNRY_ASSERT(scene_rdl2::alloc::isAligned(valsy, kSIMDAlignment));

    moonray::util::StatelessRandomEngine reng(pixelWideScramble * oddCheck(0x5b4497e3));

    // 2x2 grid samples, jittered
    for (utype i = 0; i < kSIMDSize; ++i) {
        utype sampleN = order2x2[((pixelWideScramble & 0x03) + n + i) & 0x03];
        utype sx = sampleN % 2;
        utype sy = sampleN / 2;
        const auto jitter = reng.asFloat(n + i);
        valsx[i] = (sx + jitter[0]) * 0.5f;
        valsy[i] = (sy + jitter[1]) * 0.5f;
    }
}

const unsigned char order4x4[16] = {
    7, 11,  1,  4,
    9,  5,  0, 13,
    3,  8, 14, 12,
   10,  2, 15,  6
};

finline void fillPixelSamples4x4(utype pixelWideScramble, int x, int y, int t, utype n,
                                 float* valsx, float* valsy)
{
    MNRY_ASSERT(scene_rdl2::alloc::isAligned(valsx, kSIMDAlignment));
    MNRY_ASSERT(scene_rdl2::alloc::isAligned(valsy, kSIMDAlignment));

    moonray::util::StatelessRandomEngine reng(pixelWideScramble * oddCheck(0x5b4497e3));

    // 4x4 grid samples, jittered
    for (utype i = 0; i < kSIMDSize; ++i) {
        utype sampleN = order4x4[((pixelWideScramble & 0x0f) + n + i) & 0x0f];
        utype sx = sampleN % 4;
        utype sy = sampleN / 4;
        const auto jitter = reng.asFloat(n + i);
        valsx[i] = (sx + jitter[0]) * 0.25f;
        valsy[i] = (sy + jitter[1]) * 0.25f;
    }
}

const unsigned char order8x8[64] = {
    33, 26, 38, 10, 22, 58, 14, 17,
     9, 47, 50, 15, 59, 36, 54,  2,
     3,  4, 57,  6, 24, 55, 35, 12,
     8, 11, 31, 60, 25, 20, 63, 62,
    52, 53, 18, 61, 13, 23, 21, 29,
    37,  5,  0, 34, 46, 44, 56, 43,
    42, 28, 48, 45, 27, 49, 16, 19,
    32, 30,  1, 40, 51,  7, 41, 39
};

finline void fillPixelSamples8x8(utype pixelWideScramble, int x, int y, int t, utype n,
                                 float* valsx, float* valsy)
{
    MNRY_ASSERT(scene_rdl2::alloc::isAligned(valsx, kSIMDAlignment));
    MNRY_ASSERT(scene_rdl2::alloc::isAligned(valsy, kSIMDAlignment));

    moonray::util::StatelessRandomEngine reng(pixelWideScramble * oddCheck(0x5b4497e3));

    // 8x8 grid samples, jittered
    for (utype i = 0; i < kSIMDSize; ++i) {
        utype sampleN = order8x8[((pixelWideScramble & 0x3f) + n + i) & 0x3f];
        utype sx = sampleN % 8;
        utype sy = sampleN / 8;
        const auto jitter = reng.asFloat(n + i);
        valsx[i] = (sx + jitter[0]) * 0.125f;
        valsy[i] = (sy + jitter[1]) * 0.125f;
    }
}

finline void fillLensSamples(utype pixelWideScramble, int x, int y, int t, utype n, float* valsu, float* valsv)
{
    MNRY_ASSERT(scene_rdl2::alloc::isAligned(valsu, kSIMDAlignment));
    MNRY_ASSERT(scene_rdl2::alloc::isAligned(valsv, kSIMDAlignment));

#if defined(USE_DISK_ARRAY_LENS)
    diskArrayLens(pixelWideScramble, x, y, t, n, valsu, valsv);
#elif defined(USE_RANDOM_LENS)
    randomLens(pixelWideScramble, x, y, t, n, valsu, valsv);
#elif defined(USE_PARTITIONED_PIXEL_FOR_LENS)
    partitionedPixelLens(pixelWideScramble, x, y, t, n, valsu, valsv);
#elif defined(USE_PARTITIONED_LENS)
    partitionedLens(pixelWideScramble, x, y, t, n, valsu, valsv);
#else
#error No lens sampler defined
#endif
}

finline void fillTimeSamples(utype pixelWideScramble, int x, int y, int t, utype n, float* valst)
{
    MNRY_ASSERT(scene_rdl2::alloc::isAligned(valst, kSIMDAlignment));
#if defined(USE_PARTITIONED_PIXEL_FOR_TIME)
    partitionedPixelTime(pixelWideScramble, x, y, t, n, valst);
#elif defined(USE_RANDOM_TIME)
    randomTime(pixelWideScramble, x, y, t, n, valst);
#elif defined(USE_PARTITIONED_TIME)
    partitionedTime(pixelWideScramble, x, y, t, n, valst);
#else
#error No time sampler defined
#endif
}

///
/// @class Sampler Sampler.h <rendering/pbr/sampler/Sampler.h>
/// @brief This class is responsible for generating pixel/lens/time, and
/// 1D and 2D integrator samples.  The sampler runs in two different modes,
/// depending on whether 8x8 mask deep images are being output.  If *any*
/// 8x8 mask deep images are being output, the sampler runs in 8x8 grid mode
/// for *ALL* outputs.
///
/// 8x8 grid mode performs *unfiltered jittered sampling* on an 8x8 subpixel grid
/// so that all of the 8x8 subpixels are populated.  "Unfiltered" sampling means
/// that all rays have equal contribution (actually uniform distribution: see
/// note 1) to the pixel's radiance, which is effectively a box filter.
/// It is assumed that filtering will be applied during compositing of the
/// images.  Note that lens and time samples are not changed in 8x8 grid mode, just
/// the pixel samples.
///
/// Subpixels in the 8x8 pixel grid may have multiple jittered samples if the
/// pixel sampling settings call for more than 64 samples ("pixel samples" = 8).
/// If there are fewer than 64 samples, then the samples are jittered on a 1x1,
/// 2x2, or 4x4 grid, depending on the setting.
///
/// It is not possible to mix samplers (e.g. normal and 8x8 grid) because all
/// render outputs for a pass share the same sampler in renderTiles().  Allowing
/// this would require the renderer to perform multiple passes, at a large
/// performance cost.
///
/// * note 1: Pixel filtering in non-grid mode is handled by altering the
/// distribution of the rays within the 2d screen space, thus applying weighting.
/// The individual rays' radiance contributions are not scaled.  The rays may be
/// moved outside the boundaries of the current pixel to collect radiance
/// contributions from adjacent pixels, which increases the filter's footprint.
/// Note that negative filter lobes are not possible with this technique.
class Sampler
{
public:
    // Empty constructor just for reserving space on the stack
    Sampler() {}

    explicit Sampler(const PixelScramble& scramble,
                     const PixelFilter* pixelFilter,
                     bool use8x8Grid,
                     int numSamples) :
        mPixelWideScramble(scramble),
        mPixelFilter(pixelFilter),
        mCurrentIdx(),
        mUse8x8Grid(use8x8Grid),
        mNumSamples(numSamples)
    {
        // This gets vectorized in ISPC.
        std::fill(mLensU.begin(), mLensU.end(), 0.0f);
        std::fill(mLensV.begin(), mLensV.end(), 0.0f);
    }

    Sampler(const Sampler&) = delete;

    // Named return-value optimization (NVRO) means we don't pay for returning
    // this by value.
    Sample getPrimarySample(int x, int y, int t, uint32_t subpixel, bool dofEnabled, float shutterBias)
    {
        // Check the cache and potentially fill it with primary samples.
        const utype idx = mCurrentIdx.getIndex([=](utype n) {
            fillPrimary(x, y, t, n, dofEnabled);
        }, subpixel);

        Sample sample;
        sample.pixelX = mPixelX[idx];
        sample.pixelY = mPixelY[idx];
        sample.lensU  = mLensU[idx];
        sample.lensV  = mLensV[idx];

        if (shutterBias == 0) {
            sample.time = mTime[idx];
        } else if (shutterBias > 0) {
            sample.time = scene_rdl2::math::pow(mTime[idx], 1.0f / (1.0f + shutterBias));
            // Numeric precision issues can make sample.time == 1.0f, even if the original value is strictly less than one.
            sample.time = std::min(sample.time, kMaxLessThanOne);
        } else {
            sample.time = 1 - scene_rdl2::math::pow(1 - mTime[idx], 1.0f / (1.0f - shutterBias));
        }

        return sample;
    }

private:
    PixelScramble mPixelWideScramble;

    void applyPixelFilter()
    {
        mPixelFilter->apply(kSIMDSize, mPixelX.data());
        mPixelFilter->apply(kSIMDSize, mPixelY.data());
    }

    // Cache pixel/lens/time samples.
    void fillPrimary(int x, int y, int t, utype n, bool dofEnabled)
    {
        const std::uint32_t seed = mPixelWideScramble.getSeed();
        if (mUse8x8Grid) {
            // Sample to populate the 8x8 (or 4x4, 2x2, 1x1)
            //  subpixel grid... Note that time and lens samples are not affected.
            // We also need to pick the largest subpixel resolution that will be fully
            //  populated by the number of samples such that there are no unsampled subpixels.
            if (mNumSamples >= 64) {
                fillPixelSamples8x8(seed, x, y, t, n, mPixelX.data(), mPixelY.data());
            } else if (mNumSamples >= 16) {
                fillPixelSamples4x4(seed, x, y, t, n, mPixelX.data(), mPixelY.data());
            } else if (mNumSamples >= 4) {
                fillPixelSamples2x2(seed, x, y, t, n, mPixelX.data(), mPixelY.data());
            } else {
                fillPixelSamples1x1(seed, x, y, t, n, mPixelX.data(), mPixelY.data());
            }
        } else {
            fillPixelSamples(seed, x, y, t, n, mPixelX.data(), mPixelY.data());
        }
        fillTimeSamples(seed, x, y, t, n, mTime.data());
        if (dofEnabled) {
            fillLensSamples(seed, x, y, t, n, mLensU.data(), mLensV.data());
        }

        if (!mUse8x8Grid) {
            // only move the samples around in the 2d image space if we're NOT in
            //  8x8 grid mode
            applyPixelFilter();
        }
    }

    const PixelFilter* mPixelFilter;
    SlidingWindowCache mCurrentIdx;
    bool mUse8x8Grid;
    int mNumSamples;

    typedef std::array<float, kSIMDSize> SIMDFloatArray;

    __attribute__((aligned(static_cast<int>(kSIMDAlignment)))) SIMDFloatArray mPixelX;
    __attribute__((aligned(static_cast<int>(kSIMDAlignment)))) SIMDFloatArray mPixelY;
    __attribute__((aligned(static_cast<int>(kSIMDAlignment)))) SIMDFloatArray mLensU;
    __attribute__((aligned(static_cast<int>(kSIMDAlignment)))) SIMDFloatArray mLensV;
    __attribute__((aligned(static_cast<int>(kSIMDAlignment)))) SIMDFloatArray mTime;

#ifndef __clang__
    static_assert(__alignof(mPixelX) == kSIMDAlignment, "SIMD alignment.");
    static_assert(__alignof(mPixelY) == kSIMDAlignment, "SIMD alignment.");
    static_assert(__alignof(mLensU)  == kSIMDAlignment, "SIMD alignment.");
    static_assert(__alignof(mLensV)  == kSIMDAlignment, "SIMD alignment.");
    static_assert(__alignof(mTime)   == kSIMDAlignment, "SIMD alignment.");
#endif
};

} // namespace pbr
} // namespace moonray

