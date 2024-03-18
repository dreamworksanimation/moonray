// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RSequence.h"
#include "Sample.h"
#include "SamplePartition.h"
#include "SamplingPreprocess.h"
#include "SamplingConstants.h"
#include "SequenceID.h"

#include <scene_rdl2/common/math/Math.h>
#include <moonray/common/mcrt_util/StatelessRandomEngine.h>

#include <vector>

// USE_PURE_RANDOM is useful for testing random values without any hashing. This
// should create an unbiased, but poor and unrepeatable, sampling method.
#define USE_PURE_RANDOM 0
#if USE_PURE_RANDOM
#include <scene_rdl2/render/util/Random.h>
#include <thread>
#endif

namespace moonray {
namespace pbr {

const std::size_t kSampleGridSize = 64;
#if defined(USE_PARTITIONED_PIXEL_2D)
typedef SpatialSamplePartition<Sample2D, kSampleGridSize> PixelPartition;
extern PixelPartition kPixelPartition;
#endif

#if defined(USE_PARTITIONED_PIXEL_5D)
typedef SpatialSamplePartition<Sample, kSampleGridSize> PixelPartition;
extern PixelPartition kPixelPartition;
#endif

#if defined(USE_PARTITIONED_LENS)
// sqrt(961) == 31. 31 is prime, and therefore co-prime with 64, which is the
// size of the pixel sample partition. This means that as we tile the pixel
// partition, we're going to mix the pixel samples with the lens samples.  It's
// also very nearly 64/2, which puts us near the center of pixel samples.  This
// makes the next pairing of samples as far away as possible as the tiling
// continues.
//
// Here's an example with two sequences. The first is 16 digits (first number),
// the second is 9 digits (second number). 3 and 4 are coprime, so as we tile
// each sequence individually, we get different combinations (until we get to
// the 12th column (3*4).
// // | 0 0 | 1 1 | 2 2 | 3 0 | 0 1 | 1 2 | 2 0 | 3 1 | 0 2 | 1 0 | 2 1 | 3 2 |
// // | 4 3 | 5 4 | 6 5 | 7 3 | 4 4 | 5 5 | 6 3 | 7 4 | 4 5 | 5 3 | 6 4 | 7 5 |
// // | 8 6 | 9 7 | A 8 | B 6 | 8 7 | 9 8 | A 6 | B 7 | 8 8 | 9 6 | A 7 | B 8 |
// // | C 0 | D 1 | E 2 | F 0 | C 1 | D 2 | E 0 | F 1 | C 2 | D 0 | E 1 | F 2 |
// // | 0 3 | 1 4 | 2 5 | 3 3 | 0 4 | 1 5 | 2 3 | 3 4 | 0 5 | 1 3 | 2 4 | 3 5 |
// // | 4 6 | 5 7 | 6 8 | 7 6 | 4 7 | 5 8 | 6 6 | 7 7 | 4 8 | 5 6 | 6 7 | 7 8 |
// // | 8 0 | 9 1 | A 2 | B 0 | 8 1 | 9 2 | A 0 | B 1 | 8 2 | 9 0 | A 1 | B 2 |
// // | C 3 | D 4 | E 5 | F 3 | C 4 | D 5 | E 3 | F 4 | C 5 | D 3 | E 4 | F 5 |
typedef SamplePartition<Sample2D, 31*31, 1024> LensPartition;
extern LensPartition kLensPartition;
#endif

#if defined(USE_PARTITIONED_TIME)
// sqrt(841) == 29. 29 is prime, and therefore co-prime with 64, which is the
// size of the pixel sample partition. This means that as we tile the pixel
// partition, we're going to mix the pixel samples with the time samples.
typedef SamplePartition<float, 29*29, 1024> TimePartition;
extern TimePartition kTimePartition;
#endif

#if defined(USE_PARTITIONED_1D)
extern const std::vector<float>    k1DSampleTable;
#endif

#if defined(USE_PARTITIONED_2D)
extern const std::vector<Sample2D> k2DSampleTable;
#endif

namespace detail {
// This maps a value (almost) uniformly into a range in [0, n).
template <typename IntType, IntType upper>
finline IntType uniformSelection(IntType n)
{
    static_assert(std::is_integral<IntType>::value,
        "Requires an integer type.");
    static_assert(upper > 0, "Just use zero. Quit bugging me.");

    // This is a little biased...
    return n % upper;
}

template <typename IntType>
constexpr bool isOdd(IntType t)
{
    static_assert(std::is_integral<IntType>::value,
        "Requires an integer type.");
    return (t & 1) == 1;
}


template <typename IntType, IntType v>
struct OddInt
{
    static_assert(std::is_integral<IntType>::value,
        "Requires an integer type.");
    static_assert(isOdd(v), "Requires an odd number.");
    static const IntType value = v;
};

#if USE_PURE_RANDOM
using RandomEngine = util::Random;
inline RandomEngine& getRandomEngine()
{
    static thread_local rng(std::hash<std::thread::id>()(std::this_thread::get_id()));
    return rng;
}
#endif

} // namespace detail

// We want to make sure we're using an odd integer when we're multiplying
// scramble values. We're deliberately overflowing an integer type, so we're
// doing (an implicit) mod operation on 2^n. An odd integer is coprime with
// 2^n, and so we don't get duplicate values through the (implicit) mod
// operation.
#define oddCheck(v) detail::OddInt<decltype(v), (v)>::value


finline void random1D(const SequenceID& seqid, uint32_t n, float* const out, utype nsamples)
{
#if USE_PURE_RANDOM
    auto& rng = detail::getRandomEngine();
    for (utype i = 0; i < nsamples; ++i) {
        out[i] = rng.getNextFloat();
    }
#else
    moonray::util::StatelessRandomEngine reng(seqid.getHash(0xc4837f6d));
    for (utype i = 0; i < nsamples; ++i) {
        const auto result = reng.asFloat(n + i);
        out[i] = result[0];
    }
#endif
}

#if defined(USE_PARTITIONED_1D)
finline void partitioned1D(const SequenceID& seqid, uint32_t n, float* const out, utype nsamples)
{
    constexpr utype kNumSequences = 4096;
    constexpr utype kNumSamplesPerSequence = 1024;

    if (unlikely(n + nsamples >= kNumSamplesPerSequence)) {
        // We use a different sampler when we're going to run out of pre-computed samples.
        // We could partially fill in samples until we run out, but that complicates the logic.
        const uint32_t scramble = seqid.getHash(0xdf4f4915);
        for (utype i = 0; i < nsamples; ++i) {
            const utype sampleNum = n - kNumSamplesPerSequence + i;
            const float sample = jitteredR1(sampleNum, scramble);
            out[i] = sample;
        }
    } else {
        const auto scramble = seqid.getHash(0x0740eb57);
        // If the array size is a power of two, the compiler should mask this
        // instead of modding.
        const auto seqNum = scramble % kNumSequences;

        for (utype i = 0; i < nsamples; ++i) {
            const utype sampleNum = n + i;
            out[i] = k1DSampleTable[seqNum * kNumSamplesPerSequence + sampleNum];
        }
    }
    ispc::PBR_cranleyPattersonRotation(nsamples, out, seqid.getHash(0x5b748587));
}
#endif

finline void random2D(const SequenceID& seqid, uint32_t n, float* const out[2], utype nsamples)
{
#if USE_PURE_RANDOM
    auto& rng = detail::getRandomEngine();
    for (utype i = 0; i < nsamples; ++i) {
        out[0][i] = rng.getNextFloat();
        out[1][i] = rng.getNextFloat();
    }
#else
    moonray::util::StatelessRandomEngine reng(seqid.getHash(0x58127b11));
    for (utype i = 0; i < nsamples; ++i) {
        const auto result = reng.asFloat(n + i);
        out[0][i] = result[0];
        out[1][i] = result[1];
    }
#endif
}

#if defined(USE_PARTITIONED_2D)
finline void partitioned2D(const SequenceID& seqid, uint32_t n, float* const out[2], utype nsamples)
{
    constexpr utype kNumSequences = 4096;
    constexpr utype kNumSamplesPerSequence = 1024;

    if (unlikely(n + nsamples >= kNumSamplesPerSequence)) {
        // We use a different sampler when we're going to run out of pre-computed samples.
        // We could partially fill in samples until we run out, but that complicates the logic.
        const uint32_t scramble = seqid.getHash(0xa81972b7);
        for (utype i = 0; i < nsamples; ++i) {
            const utype sampleNum = n - kNumSamplesPerSequence + i;
            const Sample2D sample = jitteredR2(sampleNum, scramble);
            out[0][i] = sample.u;
            out[1][i] = sample.v;
        }
    } else {
        const auto scramble = seqid.getHash(0x4a770fdf);
        for (utype i = 0; i < nsamples; ++i) {
            const utype sampleNum = n + i;

            // If the array size is a power of two, the compiler should mask this
            // instead of modding.
            const auto seqNum = scramble % kNumSequences;
            const auto& s = k2DSampleTable[seqNum * kNumSamplesPerSequence + sampleNum];
            out[0][i] = s.u;
            out[1][i] = s.v;
        }
    }
    ispc::PBR_cranleyPattersonRotation(nsamples, out[0], seqid.getHash(0xd0944adb));
    ispc::PBR_cranleyPattersonRotation(nsamples, out[1], seqid.getHash(0x0662aa53));
}
#endif

finline void randomPixel(utype pixelWideScramble, int /*x*/, int /*y*/, int /*t*/, utype n, float* valsx, float* valsy)
{
#if USE_PURE_RANDOM
    auto& rng = detail::getRandomEngine();
    for (utype i = 0; i < kSIMDSize; ++i) {
        valsx[i] = rng.getNextFloat();
        valsy[i] = rng.getNextFloat();
    }
#else
    moonray::util::StatelessRandomEngine reng(pixelWideScramble * oddCheck(0x5b4497e3));
    for (utype i = 0; i < kSIMDSize; ++i) {
        const auto result = reng.asFloat(n + i);
        valsx[i] = result[0];
        valsy[i] = result[1];
    }
#endif
}

#if defined(USE_PARTITIONED_PIXEL_FOR_LENS)
finline void partitionedPixelLens(utype pixelWideScramble, int x, int y, int /*t*/, utype n, float* valsx, float* valsy)
{
    static const utype maxSamples = kPixelPartition.numPixelSamples();
    for (utype i = 0; i < kSIMDSize; ++i) {
        const utype sampleNum = n + i;
        if (unlikely(sampleNum >= maxSamples)) {
            moonray::util::StatelessRandomEngine reng(pixelWideScramble);
            const auto result = reng.asFloat(sampleNum);
            valsx[i] = result[0];
            valsy[i] = result[1];
        } else {
            const auto sample = kPixelPartition(x, y, n + i);
            valsx[i] = getPrimaryValue0(sample);
            valsy[i] = getPrimaryValue1(sample);
        }
    }
}
#endif

finline void rotate2D(float& u, float& v, float theta)
{
    const float ucpy = u;
    const float vcpy = v;

    float sinTheta;
    float cosTheta;
    sincosf(theta, &sinTheta, &cosTheta);
    u = ucpy * cosTheta - vcpy * sinTheta;
    v = ucpy * sinTheta + vcpy * cosTheta;
}

#if defined(USE_PARTITIONED_LENS)
finline void partitionedLens(utype pixelWideScramble, int x, int y, int /*t*/, utype n, float* valsu, float* valsv)
{
    for (utype i = 0; i < kSIMDSize; ++i) {
        const utype offsetIndex = n + i;
        if (offsetIndex < LensPartition::kSamplesPerSet) {
            auto sample = kLensPartition(x, y, n + i);
            valsu[i] = sample.u;
            valsv[i] = sample.v;
        } else {
            moonray::util::StatelessRandomEngine reng(pixelWideScramble * oddCheck(0x564e246d));
            const auto result = reng.asFloat(n + i);
            valsu[i] = result[0];
            valsv[i] = result[1];
        }
    }
#if 0
    ispc::PBR_toUnitDisk(kSIMDSize, valsu, valsv);

    // TODO: This won't work with custom bokeh patterns
    for (utype i = 0; i < kSIMDSize; ++i) {
        rotate2D(valsu[i], valsv[i], pixelWideScramble);
    }
#endif    
}
#endif

#if defined(USE_PARTITIONED_TIME)
finline void partitionedTime(utype pixelWideScramble, int x, int y, int /*t*/, utype n, float* valst)
{
    for (utype i = 0; i < kSIMDSize; ++i) {
        const utype offsetIndex = n + i;
        if (offsetIndex < TimePartition::kSamplesPerSet) {
            valst[i] = kTimePartition(x, y, n + i);
        } else {
            moonray::util::StatelessRandomEngine reng(pixelWideScramble * oddCheck(0x564e246d));
            const auto result = reng.asFloat(n + i);
            valst[i] = result[0];
        }
    }
}
#endif

finline void polarToCartesian(float r, float phi, float& x, float& y)
{
    float cosPhi;
    float sinPhi;

    sincosf(phi, &sinPhi, &cosPhi);

    x = r * cosPhi;
    y = r * sinPhi;
}

finline void randomLens(utype pixelWideScramble, int /*x*/, int /*y*/, int /*t*/, utype n, float* valsu, float* valsv)
{
#if USE_PURE_RANDOM
    auto& rng = detail::getRandomEngine();
    for (utype i = 0; i < kSIMDSize; ++i) {
        valsu[i] = rng.getNextFloat();
        valsv[i] = rng.getNextFloat();
    }
#else
    moonray::util::StatelessRandomEngine reng(pixelWideScramble * oddCheck(0x7cc45dc9));
    for (utype i = 0; i < kSIMDSize; ++i) {
        const auto result = reng.asFloat(n + i);
        valsu[i] = result[0];
        valsv[i] = result[1];
    }
#endif
    ispc::PBR_toUnitDisk(kSIMDSize, valsu, valsv);
}

#if defined(USE_PARTITIONED_PIXEL)
inline void partitionedPixel(utype pixelWideScramble, int x, int y, int /*t*/, utype n, float* valsu, float* valsv)
{
    static const utype maxSamples = kPixelPartition.numPixelSamples();
    for (utype i = 0; i < kSIMDSize; ++i) {
        const utype sampleNum = n + i;
        if (unlikely(sampleNum >= maxSamples)) {
            moonray::util::StatelessRandomEngine reng(pixelWideScramble);
            const auto result = reng.asFloat(sampleNum);
            valsu[i] = result[0];
            valsv[i] = result[1];
        } else {
            const auto sample = kPixelPartition(x, y, sampleNum);
            valsu[i] = getPrimaryValue0(sample);
            valsv[i] = getPrimaryValue1(sample);
        }
    }
}
#endif

finline void randomTime(utype pixelWideScramble, int /*x*/, int /*y*/, int /*t*/, utype n, float* valst)
{
#if USE_PURE_RANDOM
    auto& rng = detail::getRandomEngine();
    for (utype i = 0; i < kSIMDSize; ++i) {
        valst[i] = rng.getNextFloat();
    }
#else
    moonray::util::StatelessRandomEngine reng(pixelWideScramble * oddCheck(0x7cc45dc9));
    for (utype i = 0; i < kSIMDSize; ++i) {
        const auto result = reng.asFloat(n + i);
        valst[i] = result[0];
    }
#endif
}

#if defined(USE_PARTITIONED_PIXEL_FOR_TIME)
finline void partitionedPixelTime(utype pixelWideScramble, int x, int y, int /*t*/, utype n, float* valst)
{
    static const utype maxSamples = kPixelPartition.numPixelSamples();
    for (utype i = 0; i < kSIMDSize; ++i) {
        const utype sampleNum = n + i;
        if (unlikely(sampleNum >= maxSamples)) {
            moonray::util::StatelessRandomEngine reng(pixelWideScramble);
            const auto result = reng.asFloat(sampleNum);
            valst[i] = result[0];
        } else {
            const auto sample = kPixelPartition(x, y, n + i);
            valst[i] = sample.time;
        }
    }
}
#endif

} // namespace pbr
} // namespace moonray

