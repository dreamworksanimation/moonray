// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Sampler.h"
#include "SamplingConstants.h"
#include "SlidingWindowCache.h"

#include <moonray/common/mcrt_util/StatelessRandomEngine.h>

namespace moonray {
namespace pbr {

// We have to wrap up the array so that we can have an array of these and
// have them all be aligned properly.
struct alignas(kSIMDAlignment) SIMDFloatArray
{
    typedef std::array<float, kSIMDSize> ArrayType;
    typedef ArrayType::value_type value_type;
    typedef ArrayType::pointer pointer;
    typedef ArrayType::size_type size_type;
    ArrayType m;

    constexpr size_type size() const noexcept { return m.size(); }
    float operator[](std::size_t i) const     { return m[i]; }
    float& operator[](std::size_t i)          { return m[i]; }
    float* data() noexcept                    { return m.data(); }
    const float* data() const noexcept        { return m.data(); }
};

template <utype Dims>
class IntegratorSample
{
public:
    // The number of bounces before we fall back on incoherent random numbers
    // for sampling.
    static const unsigned sNumDistributionBounces = 3;

    explicit IntegratorSample(SequenceID sid = SequenceID());
    explicit IntegratorSample(SequenceID sid, utype totalSamples, utype sampleNumber);

    void setUsePseudoRandom(bool state) {  mUsePseudoRandom = state;  }

    // In general, these functions do not have to be called, unless the sample
    // set is being used piecewise, i.e., in a case where a function call
    // continues evaluation where the sampling left off.
    //
    // resume() tells the sampler to continue where it left off
    // (samplesSoFar).
    void resume(SequenceID idx, unsigned samplesSoFar);

    // restart() will tell the sampler to start a new set of samples.
    // Providing totalSamples, the number of samples in the set, may allow the
    // sampler to provide a better distribution.
    void restart(SequenceID idx, unsigned totalSamples);

    void getSample(float returnValue[Dims], unsigned depth) const;

    utype getSampleNumber() const { return mSampleNumber; }

private:
    template <typename F>
    void getSample(float returnValue[Dims], F f, unsigned depth) const;

    template <typename ArrayType>
    static void getIntegratorSamples(const SequenceID& seqid,
                                     utype n,
                                     ArrayType returnValue[Dims],
                                     utype totalsamples)
    {
        // We fill in the arrays kSIMDSize (e.g. 8 for AVX) values at a time, because that many values are "free". The
        // values are cached for later retrieval.
        SampleDispatch<ArrayType, Dims>::fill(seqid, n, returnValue, kSIMDSize, totalsamples);
    }

    template <typename ArrayType>
    static void getIntegratorSamples(const SequenceID& seqid,
                                     utype n,
                                     ArrayType returnValue[Dims])
    {
        // We fill in the arrays kSIMDSize (e.g. 8 for AVX) values at a time, because that many values are "free". The
        // values are cached for later retrieval.
        SampleDispatch<ArrayType, Dims>::fill(seqid, n, returnValue, kSIMDSize);
    }

    bool mUsePseudoRandom;
    utype mTotalSamples;
    mutable utype mSampleNumber;
    SequenceID mSequenceID;
    mutable SIMDFloatArray mCache[Dims];
    mutable SlidingWindowCache mCacheIdx;

    static constexpr utype sInvalidTotalSamples = std::numeric_limits<utype>::max();
};

using IntegratorSample1D = IntegratorSample<1>;
using IntegratorSample2D = IntegratorSample<2>;
using IntegratorSample3D = IntegratorSample<3>;

template <utype Dims>
finline IntegratorSample<Dims>::IntegratorSample(SequenceID sid) :
    mUsePseudoRandom(false),
    mTotalSamples(sInvalidTotalSamples),
    mSampleNumber(0),
    mSequenceID(sid),
    mCacheIdx()
{
}

template <utype Dims>
finline IntegratorSample<Dims>::IntegratorSample(SequenceID sid,
        utype totalSamples, utype sampleNumber) :
    mUsePseudoRandom(false),
    mTotalSamples(totalSamples),
    mSampleNumber(sampleNumber),
    mSequenceID(sid),
    mCacheIdx()
{
}

template <utype Dims>
finline void IntegratorSample<Dims>::resume(
        SequenceID idx, unsigned samplesSoFar)
{
    mTotalSamples = sInvalidTotalSamples;
    mSequenceID = idx;
    mCacheIdx.reset();
    mSampleNumber = samplesSoFar;
}

template <utype Dims>
finline void IntegratorSample<Dims>::restart(
        SequenceID idx, unsigned totalSamples)
{
    mTotalSamples = totalSamples;
    mSequenceID = idx;
    mCacheIdx.reset();
    mSampleNumber = 0;
}

template <utype Dims>
inline void IntegratorSample<Dims>::getSample(float returnValue[Dims], unsigned depth) const
{
    // mTotalSamples gets set based on a constructor overload. In theory, this can
    // be computed at compile-time, if we want to put that kind of work in to
    // remove the conditional.

    if (mUsePseudoRandom  ||  depth >= sNumDistributionBounces) {
        static_assert(Dims <= 4, "These random engines only support up to four values.");
        moonray::util::StatelessRandomEngine reng(mSequenceID.getHash(0xa511e9b3));
        const auto result = reng.asFloat(mSampleNumber++);
        for (utype i = 0; i < Dims; ++i) {
            returnValue[i] = result[i];
        }
    } else if (mTotalSamples != sInvalidTotalSamples) {
        getSample(returnValue, [=](utype n) {
            getIntegratorSamples<SIMDFloatArray>(mSequenceID, n, mCache, mTotalSamples);
        }, depth);
    } else {
        getSample(returnValue, [=](utype n) {
            getIntegratorSamples<SIMDFloatArray>(mSequenceID, n, mCache);
        }, depth);
    }
#if defined(DEBUG)
    for (utype i = 0; i < Dims; ++i) {
        MNRY_ASSERT(returnValue[i] >= 0.0f);
        MNRY_ASSERT(returnValue[i] <  1.0f);
    }
#endif
}

template <utype Dims>
template <typename F>
inline void IntegratorSample<Dims>::getSample(float returnValue[Dims], F f, unsigned depth) const
{
    // Ask the cache to fill up, if needed, and return the index into our
    // array.
    const utype idx = mCacheIdx.getIndex(f, mSampleNumber++);
    for (utype i = 0; i < Dims; ++i) {
        returnValue[i] = mCache[i][idx];
    }
}

} // namespace pbr
} // namespace moonray

