// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "SamplingConstants.h"

#include <limits>

namespace moonray {
namespace pbr {

// This class will return the next zero-based index into an array assumed to be
// kSIMDSize in length.
//
// When necessary, this class will calculate a contiguous subset of values, of
// length kSIMDSize, through the function passed in.
//
// The passed in function must take an unsigned value, n, as the starting
// value, and compute all values in [n, n+kSIMDSize).
//
// E.g. If we ask for the 64th value in the sequence, the class may call f(64)
// and return 0 as the index into an array. The 65th value may not call the
// function, but the class will return 1.
class SlidingWindowCache
{
    utype mCachedMin;
    static const utype sDefaultState = std::numeric_limits<utype>::max();

public:
    SlidingWindowCache();
    void reset();

    template <typename CacheCalc>
    utype getIndex(CacheCalc cacheCalc, utype incrementalValue);
};

finline SlidingWindowCache::SlidingWindowCache() :
    mCachedMin(sDefaultState)
{
}

finline void SlidingWindowCache::reset()
{
    mCachedMin = sDefaultState;
}

template <typename CacheCalc>
finline utype SlidingWindowCache::getIndex(CacheCalc cacheCalc, utype incrementalValue)
{
    const utype cachedMax = mCachedMin + kSIMDSize;
    if (incrementalValue < mCachedMin || incrementalValue >= cachedMax) {
        cacheCalc(incrementalValue);
        mCachedMin = incrementalValue;
    }

    return incrementalValue - mCachedMin;
}

} // namespace pbr
} // namespace moonray

