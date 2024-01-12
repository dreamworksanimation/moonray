// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Sample.h"
#include <scene_rdl2/render/util/AlignedAllocator.h>
#include <scene_rdl2/render/util/Random.h>
#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/render/util/BitUtils.h>
#include <scene_rdl2/render/util/SManip.h>
#include <scene_rdl2/common/math/MathUtil.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>
#include <random>

namespace moonray {
namespace pbr {

namespace sp {
using size_type = std::int32_t;
}

namespace aux {
template <typename Container>
std::ostream& print(std::ostream& outs,
                    const Container& c,
                    sp::size_type dim,
                    sp::size_type depth);
} // namespace aux

constexpr int largestPrimeLessThanOrEqual(int n)
{
    return scene_rdl2::math::compile_time::isPrime(n) ? n : largestPrimeLessThanOrEqual(n - 1);
}

///
/// @class SpatialSamplePartition
/// @brief Spatially divides samples in hyper-torus [0,1]^n space into
/// kDimension*kDimension strata.
///
/// This class has the constraint that the width == height == 2**m, for some m.
/// This allows for efficient constant-time (O(1)) access to the pixel samples
/// for a given sample.
///
/// The class will wrap the samples automatically.
///
/// On build, the points are guaranteed to be added in the order specified
/// (stable), so that accessing the points will be in the order specified. This
/// allows for progressive sampling, as long as the samples passed in are in the
/// order generated.
///
///////////////////////////////////////////////////////////////////////
//
// Rotation scheme:
//
// Assume each stratum is numbered as below:
//
// Initial grid:
//    +-----------+
//    |01 02 03 04|
//    |           |
//    |05 06 07 08|
//    |           |
//    |09 10 11 12|
//    |           |
//    |13 14 15 16|
//    +-----------+
//
// To rotate as a torus, we rotate the rows and columns independently.
//
// Rotate of columns:
//    +-----------+
//    |03 04 01 02|
//    |           |
//    |07 08 05 06|
//    |           |
//    |11 12 09 10|
//    |           |
//    |15 16 13 14|
//    +-----------+
//
// Rotate of rows:
//    +-----------+
//    |11 12 09 10|
//    |           |
//    |15 16 13 14|
//    |           |
//    |03 04 01 02|
//    |           |
//    |07 08 05 06|
//    +-----------+
//
// Instead of keeping track of each stratum number independently, we just need
// the number of rotations for the rows and the number of rotations for the
// columns, and the rest can be extrapolated. The rotation amount will always
// be mod the width or the height: we just have to offset the x and y by the
// rotation amount when figuring out the strata values.
//
template <typename T, sp::size_type kDimension>
class SpatialSamplePartition
{
public:
    static_assert(scene_rdl2::util::isPowerOfTwo(kDimension),
            "The partition dimension must be a power of 2.");
    using value_type = T;
    using size_type  = sp::size_type;

    template <typename Iter>
    SpatialSamplePartition(Iter first, Iter last);

    const T& operator()(size_type pixelX, size_type pixelY, size_type n) const;
    void rotate(size_type n);
    size_type numPixelSamples() const;

    std::ostream& print(std::ostream& outs) const;

private:
    // We use a flat arry of memory for samples. We have kDimension*kDimension
    // strata, and a run-time number of points per sample, n.
    // 
    // The first stratum is at index 0. The second stratum starts at index n.
    // The third at 2*n...
    //
    // The strata are numbered according to 2D array access, the index found by
    // getStratum().
    using Container = std::vector<T, scene_rdl2::alloc::AlignedAllocator<T, kSIMDAlignment>>;
    using BuildInfo = std::pair<T, size_type>;

    static size_type dimensionMod(size_type n);
    static size_type getStratum(size_type x, size_type y);
    static BuildInfo build(const T& t);

    template <typename RandIter>
    static Container build(RandIter first, RandIter last, std::random_access_iterator_tag);
    template <typename InputIter>
    static Container build(InputIter first, InputIter last, std::input_iterator_tag);

    // Rotation amounts for torus rotation in u and v.
    size_type mURotation;
    size_type mVRotation;
    Container mData;
};

///
/// @class SamplePartition
/// @brief Partitions samples into 'sets' number of sets, each set having
/// 'samplesPerSet' samples. 'sets' must be a perfect square, but there are no
/// other constraints on the values, although frame-to-frame rotation will work
/// best if 'sets' is a power of 2 or a prime number.
///
/// On build, the points are added in the order they appear in the input, one
/// set at a time, until that set is full.
///
template <typename T, sp::size_type sets, sp::size_type samplesPerSet>
class SamplePartition
{
public:
    using value_type = T;
    using size_type  = sp::size_type;

    static const size_type kSamplesPerSet = samplesPerSet;

    template <typename Iter>
    SamplePartition(Iter first, Iter last);

    T operator()(size_type pixelX, size_type pixelY, size_type n) const;
    void rotate(size_type n);

    std::ostream& print(std::ostream& outs) const;

private:
    // We use a flat array of memory for samples. We have "sets" number of
    // strata, and "samplesPerSet" points per stratum.
    // 
    // The first stratum is at index 0. The second stratum starts at index
    // samplesPerSet.  The third at 2*samplesPerSet...
    //
    // The strata are numbered according to 2D array access, the index found by
    // getStratum().
    using Container = std::vector<T, scene_rdl2::alloc::AlignedAllocator<T, kSIMDAlignment>>;

    template <typename Point>
    static void rotateMirror(Point& p, size_type which);
    static void rotateMirror(float& f, size_type which);

    static size_type dimensionMod(size_type n);
    static size_type getStratum(size_type x, size_type y);

    // Rotation amounts for torus rotation in u and v. However, as the strata
    // are independent, we could probably make this rotation more general and
    // achieve more permutations.
    size_type mURotation;
    size_type mVRotation;
    Container mData;
};

//
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//

namespace {
int sOutputDepth = std::ios_base::xalloc();
}

// This I/O manipulator sets the number of points per pixel to output.
inline std::ios_base& setDepth(std::ios_base& os, long output)
{
    os.iword(sOutputDepth) = output;
    return os;
}

inline scene_rdl2::util::SManip<long> depth(long n)
{
    return scene_rdl2::util::SManip<long>(&setDepth, n);
}

//
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//

template <typename T, sp::size_type kDimension>
finline sp::size_type SpatialSamplePartition<T, kDimension>::dimensionMod(size_type n)
{
    static_assert(scene_rdl2::util::isPowerOfTwo(kDimension),
            "The partition dimension must be a power of 2.");

    // Modulus operator with power of two dimension.
    constexpr size_type mask = kDimension - 1u;
    return n & mask;
}

template <typename T, sp::size_type kDimension>
finline sp::size_type SpatialSamplePartition<T, kDimension>::getStratum(size_type x, size_type y)
{
    static_assert(scene_rdl2::util::isPowerOfTwo(kDimension),
            "The partition dimension must be a power of 2.");

    static constexpr size_type sShift = scene_rdl2::math::compile_time::log2i(kDimension);

    // Precondition: x and y have been modded to range, so that we can wrap our
    // values.
    MNRY_ASSERT(x < kDimension);
    MNRY_ASSERT(y < kDimension);

    // This is analogous to the usual 2D indexing scheme, were we would take
    // y * width + x, but with nice power of 2 math!
    return ((y << sShift) | x);
}

template <typename T, sp::size_type kDimension>
finline const T& SpatialSamplePartition<T, kDimension>::operator()(size_type pixelX, size_type pixelY, size_type n) const
{
    pixelX = dimensionMod(pixelX + mURotation);
    pixelY = dimensionMod(pixelY + mVRotation);
    return mData[getStratum(pixelX, pixelY) * numPixelSamples() + n];
}

// Precondition: T is in [0, 1)
template <typename T, sp::size_type kDimension>
finline typename SpatialSamplePartition<T, kDimension>::BuildInfo SpatialSamplePartition<T, kDimension>::build(const T& t)
{
    MNRY_ASSERT(getPrimaryValue0(t) >= 0.0f && getPrimaryValue0(t) < 1.0f);
    MNRY_ASSERT(getPrimaryValue1(t) >= 0.0f && getPrimaryValue1(t) < 1.0f);

    const size_type uOffset = getPrimaryValue0(t) * kDimension;
    const size_type vOffset = getPrimaryValue1(t) * kDimension;
    const auto stratum = getStratum(uOffset, vOffset);

    // Scale the sub-samples for this pixel to be in [0, 1).
    // This is essentially a modf. We're just subtracting the integral portion
    // off of what we've already calculated.
    T scaledT(t);
    getPrimaryValue0(scaledT) = getPrimaryValue0(t) * static_cast<float>(kDimension) - uOffset;
    getPrimaryValue1(scaledT) = getPrimaryValue1(t) * static_cast<float>(kDimension) - vOffset;

    MNRY_ASSERT(getPrimaryValue0(scaledT) >= 0.0f && getPrimaryValue0(scaledT) < 1.0f);
    MNRY_ASSERT(getPrimaryValue1(scaledT) >= 0.0f && getPrimaryValue1(scaledT) < 1.0f);

    return std::make_pair(scaledT, stratum);
}

template <typename T, sp::size_type kDimension>
template <typename RandIter>
finline typename SpatialSamplePartition<T, kDimension>::Container SpatialSamplePartition<T, kDimension>::build(RandIter first, RandIter last, std::random_access_iterator_tag)
{
    const size_type numPoints = std::distance(first, last);
    const size_type numStrata = kDimension * kDimension;
    const size_type pointsPerStratum = scene_rdl2::util::roundUpToPowerOfTwo(numPoints / numStrata);

    std::vector<int> count(numStrata);
    Container data(numStrata * pointsPerStratum);

    for ( ; first != last; ++first) {
        const auto bi = build(*first);
        const auto stratum = bi.second;
        if (count[stratum] < pointsPerStratum) {
            data[stratum * pointsPerStratum + count[stratum]] = bi.first;
            ++count[stratum];
        }
    }

    // Fill any un-filled strata with random point values so they're all the
    // same size.
    scene_rdl2::util::Random re(0x9084f091);
    auto canon = [&re] () {
        return re.getNextFloat();
    };
    for (size_type i = 0; i < numStrata; ++i) {
        while (count[i] < pointsPerStratum) {
            data[i * pointsPerStratum + count[i]] = generateRandomPoint<T>(canon);
            ++count[i];
        }
    }

    MNRY_ASSERT(std::all_of(count.cbegin(), count.cend(), [=](int x)
    {
        return x == pointsPerStratum;
    }));

    return data;
}

template <typename T, sp::size_type kDimension>
template <typename InputIter>
finline typename SpatialSamplePartition<T, kDimension>::Container SpatialSamplePartition<T, kDimension>::build(InputIter first, InputIter last, std::input_iterator_tag)
{
    const std::vector<T> temp(first, last);
    return build(temp.cbegin(), temp.cend(), std::random_access_iterator_tag());
}

template <typename T, sp::size_type kDimension>
template <typename Iter>
SpatialSamplePartition<T, kDimension>::SpatialSamplePartition(Iter first, Iter last) :
    mURotation(0),
    mVRotation(0),
    mData(build(first, last, typename std::iterator_traits<Iter>::iterator_category()))
{
}

template <typename T, sp::size_type kDimension>
void SpatialSamplePartition<T, kDimension>::rotate(size_type n)
{
    static_assert(scene_rdl2::util::isPowerOfTwo(kDimension),
            "The partition dimension must be a power of 2.");

    if (kDimension > 1) {
        // If kDimension is a power of 2, kDimension/2**m is a power of 2 (iff
        // kDimension > 2**m), by the fundamental theorem of arithmetic. Since
        // its prime factorization is made up of 2s, if we subtract 1 or add 1,
        // we get an odd number. An odd number can not have 2 in its prime
        // factorization.  This means that kDimension is coprime with an odd
        // number. This gives an ideal rotation amount.
        static constexpr size_type p0 = kDimension/2 + 1;
        static constexpr size_type p1 = kDimension/2 - 1;

        mURotation = dimensionMod(p0 * n);

        // Every time we get through all of the permutations, rotate just one
        // dimension. Since we're perfectly square, the permutation count is
        // the same in x and y, and we wouldn't get all of the permutations
        // available to us if we kept rotating rows and columns at the same
        // time, even if we use different rotation amounts.
        mVRotation = dimensionMod(p1 * (n/kDimension));
    }
}

template <typename T, sp::size_type kDimension>
sp::size_type SpatialSamplePartition<T, kDimension>::numPixelSamples() const
{
    return mData.size() / (kDimension * kDimension);
}

template <typename T, sp::size_type kDimension>
std::ostream& SpatialSamplePartition<T, kDimension>::print(std::ostream& outs) const
{
    const auto storedDepth = outs.iword(sOutputDepth);
    const auto depth = (storedDepth == 0) ? numPixelSamples() : storedDepth;
    return pbr::aux::print(outs, *this, kDimension, depth);
}

template <typename T, sp::size_type kDimension>
std::ostream& operator<<(std::ostream& outs, SpatialSamplePartition<T, kDimension>& part)
{
    return part.print(outs);
}

//
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//

// This function will mirror and/or rotate a point 90 degrees for more
// variation in sampling patterns.
template <typename T, sp::size_type sets, sp::size_type samplesPerSet>
template <typename Point>
finline void SamplePartition<T, sets, samplesPerSet>::rotateMirror(Point& p, size_type which)
{
    using std::swap; // Allow ADL

    which &= 7u;

    float& x = getPrimaryValue0(p);
    float& y = getPrimaryValue1(p);

    switch (which) {
        case 0:
            break;
        case 1:
            x = 1.0f - x;
            break;
        case 2:
            y = 1.0f - y;
            break;
        case 3:
            x = 1.0f - x;
            y = 1.0f - y;
            break;
        case 4:
            swap(x, y);
            break;
        case 5:
            x = 1.0f - x;
            swap(x, y);
            break;
        case 6:
            y = 1.0f - y;
            swap(x, y);
            break;
        case 7:
            x = 1.0f - x;
            y = 1.0f - y;
            swap(x, y);
            break;
        default:
            break;
    }
}

template <typename T, sp::size_type sets, sp::size_type samplesPerSet>
finline void SamplePartition<T, sets, samplesPerSet>::rotateMirror(float& f, size_type which)
{
    using std::swap; // Allow ADL

    which &= 1u;

    switch (which) {
        case 0:
            break;
        case 1:
            f = 1.0f - f;
            break;
        default:
            break;
    }
}

template <typename T, sp::size_type sets, sp::size_type samplesPerSet>
finline sp::size_type SamplePartition<T, sets, samplesPerSet>::dimensionMod(size_type n)
{
    static constexpr size_type dimension = scene_rdl2::math::compile_time::isqrt(sets);
    static_assert(dimension*dimension == sets, "Expecting a perfect square of sets");
    return n % dimension;
}

template <typename T, sp::size_type sets, sp::size_type samplesPerSet>
finline sp::size_type SamplePartition<T, sets, samplesPerSet>::getStratum(size_type x, size_type y)
{
    static constexpr size_type dimension = scene_rdl2::math::compile_time::isqrt(sets);
    static_assert(dimension*dimension == sets, "Expecting a perfect square of sets");

    // Precondition: x and y have been modded to range, so that we can wrap our
    // values.
    MNRY_ASSERT(x < dimension);
    MNRY_ASSERT(y < dimension);

    return y * dimension + x;
}

template <typename T, sp::size_type sets, sp::size_type samplesPerSet>
finline T SamplePartition<T, sets, samplesPerSet>::operator()(size_type pixelX, size_type pixelY, size_type n) const
{
    pixelX = dimensionMod(pixelX + mURotation);
    pixelY = dimensionMod(pixelY + mVRotation);

    T t = mData[getStratum(pixelX, pixelY) * samplesPerSet + n];
    rotateMirror(t, mURotation);

    return t;
}

template <typename T, sp::size_type sets, sp::size_type samplesPerSet>
template <typename Iter>
SamplePartition<T, sets, samplesPerSet>::SamplePartition(Iter first, Iter last) :
    mURotation(0),
    mVRotation(0),
    mData(sets * samplesPerSet)
{
    MNRY_ASSERT(sets * samplesPerSet == std::distance(first, last));
    for (size_type i = 0; i < sets; ++i) {
        for (size_type j = 0; j < samplesPerSet; ++j) {
            MNRY_ASSERT(first != last);
            mData[i * samplesPerSet + j] = *first++;
        }
    }
}

template <typename T, sp::size_type sets, sp::size_type samplesPerSet>
void SamplePartition<T, sets, samplesPerSet>::rotate(size_type n)
{
    static constexpr size_type dimension = scene_rdl2::math::compile_time::isqrt(sets);
    static_assert(dimension*dimension == sets, "Expecting a perfect square of sets");

    if (dimension > 1) {
        static constexpr size_type p0 = (dimension >= 4) ? largestPrimeLessThanOrEqual(dimension/2) : 1;
        static constexpr size_type p1 = (p0 >= 3) ? largestPrimeLessThanOrEqual(p0 - 1) : 1;

        mURotation = dimensionMod(p0 * n);

        // Every time we get through all of the permutations, rotate just one
        // dimension. Since we're perfectly square, the permutation count is
        // the same in x and y, and we wouldn't get all of the permutations
        // available to us if we kept rotating rows and columns at the same
        // time, even if we use different rotation amounts.
        mVRotation = dimensionMod(p1 * (n/dimension));
    }
}

template <typename T, sp::size_type sets, sp::size_type samplesPerSet>
std::ostream& SamplePartition<T, sets, samplesPerSet>::print(std::ostream& outs) const
{
    const auto storedDepth = outs.iword(sOutputDepth);
    const auto depth = (storedDepth == 0) ? samplesPerSet : storedDepth;
    constexpr size_type dim = scene_rdl2::math::compile_time::isqrt(sets);
    return pbr::aux::print(outs, *this, dim, depth);
}

template <typename T, sp::size_type sets, sp::size_type samplesPerSet>
std::ostream& operator<<(std::ostream& outs, SamplePartition<T, sets, samplesPerSet>& part)
{
    return part.print(outs);
}

//
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//

namespace aux {
template <typename Container>
std::ostream& print(std::ostream& outs,
                    const Container& c,
                    sp::size_type dim,
                    sp::size_type depth)
{
    typedef typename Container::value_type value_type;

    for (sp::size_type y = 0; y < dim; ++y) {
        for (sp::size_type x = 0; x < dim; ++x) {
            for (sp::size_type i = 0; i < depth; ++i) {
                const value_type& v = c(x, y, i);

                // Since each stratum has is in [0, 1)^2, this will actually
                // cause the points to be printed out in [0, dim]^2.
                const float xv = x + getPrimaryValue0(v);
                const float yv = y + getPrimaryValue1(v);
                outs << xv << ' ' << yv << '\n';
            }
        }
    }
    return outs;
}
}

} // namespace pbr
} // namespace moonray


