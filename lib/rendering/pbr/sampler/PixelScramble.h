// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "Array8.h"
#include "gphash.h"
#include "SamplingConstants.h"

#include <moonray/rendering/pbr/camera/StereoView.h>

#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/render/util/AlignedAllocator.h>
#include <scene_rdl2/render/util/BitUtils.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace moonray {
namespace pbr {

namespace permutations {
template <std::size_t size = 8>
bool checkPermutation(Array8 in)
{
    std::array<int, size> array;
    for (std::size_t i = 0; i < size; ++i) {
        array[i] = in[i];
    }
    std::sort(array.begin(), array.end());
    for (std::size_t i = 0; i < size; ++i) {
        if (array[i] != i) {
            return false;
        }
    }
    return true;
}

bool checkFactoradic(Array8 in);

inline Array8 base10ToFactoradic(std::uint32_t n) noexcept
{
    Array8 ret;
    for (std::uint32_t i = 0; i < 8; ++i) {
        // We ignore the first radix point (0!) 0 * 1 as it offers us no information.
        // Technically, the second radix point (1!) offers us no information either.
        ret.set(i, n % (i + 1));
        n /= (i + 1);
    }
    //MNRY_ASSERT(checkFactoradic(ret));
    return ret;
}

inline Array8 convertFactoradicNumberToPermutation(const Array8& factoradic) noexcept
{
    // Apply a form of Fisher-Yates shuffling.
    Array8 perm = Array8::permutationIdentity();
    for (int i = 7; i > 0; --i) {
        perm.swap(i, factoradic[i]);
    }
    //MNRY_ASSERT(checkPermutation<>(perm));
    return perm;
}
} // namespace permutations

namespace PixelScrambleNS {

// This round down to power of 2 is a C++11 constexpr version of
// TODO: in C++14, we can probably just use the below in a constexpr context.
//
// uint32_t flp2 (uint32_t x)
// {
//     x = x | (x >> 1);
//     x = x | (x >> 2);
//     x = x | (x >> 4);
//     x = x | (x >> 8);
//     x = x | (x >> 16);
//     return x - (x >> 1);
// }

template<typename T>
constexpr T roundDownPower2MaxShift() noexcept
{
    return sizeof(T) * T(8) / T(2);
}

template<typename T>
constexpr T roundDownPower2Helper(T x, T shift) noexcept
{
    return (shift > roundDownPower2MaxShift<T>()) ? x : roundDownPower2Helper(x | (x >> shift), shift * 2);
}

template<typename T>
constexpr T roundDownPower2(T x) noexcept
{
    return roundDownPower2Helper(x, T(1)) - (roundDownPower2Helper(x, T(1)) >> T(1));
}
} // namespace PixelScrambleNS

class PixelScramble
{
public:
    static constexpr std::uint32_t s8Factorial = 40320ul;
    static constexpr std::uint32_t s7Factorial =  5040ul;
    static constexpr std::uint32_t s6Factorial =   720ul;
    static constexpr std::uint32_t s5Factorial =   120ul;
    static constexpr std::uint32_t s4Factorial =    24ul;
    static constexpr std::uint32_t s3Factorial =     6ul;
    static constexpr std::uint32_t s2Factorial =     2ul;
    static constexpr std::uint32_t s1Factorial =     1ul;
    static constexpr std::uint32_t s0Factorial =     1ul;

    alignas(SIMD_MEMORY_ALIGNMENT) static constexpr std::uint32_t sFactorials[] = {
            s0Factorial,
            s1Factorial,
            s2Factorial,
            s3Factorial,
            s4Factorial,
            s5Factorial,
            s6Factorial,
            s7Factorial,
            s8Factorial
    };

    // These are the largest numbers less than [(phi - 1) * n!] which are coprime with n!
    // Where phi is the golden ratio.
    // This will allow us to get far away from the other points as we add a constant offset and take the modulus on the
    // permutation size.
    //
    //  import math
    //  def gcd(a, b):
    //      while a != b:
    //          if a > b:
    //              a = a - b
    //          else:
    //              b = b - a
    //      return a
    //
    //  def factorial(x):
    //      if x == 0:
    //          return 1
    //      return x * factorial(x - 1)
    //
    //  phi = (1.0 + math.sqrt(5.0)) / 2.0
    //
    //  for i in range(2, 16):
    //      f = factorial(i + 1)
    //      g = (phi - 1.0) * f
    //      n = int(g)
    //      while gcd(n, f) > 1:
    //          n -= 1
    //      print('{}ul, // {}!'.format(n, i+1))
    //
    alignas(SIMD_MEMORY_ALIGNMENT) static constexpr std::uint32_t sStrides[] = {
            0ul,     // 0!
            0ul,     // 1!
            1ul,     // 2!
            1ul,     // 3!
            13ul,    // 4!
            73ul,    // 5!
            443ul,   // 6!
            3113ul,  // 7!
            24919ul  // 8!
    };

    alignas(SIMD_MEMORY_ALIGNMENT) static constexpr std::uint32_t sFactorialsRoundDownPow2[] = {
            PixelScrambleNS::roundDownPower2(s0Factorial),
            PixelScrambleNS::roundDownPower2(s1Factorial),
            PixelScrambleNS::roundDownPower2(s2Factorial),
            PixelScrambleNS::roundDownPower2(s3Factorial),
            PixelScrambleNS::roundDownPower2(s4Factorial),
            PixelScrambleNS::roundDownPower2(s5Factorial),
            PixelScrambleNS::roundDownPower2(s6Factorial),
            PixelScrambleNS::roundDownPower2(s7Factorial),
            PixelScrambleNS::roundDownPower2(s8Factorial)
    };

    constexpr PixelScramble() noexcept
    : mSeed(0)
    {
    }

    constexpr explicit PixelScramble(std::uint32_t seed) noexcept
    : mSeed(seed)
    {
    }

    PixelScramble(std::uint32_t px, std::uint32_t py, std::uint32_t frameNumber, StereoView stereoView) noexcept
    {
        const auto stereoSeed = static_cast<std::uint32_t>(stereoView);
        std::uint32_t idx = (px << 16u) | py;
        // Quick scramble of frameNumber and stereoSeed
        // stereoSeed is generally zero-based, so add one to it to make sure it
        // doesn't interfere with frameNumber. The multiplication number is an
        // arbitrarily largish number, prime so that it's less likely to interfere
        // with the mod below, to make sure different stereo views are decently far
        // apart.
        idx ^= __builtin_bswap32(frameNumber + 2000291 * (stereoSeed + 1));
        //idx ^= _byteswap_ulong(frameNumber + 2000291 * (stereoSeed + 1));
        mSeed = gphash(idx);
    }

    // Index can be in absolute terms so we divide by simd size.
    // If we just increment to next factorial, we need to space out the factorial values so they're not in
    // lexicographical order.
    std::uint32_t permuteIndex(std::uint32_t absoluteIndex, std::uint32_t nelements) const noexcept
    {
        MNRY_ASSERT(nelements <= 8);
        const std::div_t d = std::div(static_cast<std::int32_t>(absoluteIndex),
                                      static_cast<std::int32_t>(nelements));
        const std::uint32_t& permutationIndex = d.quot;
        const std::uint32_t& index            = d.rem;

        // For every n elements, we want a different permutation.
        // E.g. if nelements == 4
        // 0 1 2 3 | 0 1 2 3 | 0 1 2 3
        // may become
        // 2 0 1 3 | 0 2 3 1 | 3 2 0 1
        // as absolute index increases beyond nelements
        const std::uint32_t seed = mSeed * permutationIndex + 1ul;
        const std::uint32_t indexHash = (permutationIndex * sStrides[nelements] + seed) % sFactorials[nelements];
        return s8Permutations[indexHash][index];
    }

    // Permute a stream of values SIMDSize items at a time.
    // E.g. if SIMDSize == 4
    // 0 1 2 3 | 4 5 6 7 | 8 9 A B
    // may become
    // 2 1 0 3 | 4 7 5 6 | B 9 8 A
    std::uint32_t permuteSequence(std::uint32_t absoluteIndex, std::uint32_t nelements) const noexcept
    {
        const std::uint32_t index = permuteIndex(absoluteIndex, nelements);
        return roundDownToMultiple(absoluteIndex, nelements) + index;
    }

    std::uint32_t getSeed() const noexcept
    {
        return mSeed;
    }

    static const std::uint32_t* getPermutationsPointer() noexcept;

private:
    using Container = std::vector<Array8, scene_rdl2::alloc::AlignedAllocator<Array8, kSIMDAlignment>>;

    static constexpr std::uint32_t roundDownToMultiple(std::uint32_t n, std::uint32_t multiple) noexcept
    {
        return n/multiple * multiple;
    }

    static Container generate8ElementPermutations();

    static Container s8Permutations;
    std::uint32_t mSeed;
};

} // namespace pbr
} // namespace moonray


