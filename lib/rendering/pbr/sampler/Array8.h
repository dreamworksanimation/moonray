// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/Platform.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ostream>

// An array of 10 unsigned integer values of 3 bits each (values in [0-7]).
class Array8
{
    static constexpr std::uint32_t sMask         = 0x7ul;
    static constexpr std::uint32_t sBitsPerDigit = 0x3ul;
public:
    struct NoInit {};

    constexpr Array8() noexcept
    : mData(0)
    {
    }

    constexpr explicit Array8(std::uint32_t init) noexcept
    : mData(init)
    {
    }

    constexpr explicit Array8(NoInit) noexcept
    : mData()
    {
    }

    static constexpr Array8 iota8() noexcept
    {
        //    111 110 101 100 011 010 001 000  // We want this bit pattern [0-7]
        return Array8(076543210ul);
    }

    static constexpr Array8 permutationIdentity() noexcept
    {
        // This value, which is simply 0-7 rotated, results in the identity permutation when the factoradic number is
        // zero. This is because of the way we do the Fisher-Yates shuffle: when zero, we still repeatedly swap with the
        // last digit, resulting in a rotation.

        // This identity value allows us to have lexicographical comparisons in line with the numbers used to generate
        // them. E.g. a seed of 0 is the identity permutation. A seed of 1 is one permutation away in lexicographic
        // ordering.

        // This lexicographic order is important so that we can use a subset of the generated set of permutations.
        // E.g. If we generate 8! permutations, we can access the first 5! values to get permutations of 5.

        //    111 110 101 100 011 010 001 000  // We want this bit pattern [0-7]
        // -> 110 101 100 011 010 001 000 111  // Rotated to account for the way we apply Fisher-Yates
        return Array8(065432107ul);
    }

    std::uint32_t get(std::uint32_t idx) const noexcept
    {
        return (mData >> (idx * sBitsPerDigit)) & sMask;
    }

    std::uint32_t operator[](std::uint32_t idx) const noexcept
    {
        return get(idx);
    }

    void clear(std::uint32_t idx) noexcept
    {
        mData &= (~(sMask << (idx * sBitsPerDigit)));
    }

    void set(std::uint32_t idx, std::uint32_t val) noexcept
    {
        MNRY_ASSERT(val < 8);
        MNRY_ASSERT(idx < 10);
        clear(idx);
        mData |= val << (idx * sBitsPerDigit);
    }

    constexpr std::uint32_t raw() const noexcept
    {
        return mData;
    }

    void swap(std::uint32_t a, std::uint32_t b) noexcept
    {
        const auto t = get(a);
        set(a, get(b));
        set(b, t);
    }

private:
    std::uint32_t mData;
};

inline std::ostream& operator<<(std::ostream& outs, const Array8& array)
{
    outs << std::dec << static_cast<int>(array[0]);
    for (int i = 1; i < 10; ++i) {
        outs << ", " << std::dec << static_cast<int>(array[i]);
    }
    return outs;
}


