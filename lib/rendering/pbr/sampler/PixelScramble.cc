// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#include "Array8.h"
#include "PixelScramble.h"

namespace moonray {
namespace pbr {
alignas(SIMD_MEMORY_ALIGNMENT) constexpr std::uint32_t PixelScramble::sFactorials[];
alignas(SIMD_MEMORY_ALIGNMENT) constexpr std::uint32_t PixelScramble::sStrides[];
alignas(SIMD_MEMORY_ALIGNMENT) constexpr std::uint32_t PixelScramble::sFactorialsRoundDownPow2[];
PixelScramble::Container PixelScramble::s8Permutations = PixelScramble::generate8ElementPermutations();
} // namespace pbr
} // namespace moonray

extern "C" {
    // Setup extern pointers so they can be accessed within ISPC.
    const std::uint32_t* kISPC8Permutations          = moonray::pbr::PixelScramble::getPermutationsPointer();
    const std::uint32_t* kISPCFactorials             = moonray::pbr::PixelScramble::sFactorials;
    const std::uint32_t* kISPCFactorialRoundDownPow2 = moonray::pbr::PixelScramble::sFactorialsRoundDownPow2;
    const std::uint32_t* kISPCStrides                = moonray::pbr::PixelScramble::sStrides;
}

namespace moonray {
namespace pbr {

namespace permutations {
bool checkFactoradic(Array8 in)
{
    for (std::uint32_t i = 0; i < 8; ++i) {
        if (in[i] >= (i + 1)) {
            return false;
        }
    }
    return true;
}
} // namespace permutations

PixelScramble::Container PixelScramble::generate8ElementPermutations()
{
    Container values;
    values.resize(s8Factorial);

    for (std::uint32_t i = 0; i < s8Factorial; ++i) {
        const Array8 factoradic  = permutations::base10ToFactoradic(i);
        const Array8 permutation = permutations::convertFactoradicNumberToPermutation(factoradic);
        values.at(i) = permutation;
    }
    return values;
}

const std::uint32_t* PixelScramble::getPermutationsPointer() noexcept
{
    MNRY_ASSERT(!s8Permutations.empty());
    static_assert(sizeof(Array8) == sizeof(std::uint32_t), "We're treating Array8s as uint32_ts");
    const Array8& front = s8Permutations.front();
    return reinterpret_cast<const std::uint32_t*>(std::addressof(front));
}

} // namespace pbr
} // namespace moonray


