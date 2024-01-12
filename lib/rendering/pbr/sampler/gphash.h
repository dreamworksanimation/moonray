// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once


#include <scene_rdl2/render/util/BitUtils.h>
#include <cstdint>

namespace moonray {
namespace pbr {

// Multiply 2^exponent - 1 by factor
inline std::uint32_t mersenneMultiply(std::uint32_t exponent, std::uint32_t factor) noexcept
{
    std::uint32_t result = factor << exponent;
    result -= factor;
    return result;
}

constexpr std::uint32_t hashCombine(std::uint32_t seed, std::uint32_t v) noexcept
{
    return seed ^ (v + 0x9e3779b9u + (seed<<6u) + (seed>>2u));
}

// From "Evolving Hash Functions by Means of Genetic Programming" by Estebanez,
// Hernadez-Castro, and Ribagorda
inline std::uint32_t gphash(std::uint32_t in, std::uint32_t seed = 65521u) noexcept
{
    // This seed modification is a departure from the paper to improve cascades
    // on sequential seed values.
    seed ^= (0xb994ef91u + (seed<<6u) + (seed>>2u));

    constexpr std::uint32_t magicNumber = 0x6CF575C5u;
    std::uint32_t AUX = magicNumber * (seed + in);
    AUX = scene_rdl2::util::rotateRight(AUX, 18u);
    return magicNumber * AUX;
}

} // namespace pbr
} // namespace moonray

