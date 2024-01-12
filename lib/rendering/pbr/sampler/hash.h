// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <cstdint>

namespace moonray {
namespace pbr {

// From https://github.com/skeeto/hash-prospector
inline uint32_t triple32inc(uint32_t x)
{
    ++x;
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    return x;
}

// From https://github.com/skeeto/hash-prospector
// inverse
inline uint32_t triple32inc_inverse(uint32_t x)
{
    x ^= (x >> 14) ^ (x >> 28);
    x *= 0x32b21703;
    x ^= (x >> 15) ^ (x >> 30);
    x *= 0x469e0db1;
    x ^= (x >> 11) ^ (x >> 22);
    x *= 0x79a85073;
    x ^= x >> 17;
    --x;
    return x;
}

// From https://github.com/skeeto/hash-prospector
// No multiplication (-Imn6)
// bias = 0.023840118344741465
inline uint16_t hash16_s6(uint16_t x)
{
    x += x << 7; x ^= x >> 8;
    x += x << 3; x ^= x >> 2;
    x += x << 4; x ^= x >> 8;
    return x;
}

} // namespace pbr
} // namespace moonray


