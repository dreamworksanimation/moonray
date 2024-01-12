// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/platform/Platform.h>

#include <cstddef>
#include <cstdint>

namespace moonray {
namespace pbr {

typedef std::uint32_t utype;

constexpr std::size_t kSIMDAlignment  = SIMD_MEMORY_ALIGNMENT;
constexpr utype       kSIMDSize       = VLEN;
constexpr float       kMaxLessThanOne = 0x1.fffffep-1f;

} // namespace pbr
} // namespace moonray

