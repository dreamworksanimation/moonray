// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
#pragma once

namespace moonray {
namespace shading {

typedef uint32_t RayStateIndex;

// Important that this structure has a memory footprint no larger than a pointer
// (we assume 64-bit).
struct ALIGN(8) SortedRayState
{
    RayStateIndex   mRsIdx;
    uint32_t        mSortKey;
};

MNRY_STATIC_ASSERT(sizeof(SortedRayState) <= sizeof(void *));

} // namespace shading
} // namespace moonray


