// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once
#include <moonray/rendering/pbr/Types.h>
#include "RayState.hh"

#include <moonray/rendering/mcrt_common/Ray.h>
#include <moonray/rendering/shading/Types.h>

namespace moonray {
namespace shading { class Intersection; }
namespace pbr {

//
// TODO: It's worth noting that the current version of RayState is a dumping
//       ground for all data we need to get vectorization phase 2 up and running.
//       For the initial implementation, we're not paying undue attention to the
//       size of the RayState structure. As a second part of the phase 2
//       implementation, we'll need to shrink down its size and pay attention
//       to which data goes on which cachelines.
//

// Subpixel and PathVertex were moved from PathIntegrator to here.

// Identifies where the primary ray comes from
struct Subpixel {
    SUBPIXEL_MEMBERS;

    // HVD validation.
    static uint32_t hvdValidation(bool verbose) { SUBPIXEL_VALIDATION(VLEN); }
};

// Keep track of state along the path recursion, specifically at the ray
// origin of the current ray being processed.
struct PathVertex {
    PATH_VERTEX_MEMBERS;

    // HVD validation.
    static uint32_t hvdValidation(bool verbose) { PATH_VERTEX_VALIDATION(VLEN); }
};


// Structure which encapsulates the state of a ray as it flows through
// the pipeline.
// TODO: shrink the information required to queue a ray.
struct CACHE_ALIGN RayState
{
    RAY_STATE_MEMBERS;

    // HVD validation.
    static uint32_t hvdValidation(bool verbose) { RAY_STATE_VALIDATION(VLEN); }
};

struct CACHE_ALIGN RayStatev
{
    uint8_t mPlaceholder[sizeof(RayState) * VLEN];
};

MNRY_STATIC_ASSERT(sizeof(RayState) * VLEN == sizeof(RayStatev));

// Helpers for converting between a 4 byte and an 8 byte reference to a RayState.
shading::RayStateIndex rayStateToIndex(const RayState *rs);
RayState *indexToRayState(shading::RayStateIndex index);
unsigned getRayStatePoolSize();

inline bool
isValid(const RayState *rs)
{
    MNRY_ASSERT(rs);
    MNRY_ASSERT(rayStateToIndex(rs) < getRayStatePoolSize());
    MNRY_ASSERT(rs->mRay.isValid());
    // TODO: add various validation criteria here...
    return true;
}

// Verify data is valid.
bool isRayStateListValid(pbr::TLState *tls, unsigned numEntries, RayState **entries);

inline void
encodeRayStateInPlace(RayState **rs, uint32_t sortKey)
{
    shading::SortedRayState *srs = reinterpret_cast<shading::SortedRayState *>(rs);
    srs->mRsIdx = rayStateToIndex(*rs);
    srs->mSortKey = sortKey;
}

inline RayState **
decodeRayStatesInPlace(unsigned numEntries, shading::SortedRayState *srcDst)
{
    RayState **dst = reinterpret_cast<RayState **>(srcDst);

    for (unsigned i = 0; i < numEntries; ++i) {
        dst[i] = indexToRayState(srcDst[i].mRsIdx);
    }

    return dst;
}

} // namespace pbr
} // namespace moonray

