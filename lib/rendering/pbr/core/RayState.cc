// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#include "PbrTLState.h"
#include "RayState.h"
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/common/platform/HybridVaryingData.h>

namespace ispc {
extern "C" uint32_t Subpixel_hvdValidation(bool);
extern "C" uint32_t PathVertex_hvdValidation(bool);
extern "C" uint32_t RayState_hvdValidation(bool);
}

namespace moonray {
namespace pbr {

bool
isRayStateListValid(pbr::TLState *tls, unsigned numEntries, RayState **entries)
{
    MNRY_ASSERT(entries);

    // Verify there are no duplicate pointers in the list.
    scene_rdl2::alloc::Arena *arena = tls->mArena;
    SCOPED_MEM(arena);

    RayState **p = arena->allocArray<RayState *>(numEntries);
    memcpy(p, entries, numEntries * sizeof(RayState *));
    std::sort(p, p + numEntries);
    MNRY_ASSERT(scene_rdl2::util::isSortedAndUnique(numEntries, p));
    
    for (unsigned i = 0; i < numEntries; ++i) {
        MNRY_ASSERT(isValid(p[i]));
    }

    return true;
}

HVD_VALIDATOR(Subpixel);
HVD_VALIDATOR(PathVertex);
HVD_VALIDATOR(RayState);


} // namespace pbr
} // namespace moonray


