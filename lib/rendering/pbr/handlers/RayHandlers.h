// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

namespace moonray {

namespace mcrt_common {
class ThreadLocalState;
}

namespace pbr {

struct BundledOcclRay;
struct BundledRadiance;
struct RayState;
class TLState;
union WrappedRayState;

// Passed into ray handlers via userData. These are static for the lifetime of
// the queue.
enum RayHandlerFlags
{
    // Currently empty
};

//
// Bundling handlers for ray intersection and occlusion.
//
void rayBundleHandler(mcrt_common::ThreadLocalState *tls, unsigned numRayStates,
                      WrappedRayState *rayStates, void *userData);

void occlusionQueryBundleHandler(mcrt_common::ThreadLocalState *tls,
                                 unsigned numEntries, BundledOcclRay **entries,
                                 void *userData);

unsigned computeOcclusionQueriesBundled(pbr::TLState *pbrTls, unsigned numEntries,
                                        BundledOcclRay **entries, BundledRadiance *results,
                                        RayHandlerFlags flags);

void presenceShadowsQueryBundleHandler(mcrt_common::ThreadLocalState *tls,
                                       unsigned numEntries, BundledOcclRay **entries,
                                       void *userData);

} // namespace pbr
} // namespace moonray


