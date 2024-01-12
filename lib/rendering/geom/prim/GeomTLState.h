// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/geom/prim/Statistics.h>
#include <moonray/rendering/geom/prim/VolumeRayState.h>

#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <scene_rdl2/scene/rdl2/SceneObject.h>

namespace moonray {
namespace geom {
namespace internal {

// Expose for HUD validation.
class TLState;
typedef geom::internal::TLState GeomTLState;

//-----------------------------------------------------------------------------

class CACHE_ALIGN TLState : public mcrt_common::BaseTLState
{
public:
    TLState(mcrt_common::ThreadLocalState *tls,
            const mcrt_common::TLSInitParams &initParams,
            bool okToAllocBundledResources);

    virtual ~TLState();

    virtual void reset() override;

    // estimateInScatter=true : This is a case when this function is called from 
    // PathIntegrator::estimateInScatteringSourceTerm(). (i.e. light transmittance
    // computation for volume)
    // estimateInScatter=false : All other situations.
    finline void resetVolumeRayState(float tMax, bool estimateInScatter) {
        mVolumeRayState.resetState(tMax, estimateInScatter);
    }

    // Used as a callback which is registered with TLSInitParams.
    static std::shared_ptr<TLState> allocTls(mcrt_common::ThreadLocalState *tls,
            const mcrt_common::TLSInitParams &initParams,
            bool okToAllocBundledResources);

    VolumeRayState mVolumeRayState;
    const scene_rdl2::rdl2::SceneObject * mSubsurfaceTraceSet;
    Statistics mStatistics;

    DISALLOW_COPY_OR_ASSIGNMENT(TLState);
};

//-----------------------------------------------------------------------------

/// Convenience function for iterating over all existing shading TLS instances.
template <typename Body>
finline void forEachTLS(Body const &body)
{
    unsigned numTLS = mcrt_common::getNumTBBThreads();
    mcrt_common::ThreadLocalState *tlsList = mcrt_common::getTLSList();
    for (unsigned i = 0; i < numTLS; ++i) {
        auto geomTls = tlsList[i].mGeomTls.get();
        if (geomTls) {
            body(geomTls);
        }
    }
}

} // namespace internal
} // namespace geom

inline mcrt_common::ExclusiveAccumulators *
getExclusiveAccumulators(geom::internal::TLState *tls)
{
    MNRY_ASSERT(tls);
    return tls->getInternalExclusiveAccumulatorsPtr();
}

} // namespace moonray


