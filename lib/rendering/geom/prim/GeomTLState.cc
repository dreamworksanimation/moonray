// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "GeomTLState.h"

namespace moonray {
namespace geom {
namespace internal {

TLState::TLState(mcrt_common::ThreadLocalState *tls, 
        const mcrt_common::TLSInitParams &initParams,
        bool okToAllocBundledResources):
    BaseTLState(tls->mThreadIdx, tls->mArena, tls->mPixelArena),
    mSubsurfaceTraceSet(nullptr)
{
    reset();
}

TLState::~TLState()
{
}

void
TLState::reset()
{
}

std::shared_ptr<TLState>
TLState::allocTls(mcrt_common::ThreadLocalState *tls,
        const mcrt_common::TLSInitParams &initParams,
        bool okToAllocBundledResources)
{
    return std::make_shared<TLState>(tls, initParams, okToAllocBundledResources);
}

//-----------------------------------------------------------------------------

} // namespace internal
} // namespace geom
} // namespace moonray


