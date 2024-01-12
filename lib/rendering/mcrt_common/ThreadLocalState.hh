// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#define BASE_TL_STATE_MEMBERS                                       \
    HUD_VIRTUAL_BASE_CLASS();                                       \
    HUD_MEMBER( const uint32_t, mThreadIdx );                       \
    HUD_PTR( Arena*, mArena );                                      \
    HUD_PTR( Arena*, mPixelArena );                                 \
    HUD_CPP_PTR(ThreadLocalAccumulator *, mIspcAccumulator);        \
    HUD_CPP_PTR(ExclusiveAccumulators *, mExclusiveAccumulatorsPtr)

#define BASE_TL_STATE_VALIDATION                            \
    HUD_BEGIN_VALIDATION(BaseTLState);                      \
    HUD_VALIDATE(BaseTLState, mThreadIdx);                  \
    HUD_VALIDATE(BaseTLState, mArena);                      \
    HUD_VALIDATE(BaseTLState, mPixelArena);                 \
    HUD_VALIDATE(BaseTLState, mIspcAccumulator);            \
    HUD_VALIDATE(BaseTLState, mExclusiveAccumulatorsPtr);   \
    HUD_END_VALIDATION


