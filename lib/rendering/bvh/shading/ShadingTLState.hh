// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#include <moonray/rendering/mcrt_common/ThreadLocalState.hh>

#define SHADING_TL_STATE_MEMBERS                            \
    HUD_PTR(const int *, mAttributeOffsets);                \
    HUD_ISPC_PAD(mPad, 48)

#define SHADING_TL_STATE_VALIDATION                         \
    HUD_BEGIN_VALIDATION(ShadingTLState);                   \
    HUD_VALIDATE(ShadingTLState, mAttributeOffsets);        \
    HUD_END_VALIDATION

