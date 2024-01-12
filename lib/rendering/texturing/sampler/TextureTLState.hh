// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#include <moonray/rendering/mcrt_common/ThreadLocalState.hh>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#define TEXTURE_TL_STATE_MEMBERS                    \
    HUD_PTR( TextureSampler *, mTextureSampler);    \
    HUD_PTR( TextureSystem *, mTextureSystem);      \
    HUD_PTR( Perthread *, mOIIOThreadData )

#define TEXTURE_TL_STATE_VALIDATION                 \
    HUD_BEGIN_VALIDATION(TextureTLState);           \
    HUD_VALIDATE(TextureTLState, mTextureSampler);  \
    HUD_VALIDATE(TextureTLState, mTextureSystem);   \
    HUD_VALIDATE(TextureTLState, mOIIOThreadData);  \
    HUD_END_VALIDATION

