// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>

#define DISPLAYFILTER_INPUTBUFFER_MEMBERS                   \
    HUD_ISPC_PAD(mPad1, 16);                                \
    HUD_PTR(VariablePixelBuffer *, mPixelBuffer);           \
    HUD_MEMBER(unsigned int, mStartX);                      \
    HUD_MEMBER(unsigned int, mStartY);                      \
    HUD_ARRAY(int, mPad2, 8)

#define DISPLAYFILTER_INPUTBUFFER_VALIDATION                \
    HUD_BEGIN_VALIDATION(InputBuffer);                      \
    HUD_VALIDATE(InputBuffer, mPixelBuffer);                \
    HUD_VALIDATE(InputBuffer, mStartX);                     \
    HUD_VALIDATE(InputBuffer, mStartY);                     \
    HUD_VALIDATE(InputBuffer, mPad2);                       \
    HUD_END_VALIDATION

