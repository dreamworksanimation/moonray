// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#define LIGHT_SET_MEMBERS                                                                   \
    HUD_PTR(const HUD_UNIFORM Light * const HUD_UNIFORM *, mLights);                        \
    HUD_PTR(const HUD_UNIFORM LightFilterList * const HUD_UNIFORM *, mLightFilterLists);    \
    HUD_MEMBER(int32_t, mLightCount);                                                       \
    HUD_PTR(const LightAccelerator*, mAccelerator);                                         \
    HUD_PTR(const HUD_UNIFORM int * HUD_UNIFORM, mAcceleratorLightIdMap)


#define LIGHT_SET_VALIDATION                        \
    HUD_BEGIN_VALIDATION(LightSet);                 \
    HUD_VALIDATE(LightSet, mLights);                \
    HUD_VALIDATE(LightSet, mLightFilterLists);      \
    HUD_VALIDATE(LightSet, mLightCount);            \
    HUD_VALIDATE(LightSet, mAccelerator);           \
    HUD_VALIDATE(LightSet, mAcceleratorLightIdMap); \
    HUD_END_VALIDATION


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

