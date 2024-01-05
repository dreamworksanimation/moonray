// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>

#define LIGHT_TREE_MEMBERS                                                      \
    HUD_PTR(const HUD_UNIFORM Light * const HUD_UNIFORM *, mBoundedLights);     \
    HUD_MEMBER(uint32_t, mBoundedLightCount);                                   \
    HUD_PTR(const HUD_UNIFORM Light * const HUD_UNIFORM *, mUnboundedLights);   \
    HUD_MEMBER(uint32_t, mUnboundedLightCount);                                 \
    HUD_MEMBER(float, mSceneDiameter);                                          \
    HUD_MEMBER(float, mSamplingThreshold)


#define LIGHT_TREE_VALIDATION                        \
    HUD_BEGIN_VALIDATION(LightTree);                 \
    HUD_VALIDATE(LightTree, mBoundedLights);         \
    HUD_VALIDATE(LightTree, mBoundedLightCount);     \
    HUD_VALIDATE(LightTree, mUnboundedLights);       \
    HUD_VALIDATE(LightTree, mUnboundedLightCount);   \
    HUD_VALIDATE(LightTree, mSceneDiameter);         \
    HUD_VALIDATE(LightTree, mSamplingThreshold);     \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------