// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>

// These counts were experimentally determined by changing the number of lights in the
// production scene /work/rd/raas/scenes/AB_localized/sq1001_s3/scene.rdla. They represent
// the number of lights in a light set at which the LightAccelerator runs faster than
// linear search.

#define SCALAR_THRESHOLD_COUNT 25
#define VECTOR_THRESHOLD_COUNT 50

//----------------------------------------------------------------------------

#define LIGHT_ACCELERATOR_MEMBERS                                               \
    HUD_MEMBER(RTCScene, mRtcScene);                                            \
    HUD_PTR(const HUD_UNIFORM Light * const HUD_UNIFORM *, mLights);            \
    HUD_MEMBER(int32_t, mLightCount);                                           \
    HUD_PTR(const HUD_UNIFORM Light * const HUD_UNIFORM *, mBoundedLights);     \
    HUD_MEMBER(int32_t, mBoundedLightCount);                                    \
    HUD_PTR(const HUD_UNIFORM Light * const HUD_UNIFORM *, mUnboundedLights);   \
    HUD_MEMBER(int32_t, mUnboundedLightCount);                                  \
    HUD_MEMBER(HUD_UNIFORM LightTree, mSamplingTree)

#define LIGHT_ACCELERATOR_VALIDATION                        \
    HUD_BEGIN_VALIDATION(LightAccelerator);                 \
    HUD_VALIDATE(LightAccelerator, mRtcScene);              \
    HUD_VALIDATE(LightAccelerator, mLights);                \
    HUD_VALIDATE(LightAccelerator, mLightCount);            \
    HUD_VALIDATE(LightAccelerator, mBoundedLights);         \
    HUD_VALIDATE(LightAccelerator, mBoundedLightCount);     \
    HUD_VALIDATE(LightAccelerator, mUnboundedLights);       \
    HUD_VALIDATE(LightAccelerator, mUnboundedLightCount);   \
    HUD_VALIDATE(LightAccelerator, mSamplingTree);          \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------

