// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>

// ---------------------------------------------------------------------------

#define CONE_MEMBERS                                                \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mAxis);      \
    HUD_MEMBER(float, mCosThetaO);                                  \
    HUD_MEMBER(float, mCosThetaE);                                  \
    HUD_MEMBER(bool, mTwoSided)

#define CONE_VALIDATION                                             \
    HUD_BEGIN_VALIDATION(Cone);                                     \
    HUD_VALIDATE(Cone, mAxis);                                      \
    HUD_VALIDATE(Cone, mCosThetaO);                                 \
    HUD_VALIDATE(Cone, mCosThetaE);                                 \
    HUD_VALIDATE(Cone, mTwoSided);                                  \
    HUD_END_VALIDATION

// ---------------------------------------------------------------------------

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