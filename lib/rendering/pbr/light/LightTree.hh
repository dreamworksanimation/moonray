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

#define NODE_MEMBERS                                                \
    HUD_MEMBER(uint32_t, mStartIndex);                              \
    HUD_MEMBER(uint32_t, mRightNodeIndex);                          \
    HUD_MEMBER(uint32_t, mLightCount);                              \
    HUD_MEMBER(int32_t, mLightIndex);                               \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, BBox3f), mBBox);     \
    HUD_MEMBER(Cone, mCone);                                        \
    HUD_MEMBER(float, mEnergy);                                     \
    HUD_MEMBER(float, mEnergyVariance);                             \
    HUD_MEMBER(float, mEnergyMean)

#define NODE_VALIDATION                                             \
    HUD_BEGIN_VALIDATION(Node);                                     \
    HUD_VALIDATE(Node, mStartIndex);                                \
    HUD_VALIDATE(Node, mRightNodeIndex);                            \
    HUD_VALIDATE(Node, mLightCount);                                \
    HUD_VALIDATE(Node, mLightIndex);                                \
    HUD_VALIDATE(Node, mBBox);                                      \
    HUD_VALIDATE(Node, mCone);                                      \
    HUD_VALIDATE(Node, mEnergy);                                    \
    HUD_VALIDATE(Node, mEnergyVariance);                            \
    HUD_VALIDATE(Node, mEnergyMean);                                \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------

#define LIGHT_TREE_MEMBERS                                                      \
    HUD_PTR(const HUD_UNIFORM Light * const HUD_UNIFORM *, mBoundedLights);     \
    HUD_MEMBER(uint32_t, mBoundedLightCount);                                   \
    HUD_PTR(const HUD_UNIFORM Light * const HUD_UNIFORM *, mUnboundedLights);   \
    HUD_MEMBER(uint32_t, mUnboundedLightCount);                                 \
    HUD_PTR(Node*, mNodesPtr);                                                  \
    HUD_PTR(uint*, mLightIndicesPtr);                                           \
    HUD_MEMBER(float, mSceneDiameter);                                          \
    HUD_MEMBER(float, mSamplingThreshold)


#define LIGHT_TREE_VALIDATION                        \
    HUD_BEGIN_VALIDATION(LightTree);                 \
    HUD_VALIDATE(LightTree, mBoundedLights);         \
    HUD_VALIDATE(LightTree, mBoundedLightCount);     \
    HUD_VALIDATE(LightTree, mUnboundedLights);       \
    HUD_VALIDATE(LightTree, mUnboundedLightCount);   \
    HUD_VALIDATE(LightTree, mNodesPtr);              \
    HUD_VALIDATE(LightTree, mLightIndicesPtr);       \
    HUD_VALIDATE(LightTree, mSceneDiameter);         \
    HUD_VALIDATE(LightTree, mSamplingThreshold);     \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------