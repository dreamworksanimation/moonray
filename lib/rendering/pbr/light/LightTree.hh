// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>

// ---------------------------------------------------------------------------

#define LIGHT_TREE_CONE_MEMBERS                                     \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mAxis);      \
    HUD_MEMBER(float, mCosThetaO);                                  \
    HUD_MEMBER(float, mCosThetaE);                                  \
    HUD_MEMBER(bool, mTwoSided)

#define LIGHT_TREE_CONE_VALIDATION                                           \
    HUD_BEGIN_VALIDATION(LightTreeCone);                                     \
    HUD_VALIDATE(LightTreeCone, mAxis);                                      \
    HUD_VALIDATE(LightTreeCone, mCosThetaO);                                 \
    HUD_VALIDATE(LightTreeCone, mCosThetaE);                                 \
    HUD_VALIDATE(LightTreeCone, mTwoSided);                                  \
    HUD_END_VALIDATION

// ---------------------------------------------------------------------------

#define LIGHT_TREE_NODE_MEMBERS                                     \
    HUD_MEMBER(uint32_t, mStartIndex);                              \
    HUD_MEMBER(uint32_t, mRightNodeIndex);                          \
    HUD_MEMBER(uint32_t, mLightCount);                              \
    HUD_MEMBER(int32_t, mLightIndex);                               \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, BBox3f), mBBox);     \
    HUD_MEMBER(LightTreeCone, mCone);                               \
    HUD_MEMBER(float, mEnergy);                                     \
    HUD_MEMBER(float, mEnergyVariance);                             \
    HUD_MEMBER(float, mEnergyMean)

#define LIGHT_TREE_NODE_VALIDATION                                  \
    HUD_BEGIN_VALIDATION(LightTreeNode);                            \
    HUD_VALIDATE(LightTreeNode, mStartIndex);                       \
    HUD_VALIDATE(LightTreeNode, mRightNodeIndex);                   \
    HUD_VALIDATE(LightTreeNode, mLightCount);                       \
    HUD_VALIDATE(LightTreeNode, mLightIndex);                       \
    HUD_VALIDATE(LightTreeNode, mBBox);                             \
    HUD_VALIDATE(LightTreeNode, mCone);                             \
    HUD_VALIDATE(LightTreeNode, mEnergy);                           \
    HUD_VALIDATE(LightTreeNode, mEnergyVariance);                   \
    HUD_VALIDATE(LightTreeNode, mEnergyMean);                       \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------

#define LIGHT_TREE_MEMBERS                                                      \
    HUD_PTR(const HUD_UNIFORM Light * const HUD_UNIFORM *, mBoundedLights);     \
    HUD_MEMBER(uint32_t, mBoundedLightCount);                                   \
    HUD_PTR(const HUD_UNIFORM Light * const HUD_UNIFORM *, mUnboundedLights);   \
    HUD_MEMBER(uint32_t, mUnboundedLightCount);                                 \
    HUD_PTR(LightTreeNode*, mNodesPtr);                                         \
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