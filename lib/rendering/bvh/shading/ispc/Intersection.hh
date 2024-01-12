// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/HybridVaryingData.hh>


//----------------------------------------------------------------------------

enum AttributeStatus
{
    ATTRIBUTE_UNINITIALIZED  = 0,
    ATTRIBUTE_INITIALIZED    = 1 << 0,
    ATTRIBUTE_DS_INITIALIZED = 1 << 1,
    ATTRIBUTE_DT_INITIALIZED = 1 << 2
};

#define INTERSECTION_MEMBERS                                                \
                                                                            \
    /* ----- Cache line 1 ----- */                                          \
                                                                            \
    /* The rdl2 geometry for this intersection */                           \
    HVD_PTR(const scene_rdl2::rdl2::Geometry*, mGeometryObject);            \
                                                                            \
    /* The assigned material for this intersection */                       \
    HVD_PTR(const scene_rdl2::rdl2::Material*, mMaterial);                  \
                                                                            \
    /* The assigned attribute table for this intersection */                \
    HVD_PTR(const AttributeTable*, mTable);                                 \
                                                                            \
    /* Primitive attribute table */                                         \
    HVD_PTR(char*, mData);                                                  \
    HVD_MEMBER(int32_t, mValidTableOffset);                                 \
    HVD_MEMBER(int32_t, mNumKeys);                                          \
                                                                            \
    /* Id in layer where mMaterial was retrieved from. This is used to */   \
    /* lookup a parallel "runtime" lightset array stored in the Scene class */  \
    HVD_MEMBER(int32_t,  mLayerAssignmentId);                               \
                                                                            \
    /* Vertex index for the intersected triangle */                         \
    HVD_MEMBER(uint32_t, mId1);                                             \
    HVD_MEMBER(uint32_t, mId2);                                             \
    HVD_MEMBER(uint32_t, mId3);                                             \
                                                                            \
    /* Differential Geometry */                                             \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec2f), mSt);                \
                                                                            \
    /* ----- Cache line 2 ----- */                                          \
                                                                            \
    /* Differential Geometry (continued) */                                 \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mP);                 \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mNg);                \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mN);                 \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mdPds);              \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mdPdt);              \
                                                                            \
    HVD_MEMBER(float, mEpsilonHint);                                        \
                                                                            \
    /* ----- Cache line 3 ----- */                                          \
                                                                            \
    HVD_MEMBER(float, mShadowEpsilonHint);                                  \
    HVD_MEMBER(Flags, mFlags);                                              \
                                                                            \
    /* WARNING: Do not change the memory layout of md{ST}d{xy} */           \
    /* as we pass them in to scene_rdl2::math::solve2x2LinearSystem() */    \
    HVD_MEMBER(float, mdSdx);                                               \
    HVD_MEMBER(float, mdTdx);                                               \
    HVD_MEMBER(float, mdSdy);                                               \
    HVD_MEMBER(float, mdTdy);                                               \
                                                                            \
    /* Normal partial derivatives wrt to uv coordinates */                  \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mdNds);              \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mdNdt);              \
                                                                            \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec2f), mMinRoughness);      \
    HVD_MEMBER(float, mMediumIor);                                          \
    HVD_MEMBER(int32_t, mPad0);                                             \
                                                                            \
    /* ----- Cache line 4 -----*/                                           \
                                                                            \
    /* breaks bi-directional algorithms */                                  \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mWo);                \
    HVD_ISPC_PAD(mPad, 52)

#define INTERSECTION_VALIDATION(vlen)                                       \
    HVD_BEGIN_VALIDATION(Intersection, vlen);                               \
    HVD_VALIDATE(Intersection, mGeometryObject);                            \
    HVD_VALIDATE(Intersection, mMaterial);                                  \
    HVD_VALIDATE(Intersection, mTable);                                     \
    HVD_VALIDATE(Intersection, mData);                                      \
    HVD_VALIDATE(Intersection, mValidTableOffset);                          \
    HVD_VALIDATE(Intersection, mNumKeys);                                   \
    HVD_VALIDATE(Intersection, mLayerAssignmentId);                         \
    HVD_VALIDATE(Intersection, mId1);                                       \
    HVD_VALIDATE(Intersection, mId2);                                       \
    HVD_VALIDATE(Intersection, mId3);                                       \
    HVD_VALIDATE(Intersection, mSt);                                        \
    HVD_VALIDATE(Intersection, mP);                                         \
    HVD_VALIDATE(Intersection, mNg);                                        \
    HVD_VALIDATE(Intersection, mN);                                         \
    HVD_VALIDATE(Intersection, mdPds);                                      \
    HVD_VALIDATE(Intersection, mdPdt);                                      \
    HVD_VALIDATE(Intersection, mEpsilonHint);                               \
    HVD_VALIDATE(Intersection, mShadowEpsilonHint);                         \
    HVD_VALIDATE(Intersection, mFlags);                                     \
    HVD_VALIDATE(Intersection, mdSdx);                                      \
    HVD_VALIDATE(Intersection, mdTdx);                                      \
    HVD_VALIDATE(Intersection, mdSdy);                                      \
    HVD_VALIDATE(Intersection, mdTdy);                                      \
    HVD_VALIDATE(Intersection, mdNds);                                      \
    HVD_VALIDATE(Intersection, mdNdt);                                      \
    HVD_VALIDATE(Intersection, mMinRoughness);                              \
    HVD_VALIDATE(Intersection, mMediumIor);                                 \
    HVD_VALIDATE(Intersection, mPad0);                                      \
    HVD_VALIDATE(Intersection, mWo);                                        \
    HVD_END_VALIDATION


//----------------------------------------------------------------------------

