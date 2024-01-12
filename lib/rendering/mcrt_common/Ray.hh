// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#include <scene_rdl2/common/platform/HybridVaryingData.hh>

//----------------------------------------------------------------------------

// instance0 is used in intersection test (instance attribute and motion vector)
// light is used in occlusion test (shadow linking)
// l2r: concatenated local to render space transform used by multilevel instance,
// be sure this is initialized to the identity.
#define MCRT_COMMON_RAY_EXTENSION_MEMBERS                  \
    HVD_MEMBER(int32_t, materialID);                       \
    HVD_MEMBER(int32_t, depth);                            \
    HVD_PTR(void*, userData);                              \
    HVD_PTR(void*, geomTls);                               \
    HVD_PTR(const void*, priorityMaterial0);               \
    HVD_PTR(const void*, priorityMaterial1);               \
    HVD_PTR(const void*, priorityMaterial2);               \
    HVD_PTR(const void*, priorityMaterial3);               \
    HVD_MEMBER(int32_t, priorityMaterial0Count);           \
    HVD_MEMBER(int32_t, priorityMaterial1Count);           \
    HVD_MEMBER(int32_t, priorityMaterial2Count);           \
    HVD_MEMBER(int32_t, priorityMaterial3Count);           \
    HVD_PTR(const void*, instance0OrLight);                \
    HVD_PTR(const void*, instance1);                       \
    HVD_PTR(const void*, instance2);                       \
    HVD_PTR(const void*, instance3);                       \
    HVD_MEMBER(int32_t, instanceAttributesDepth);          \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Xform3f), l2r);         \
    HVD_MEMBER(int32_t, volumeInstanceState);              \
    HVD_MEMBER(int32_t, shadowReceiverId);                 \
    HVD_ISPC_PAD(pad, 4)

#define MCRT_COMMON_RAY_EXTENSION_VALIDATION(vlen)         \
    HVD_BEGIN_VALIDATION(RayExtension, vlen);              \
    HVD_VALIDATE(RayExtension, materialID);                \
    HVD_VALIDATE(RayExtension, depth);                     \
    HVD_VALIDATE(RayExtension, userData);                  \
    HVD_VALIDATE(RayExtension, geomTls);                   \
    HVD_VALIDATE(RayExtension, priorityMaterial0);         \
    HVD_VALIDATE(RayExtension, priorityMaterial1);         \
    HVD_VALIDATE(RayExtension, priorityMaterial2);         \
    HVD_VALIDATE(RayExtension, priorityMaterial3);         \
    HVD_VALIDATE(RayExtension, priorityMaterial0Count);    \
    HVD_VALIDATE(RayExtension, priorityMaterial1Count);    \
    HVD_VALIDATE(RayExtension, priorityMaterial2Count);    \
    HVD_VALIDATE(RayExtension, priorityMaterial3Count);    \
    HVD_VALIDATE(RayExtension, instance0OrLight);          \
    HVD_VALIDATE(RayExtension, instance1);                 \
    HVD_VALIDATE(RayExtension, instance2);                 \
    HVD_VALIDATE(RayExtension, instance3);                 \
    HVD_VALIDATE(RayExtension, instanceAttributesDepth);   \
    HVD_VALIDATE(RayExtension, l2r);                       \
    HVD_VALIDATE(RayExtension, volumeInstanceState);       \
    HVD_VALIDATE(RayExtension, shadowReceiverId);          \
    HVD_END_VALIDATION

#define MCRT_COMMON_RAY_MEMBERS                            \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), org);    \
    HVD_MEMBER(float, tnear);                              \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), dir);           \
    HVD_MEMBER(float, time);                               \
    HVD_MEMBER(float, tfar);                               \
    HVD_MEMBER(int32_t, mask);                             \
    HVD_MEMBER(uint32_t, id);                              \
    HVD_MEMBER(uint32_t, pad);                             \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), Ng);            \
    HVD_MEMBER(float, u);                                  \
    HVD_MEMBER(float, v);                                  \
    HVD_MEMBER(int32_t, primID);                           \
    HVD_MEMBER(int32_t, geomID);                           \
    HVD_MEMBER(int32_t, instID);                           \
    HVD_MEMBER(RayExtension, ext);                         \
    HVD_MEMBER(Flags, mFlags)

#define MCRT_COMMON_RAY_DIFFERENTIAL_MEMBERS               \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mOriginX);      \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mDirX);         \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mOriginY);      \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mDirY);         \
    HVD_MEMBER(float, mOrigTfar)

#define MCRT_COMMON_RAY_DIFFERENTIAL_VALIDATION(vlen)       \
    HVD_BEGIN_VALIDATION(RayDifferential, vlen);            \
    HVD_VALIDATE(RayDifferential, org);                     \
    HVD_VALIDATE(RayDifferential, tnear);                   \
    HVD_VALIDATE(RayDifferential, dir);                     \
    HVD_VALIDATE(RayDifferential, time);                    \
    HVD_VALIDATE(RayDifferential, tfar);                    \
    HVD_VALIDATE(RayDifferential, mask);                    \
    HVD_VALIDATE(RayDifferential, id);                      \
    HVD_VALIDATE(RayDifferential, pad);                     \
    HVD_VALIDATE(RayDifferential, Ng);                      \
    HVD_VALIDATE(RayDifferential, u);                       \
    HVD_VALIDATE(RayDifferential, v);                       \
    HVD_VALIDATE(RayDifferential, primID);                  \
    HVD_VALIDATE(RayDifferential, geomID);                  \
    HVD_VALIDATE(RayDifferential, instID);                  \
    HVD_VALIDATE(RayDifferential, ext);                     \
    HVD_VALIDATE(RayDifferential, mFlags);                  \
    HVD_VALIDATE(RayDifferential, mOriginX);                \
    HVD_VALIDATE(RayDifferential, mDirX);                   \
    HVD_VALIDATE(RayDifferential, mOriginY);                \
    HVD_VALIDATE(RayDifferential, mDirY);                   \
    HVD_VALIDATE(RayDifferential, mOrigTfar);               \
    HVD_END_VALIDATION


//----------------------------------------------------------------------------



