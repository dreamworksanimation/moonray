// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#ifdef __APPLE__
#include <scene_rdl2/common/platform/platform.hh>
#endif
#include <scene_rdl2/common/platform/HybridVaryingData.hh>

//----------------------------------------------------------------------------

// instance0 is used in intersection test (instance attribute and motion vector)
// light is used in occlusion test (shadow linking)
// l2r: concatenated local to render space transform used by multilevel instance,
// be sure this is initialized to the identity.
#define MCRT_COMMON_RAY_DIFFERENTIAL_PADDING 4  /*Alignment: 8, Total size: 164, Padded size: 168 */

#define MCRT_COMMON_RAY_EXTENSION_MEMBERS                       /*  size */\
    HVD_MEMBER(int32_t, materialID);                            /*    4  */\
    HVD_MEMBER(int32_t, depth);                                 /*    8  */\
    HVD_PTR(void*, userData);                                   /*   16  */\
    HVD_PTR(void*, geomTls);                                    /*   24  */\
    HVD_PTR(const void*, priorityMaterial0);                    /*   32  */\
    HVD_PTR(const void*, priorityMaterial1);                    /*   40  */\
    HVD_PTR(const void*, priorityMaterial2);                    /*   48  */\
    HVD_PTR(const void*, priorityMaterial3);                    /*   56  */\
    HVD_MEMBER(int32_t, priorityMaterial0Count);                /*   60  */\
    HVD_MEMBER(int32_t, priorityMaterial1Count);                /*   64  */\
    HVD_MEMBER(int32_t, priorityMaterial2Count);                /*   68  */\
    HVD_MEMBER(int32_t, priorityMaterial3Count);                /*   72  */\
    HVD_PTR(const void*, instance0OrLight);                     /*   80  */\
    HVD_PTR(const void*, instance1);                            /*   88  */\
    HVD_PTR(const void*, instance2);                            /*   96  */\
    HVD_PTR(const void*, instance3);                            /*  104  */\
    HVD_MEMBER(int32_t, instanceAttributesDepth);               /*  108  */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Xform3f), l2r);  /*  156  */\
    HVD_MEMBER(int32_t, volumeInstanceState);                   /*  160  */\
    HVD_MEMBER(int32_t, shadowReceiverId);                      /*  164  */\
    HVD_ISPC_PAD(pad, MCRT_COMMON_RAY_DIFFERENTIAL_PADDING)     /*  168  */\
                                          /* macOS: 168 * 4 lanes = 672  */\
                                         /* linux: 168 * 8 lanes = 1344  */\

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

#define MCRT_COMMON_RAY_MEMBERS                                 /*  size */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), org);    /*   12  */\
    HVD_MEMBER(float, tnear);                                   /*   16  */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), dir);    /*   28  */\
    HVD_MEMBER(float, time);                                    /*   32  */\
    HVD_MEMBER(float, tfar);                                    /*   36  */\
    HVD_MEMBER(int32_t, mask);                                  /*   40  */\
    HVD_MEMBER(uint32_t, id);                                   /*   44  */\
    HVD_MEMBER(uint32_t, pad);                                  /*   48  */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), Ng);     /*   60  */\
    HVD_MEMBER(float, u);                                       /*   64  */\
    HVD_MEMBER(float, v);                                       /*   68  */\
    HVD_MEMBER(int32_t, primID);                                /*   72  */\
    HVD_MEMBER(int32_t, geomID);                                /*   76  */\
    HVD_MEMBER(int32_t, instID);                                /*   80  */\
    HVD_MEMBER(RayExtension, ext);                              /*  248  */\
    HVD_MEMBER(Flags, mFlags)                                   /*  252  */\
                                         /* macOS: 252 * 4 lanes = 1008  */\
                                         /* linux: 252 * 8 lanes = 2016  */\

#if CACHE_LINE_SIZE == 128
#define MCRT_COMMON_RAY_DIFFERENTIAL_MEMBERS_CACHE_PAD   (4)
#else
#define MCRT_COMMON_RAY_DIFFERENTIAL_MEMBERS_CACHE_PAD   0
#endif

#define MCRT_COMMON_RAY_DIFFERENTIAL_MEMBERS                                    /*   size   macOS */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mOriginX);               /*    264    264  */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mDirX);                  /*    276    276  */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mOriginY);               /*    288    288  */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mDirY);                  /*    300    300  */\
    HVD_MEMBER(float, mOrigTfar);                                               /*    304    304  */\
    HVD_ARRAY(uint32_t, pad1, MCRT_COMMON_RAY_DIFFERENTIAL_MEMBERS_CACHE_PAD)   /*    304    320  */\
                                                                  /* macOS: 320 * 4 lanes = 1280  */\
                                                                  /* linux: 304 * 8 lanes = 2432  */\

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



