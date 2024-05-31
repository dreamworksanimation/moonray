// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#include <scene_rdl2/common/platform/HybridVaryingData.hh>


//----------------------------------------------------------------------------

// Identifies where the primary ray comes from
#define SUBPIXEL_MEMBERS                                                \
    /* Pixel location, this doubles as a 32-bit sort key for the */     \
    /* radiance queues. */                                              \
    HVD_MEMBER(uint32_t, mPixel);                                       \
    HVD_MEMBER(int, mSubpixelIndex);                                    \
    HVD_MEMBER(float, mSubpixelX);                                      \
    HVD_MEMBER(float, mSubpixelY);                                      \
    HVD_MEMBER(int, mPixelSamples);                                     \
    HVD_MEMBER(float, mSampleClampingValue);                            \
    HVD_MEMBER(float, mPrimaryRayDiffScale);                            \
    HVD_MEMBER(float, mTextureDiffScale)

#define SUBPIXEL_VALIDATION(vlen)                                       \
    HVD_BEGIN_VALIDATION(Subpixel, vlen);                               \
    HVD_VALIDATE(Subpixel, mPixel);                                     \
    HVD_VALIDATE(Subpixel, mSubpixelIndex);                             \
    HVD_VALIDATE(Subpixel, mSubpixelX);                                 \
    HVD_VALIDATE(Subpixel, mSubpixelY);                                 \
    HVD_VALIDATE(Subpixel, mPixelSamples);                              \
    HVD_VALIDATE(Subpixel, mSampleClampingValue);                       \
    HVD_VALIDATE(Subpixel, mPrimaryRayDiffScale);                       \
    HVD_VALIDATE(Subpixel, mTextureDiffScale);                          \
    HVD_END_VALIDATION


//----------------------------------------------------------------------------

#define PATH_VERTEX_MEMBERS                                                 /*  size */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), pathThroughput);     /*   12  */\
    /* Frame buffer path weight. */                                                    \
    HVD_MEMBER(float, pathPixelWeight);                                     /*   16  */\
    HVD_MEMBER(float, aovPathPixelWeight);                                  /*   20  */\
    HVD_MEMBER(float, pathDistance);                                        /*   24  */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec2f), minRoughness);       /*   32  */\
    HVD_MEMBER(int, diffuseDepth);                                          /*   36  */\
    HVD_MEMBER(int, volumeDepth);                                           /*   40  */\
    HVD_MEMBER(int, glossyDepth);                                           /*   44  */\
    HVD_MEMBER(int, mirrorDepth);                                           /*   48  */\
    HVD_MEMBER(int, nonMirrorDepth);                                        /*   52  */\
    HVD_MEMBER(int, presenceDepth);                                         /*   56  */\
    HVD_MEMBER(float, totalPresence);                                       /*   60  */\
    HVD_MEMBER(int, hairDepth);                                             /*   64  */\
    HVD_MEMBER(int, subsurfaceDepth);                                       /*   68  */\
    HVD_MEMBER(float, accumOpacity);                                        /*   72  */\
    /* for lpe aovs */                                                                 \
    HVD_MEMBER(int, lpeStateId);                                            /*   76  */\
    /* only used by bundling incoherent ray */                                         \
    /* queue, invalid in all other cases    */                                         \
    HVD_MEMBER(int, lpeStateIdLight);                                       /*   80  */\
    HVD_MEMBER(int, lobeType)                                               /*   84  */


#define PATH_VERTEX_VALIDATION(vlen)                \
    HVD_BEGIN_VALIDATION(PathVertex, vlen);         \
    HVD_VALIDATE(PathVertex, pathThroughput);       \
    HVD_VALIDATE(PathVertex, pathPixelWeight);      \
    HVD_VALIDATE(PathVertex, aovPathPixelWeight);   \
    HVD_VALIDATE(PathVertex, pathDistance);         \
    HVD_VALIDATE(PathVertex, minRoughness);         \
    HVD_VALIDATE(PathVertex, diffuseDepth);         \
    HVD_VALIDATE(PathVertex, volumeDepth);          \
    HVD_VALIDATE(PathVertex, glossyDepth);          \
    HVD_VALIDATE(PathVertex, mirrorDepth);          \
    HVD_VALIDATE(PathVertex, nonMirrorDepth);       \
    HVD_VALIDATE(PathVertex, presenceDepth);        \
    HVD_VALIDATE(PathVertex, totalPresence);        \
    HVD_VALIDATE(PathVertex, hairDepth);            \
    HVD_VALIDATE(PathVertex, subsurfaceDepth);      \
    HVD_VALIDATE(PathVertex, accumOpacity);         \
    HVD_VALIDATE(PathVertex, lpeStateId);           \
    HVD_VALIDATE(PathVertex, lpeStateIdLight);      \
    HVD_VALIDATE(PathVertex, lobeType);             \
    HVD_END_VALIDATION


//----------------------------------------------------------------------------


#if CACHE_LINE_SIZE == 128
/*Alignment: 128 (CACHE_LINE_SIZE), Total size: 584, Padded size: 640*/
#define RAY_STATE_MEMBERS_PAD   (46+8)
#else
/*Alignment: 64 (CACHE_LINE_SIZE), Total size: 568, Padded size: 576 */
#define RAY_STATE_MEMBERS_PAD   8
#endif

#define RAY_STATE_MEMBERS                                                   /*   size   macOS  */\
    HVD_MEMBER(HVD_NAMESPACE(mcrt_common, RayDifferential), mRay);          /*    304    320   */\
    HVD_MEMBER(PathVertex, mPathVertex);                                    /*    388    404   */\
    HVD_MEMBER(uint32_t, mSequenceID);                                      /*    392    408   */\
    HVD_MEMBER(Subpixel, mSubpixel);                                        /*    424    440   */\
    HVD_MEMBER(uint32_t, mPad0);                                            /*    428    444   */\
    HVD_MEMBER(uint32_t, mTilePass);                                        /*    432    448   */\
    HVD_MEMBER(uint32_t, mRayStateIdx);                                     /*    436    452   */\
    HVD_ISPC_PAD(mPad1, 4);                                                 /*    440    456   */\
    HVD_PTR(HVD_NAMESPACE(shading, Intersection) *, mAOSIsect);             /*    448    464   */\
    HVD_MEMBER(uint32_t, mDeepDataHandle);                                  /*    452    468   */\
    HVD_MEMBER(uint32_t, mCryptomatteDataHandle);                           /*    456    472   */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mCryptoRefP);        /*    468    484   */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mCryptoP0);          /*    480    496   */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mCryptoRefN);        /*    492    508   */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec2f), mCryptoUV);          /*    500    516   */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mVolRad);            /*    512    528   */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mVolTr);             /*    524    540   */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mVolTh);             /*    536    552   */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mVolTalpha);         /*    548    564   */\
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mVolTm);             /*    560    576   */\
    HVD_MEMBER(uint32_t, mVolHit);                                          /*    564    580   */\
    HVD_MEMBER(float, mVolumeSurfaceT);                                     /*    568    584   */\
    HVD_ISPC_PAD(pad, RAY_STATE_MEMBERS_PAD)                                /*    576    640   */\
                                                              /* macOS: 640 * 4 lanes = 2560   */\
                                                              /* linux: 576 * 8 lanes = 4608   */\


#define RAY_STATE_VALIDATION(vlen)                                          \
    HVD_BEGIN_VALIDATION(RayState, vlen);                                   \
    HVD_VALIDATE(RayState, mRay);                                           \
    HVD_VALIDATE(RayState, mPathVertex);                                    \
    HVD_VALIDATE(RayState, mSequenceID);                                    \
    HVD_VALIDATE(RayState, mSubpixel);                                      \
    HVD_VALIDATE(RayState, mPad0);                                          \
    HVD_VALIDATE(RayState, mTilePass);                                      \
    HVD_VALIDATE(RayState, mRayStateIdx);                                   \
    HVD_VALIDATE(RayState, mAOSIsect);                                      \
    HVD_VALIDATE(RayState, mDeepDataHandle);                                \
    HVD_VALIDATE(RayState, mCryptomatteDataHandle);                         \
    HVD_VALIDATE(RayState, mCryptoRefP);                                    \
    HVD_VALIDATE(RayState, mCryptoP0);                                      \
    HVD_VALIDATE(RayState, mCryptoRefN);                                    \
    HVD_VALIDATE(RayState, mCryptoUV);                                      \
    HVD_VALIDATE(RayState, mVolRad);                                        \
    HVD_VALIDATE(RayState, mVolTr);                                         \
    HVD_VALIDATE(RayState, mVolTh);                                         \
    HVD_VALIDATE(RayState, mVolTalpha);                                     \
    HVD_VALIDATE(RayState, mVolTm);                                         \
    HVD_VALIDATE(RayState, mVolHit);                                        \
    HVD_VALIDATE(RayState, mVolumeSurfaceT);                                \
    HVD_END_VALIDATION


//----------------------------------------------------------------------------

