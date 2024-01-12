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

#define PATH_VERTEX_MEMBERS                                 \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), pathThroughput); \
    /* Frame buffer path weight. */                         \
    HVD_MEMBER(float, pathPixelWeight);                     \
    HVD_MEMBER(float, aovPathPixelWeight);                  \
    HVD_MEMBER(float, pathDistance);                        \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec2f), minRoughness);   \
    HVD_MEMBER(int, diffuseDepth);                          \
    HVD_MEMBER(int, volumeDepth);                           \
    HVD_MEMBER(int, glossyDepth);                           \
    HVD_MEMBER(int, mirrorDepth);                           \
    HVD_MEMBER(int, nonMirrorDepth);                        \
    HVD_MEMBER(int, presenceDepth);                         \
    HVD_MEMBER(float, totalPresence);                       \
    HVD_MEMBER(int, hairDepth);                             \
    HVD_MEMBER(int, subsurfaceDepth);                       \
    HVD_MEMBER(float, accumOpacity);                        \
    /* for lpe aovs */                                      \
    HVD_MEMBER(int, lpeStateId);                            \
    /* only used by bundling incoherent ray */              \
    /* queue, invalid in all other cases    */              \
    HVD_MEMBER(int, lpeStateIdLight);                       \
    HVD_MEMBER(int, lobeType)


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

#define RAY_STATE_MEMBERS                                                   \
                                                                            \
    HVD_MEMBER(HVD_NAMESPACE(mcrt_common, RayDifferential), mRay);          \
    HVD_MEMBER(PathVertex, mPathVertex);                                    \
    HVD_MEMBER(uint32_t, mSequenceID);                                      \
    HVD_MEMBER(Subpixel, mSubpixel);                                        \
    HVD_MEMBER(uint32_t, mPad0);                                            \
    HVD_MEMBER(uint32_t, mTilePass);                                        \
    HVD_MEMBER(uint32_t, mRayStateIdx);                                     \
    HVD_ISPC_PAD(mPad1, 4);                                                 \
    HVD_PTR(HVD_NAMESPACE(shading, Intersection) *, mAOSIsect);             \
    HVD_MEMBER(uint32_t, mDeepDataHandle);                                  \
    HVD_MEMBER(uint32_t, mCryptomatteDataHandle);                           \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mCryptoRefP);        \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mCryptoRefN);        \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec2f), mCryptoUV);          \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mVolRad);            \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mVolTr);             \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mVolTh);             \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mVolTalpha);         \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mVolTm);             \
    HVD_MEMBER(uint32_t, mVolHit);                                          \
    HVD_MEMBER(float, mVolumeSurfaceT);                                     \
    HVD_ISPC_PAD(mPad, 20)


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

