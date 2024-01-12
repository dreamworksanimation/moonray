// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once
#include <scene_rdl2/common/platform/HybridUniformData.hh>
#include <scene_rdl2/common/platform/HybridVaryingData.hh>

// Bundled occlusion rays can be forced to unconditionally be unoccluded, instead of invoking the standard occlusion
// test. This functionality is used by the "ray termination color" feature.
enum OcclTestType
{
    STANDARD,
    FORCE_NOT_OCCLUDED
};

//
// BundledOcclRay:
//

#define BUNDLED_OCCL_RAY_MEMBERS                                    \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mOrigin);    \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mDir);       \
    HVD_MEMBER(float, mMinT);                                       \
    HVD_MEMBER(float, mMaxT);                                       \
    HVD_MEMBER(float, mTime);                                       \
    HVD_MEMBER(int, mDepth);                                        \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mRadiance);  \
    HVD_MEMBER(uint32_t, mPixel);                                   \
    HVD_MEMBER(int, mSubpixelIndex);                                \
    HVD_MEMBER(uint32_t, mSequenceID);                              \
    HVD_MEMBER(uint32_t, mTilePass);                                \
    HVD_MEMBER(uint32_t, mDataPtrHandle);                           \
    HVD_MEMBER(uint32_t, mDeepDataHandle);                          \
    HVD_MEMBER(uint32_t, mCryptomatteDataHandle);                   \
    HUD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mCryptoRefP); \
    HUD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mCryptoRefN); \
    HUD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec2f), mCryptoUV);  \
    HVD_MEMBER(uint32_t, mOcclTestType);                            \
    HVD_MEMBER(int32_t,  mShadowReceiverId);                        \
    HVD_ISPC_PAD(mIspcPad, 8)

#define BUNDLED_OCCL_RAY_VALIDATION(vlen)                   \
    HVD_BEGIN_VALIDATION(BundledOcclRay, vlen);             \
    HVD_VALIDATE(BundledOcclRay, mOrigin);                  \
    HVD_VALIDATE(BundledOcclRay, mDir);                     \
    HVD_VALIDATE(BundledOcclRay, mMinT);                    \
    HVD_VALIDATE(BundledOcclRay, mMaxT);                    \
    HVD_VALIDATE(BundledOcclRay, mTime);                    \
    HVD_VALIDATE(BundledOcclRay, mDepth);                   \
    HVD_VALIDATE(BundledOcclRay, mRadiance);                \
    HVD_VALIDATE(BundledOcclRay, mPixel);                   \
    HVD_VALIDATE(BundledOcclRay, mSubpixelIndex);           \
    HVD_VALIDATE(BundledOcclRay, mSequenceID);              \
    HVD_VALIDATE(BundledOcclRay, mTilePass);                \
    HVD_VALIDATE(BundledOcclRay, mDataPtrHandle);           \
    HVD_VALIDATE(BundledOcclRay, mDeepDataHandle);          \
    HVD_VALIDATE(BundledOcclRay, mCryptomatteDataHandle);   \
    HUD_VALIDATE(BundledOcclRay, mCryptoRefP);              \
    HUD_VALIDATE(BundledOcclRay, mCryptoRefN);              \
    HUD_VALIDATE(BundledOcclRay, mCryptoUV);                \
    HVD_VALIDATE(BundledOcclRay, mOcclTestType);            \
    HVD_VALIDATE(BundledOcclRay, mShadowReceiverId);        \
    HVD_END_VALIDATION

#define BUNDLED_OCCL_RAY_DATA_MEMBERS                                   \
    HUD_MEMBER(uint32_t, mFlags);                                       \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Color), mLpeRadiance);   \
    HUD_MEMBER(int, mLpeStateId);                                       \
    HUD_MEMBER(float, mRayEpsilon);                                     \
    HUD_PTR(const Light *, mLight)

#define BUNDLED_OCCL_RAY_DATA_VALIDATION                \
    HUD_BEGIN_VALIDATION(BundledOcclRayData);           \
    HUD_VALIDATE(BundledOcclRayData, mFlags);           \
    HUD_VALIDATE(BundledOcclRayData, mLpeRadiance);     \
    HUD_VALIDATE(BundledOcclRayData, mLpeStateId);      \
    HUD_VALIDATE(BundledOcclRayData, mRayEpsilon);      \
    HUD_VALIDATE(BundledOcclRayData, mLight);           \
    HUD_END_VALIDATION

//
// BundledRadiance:
//

// mRadiance            Alpha is in 4th component.
// mPathPixelWeight     How much to weight this radiance in the frame buffer.
// mPixel               Screen coordinates of pixel.
// mTilePass            Which tile and pass generated this radiance.
#ifdef __AVX512F__

#define BUNDLED_RADIANCE_MEMBERS                                    \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec4f), mRadiance);  \
    HVD_MEMBER(float, mPathPixelWeight);                            \
    HVD_MEMBER(uint32_t, mPixel);                                   \
    HVD_MEMBER(uint32_t, mSubPixelIndex);                           \
    HVD_MEMBER(uint32_t, mDeepDataHandle);                          \
    HVD_MEMBER(uint32_t, mCryptomatteDataHandle);                   \
    HUD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mCryptoRefP); \
    HUD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mCryptoRefN); \
    HUD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec2f), mCryptoUV);  \
    HVD_MEMBER(uint32_t, mTilePass);                                \
    HUD_ARRAY(int, mPad1, 5);

#else

#define BUNDLED_RADIANCE_MEMBERS                                    \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec4f), mRadiance);  \
    HVD_MEMBER(float, mPathPixelWeight);                            \
    HVD_MEMBER(uint32_t, mPixel);                                   \
    HVD_MEMBER(uint32_t, mSubPixelIndex);                           \
    HVD_MEMBER(uint32_t, mDeepDataHandle);                          \
    HVD_MEMBER(uint32_t, mCryptomatteDataHandle);                   \
    HUD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mCryptoRefP); \
    HUD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mCryptoRefN); \
    HUD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec2f), mCryptoUV);  \
    HVD_MEMBER(uint32_t, mTilePass);                                \
    HUD_ARRAY(int, mPad, 6)

#endif

#ifdef __AVX512F__

#define BUNDLED_RADIANCE_VALIDATION(vlen)                  \
    HVD_BEGIN_VALIDATION(BundledRadiance, vlen);           \
    HVD_VALIDATE(BundledRadiance, mRadiance);              \
    HVD_VALIDATE(BundledRadiance, mPathPixelWeight);       \
    HVD_VALIDATE(BundledRadiance, mPixel);                 \
    HVD_VALIDATE(BundledRadiance, mSubPixelIndex);         \
    HVD_VALIDATE(BundledRadiance, mDeepDataHandle);        \
    HVD_VALIDATE(BundledRadiance, mCryptomatteDataHandle); \
    HUD_VALIDATE(BundledRadiance, mCryptoRefP);            \
    HUD_VALIDATE(BundledRadiance, mCryptoRefN);            \
    HUD_VALIDATE(BundledRadiance, mCryptoUV);              \
    HVD_VALIDATE(BundledRadiance, mPad1);                  \
    HVD_END_VALIDATION

#else

#define BUNDLED_RADIANCE_VALIDATION(vlen)                  \
    HVD_BEGIN_VALIDATION(BundledRadiance, vlen);           \
    HVD_VALIDATE(BundledRadiance, mRadiance);              \
    HVD_VALIDATE(BundledRadiance, mPathPixelWeight);       \
    HVD_VALIDATE(BundledRadiance, mPixel);                 \
    HVD_VALIDATE(BundledRadiance, mSubPixelIndex);         \
    HVD_VALIDATE(BundledRadiance, mDeepDataHandle);        \
    HVD_VALIDATE(BundledRadiance, mCryptomatteDataHandle); \
    HUD_VALIDATE(BundledRadiance, mCryptoRefP);            \
    HUD_VALIDATE(BundledRadiance, mCryptoRefN);            \
    HUD_VALIDATE(BundledRadiance, mCryptoUV);              \
    HVD_VALIDATE(BundledRadiance, mTilePass);              \
    HVD_VALIDATE(BundledRadiance, mPad);                   \
    HVD_END_VALIDATION

#endif


#define DEEP_DATA_MEMBERS                                               \
    HUD_CPP_MEMBER(tbb::atomic<int>, mRefCount, 4);                     \
    HUD_MEMBER(uint32_t, mHitDeep);                                     \
    HUD_MEMBER(float, mSubpixelX);                                      \
    HUD_MEMBER(float, mSubpixelY);                                      \
    HUD_MEMBER(int, mLayer);                                            \
    HUD_MEMBER(float, mRayZ);                                           \
    HUD_MEMBER(float, mDeepT);                                          \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mDeepNormal);    \
    HUD_ARRAY(float, mDeepIDs, 6)
// Size limit of 6 is checked in RenderContext::buildFrameState()
// 6 is needed to maintain the 64-byte size of DeepData

#define DEEP_DATA_VALIDATION                            \
    HUD_BEGIN_VALIDATION(DeepData);                     \
    HUD_VALIDATE(DeepData, mRefCount);                  \
    HUD_VALIDATE(DeepData, mHitDeep);                   \
    HUD_VALIDATE(DeepData, mSubpixelX);                 \
    HUD_VALIDATE(DeepData, mSubpixelY);                 \
    HUD_VALIDATE(DeepData, mLayer);                     \
    HUD_VALIDATE(DeepData, mRayZ);                      \
    HUD_VALIDATE(DeepData, mDeepT);                     \
    HUD_VALIDATE(DeepData, mDeepNormal);                \
    HUD_VALIDATE(DeepData, mDeepIDs);                   \
    HUD_END_VALIDATION


#define CRYPTOMATTE_DATA_MEMBERS                                    \
    HUD_CPP_MEMBER(tbb::atomic<int>, mRefCount, 4);                 \
    HUD_CPP_PTR(pbr::CryptomatteBuffer*, mCryptomatteBuffer);       \
    HUD_MEMBER(uint32_t, mHit);                                     \
    HUD_MEMBER(uint32_t, mPrevPresence);                            \
    HUD_MEMBER(float, mId);                                         \
    HUD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mPosition);  \
    HUD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Vec3f), mNormal);    \
    HUD_MEMBER(int32_t, mPresenceDepth);                            \
    HUD_MEMBER(float, mPathPixelWeight);                            \
    HUD_MEMBER(uint32_t, mIsFirstSample)                   

#define CRYPTOMATTE_DATA_VALIDATION                     \
    HUD_BEGIN_VALIDATION(CryptomatteData);              \
    HUD_VALIDATE(CryptomatteData, mRefCount);           \
    HUD_VALIDATE(CryptomatteData, mCryptomatteBuffer);  \
    HUD_VALIDATE(CryptomatteData, mHit);                \
    HUD_VALIDATE(CryptomatteData, mPrevPresence);       \
    HUD_VALIDATE(CryptomatteData, mId);                 \
    HUD_VALIDATE(CryptomatteData, mPosition);           \
    HUD_VALIDATE(CryptomatteData, mNormal);             \
    HUD_VALIDATE(CryptomatteData, mPresenceDepth);      \
    HUD_VALIDATE(CryptomatteData, mPathPixelWeight);    \
    HUD_VALIDATE(CryptomatteData, mIsFirstSample);      \
    HUD_END_VALIDATION

//
// FrameState:
//

// mExecutionMode                   Scalar, bundled, or xpu execution?
// mEmbreeAccel                     Root level Embree geometry accelerator.
// mGPUAccel                        Root level GPU geometry accelerator.
// mLayer                           The active RDL Layer we're rendering from this frame.
// mTextureBlur                     Control texture blurriness.
// mFatalColor                      The color marks incorrect rendering result
// mPropagateVisibilityBounceType   Cache the scene variable value on whether ray
//                                  visibility mask should be propagated
// mIntegrator                      Integrator we are using to render the current frame.
// mScene                           Scene we are rendering this frame.
// mAovSchema                       Copy of Aov schema.
// mMaterialAovs                    Material Aov manager.
// mRequiresHeatMap                 True if producing a heat map.
// mShadingWorkloadChunkSize        The number of entries which should be processed
//                                  in a single iteration. This should be tweaked
//                                  per architecture such that the working data set
//                                  can be kept inside of our cache hierarchy.
// mMaxPresenceDepth                The maximum depth the ray can travel through
//                                  presence < 1 object

#define FRAME_STATE_MEMBERS                                                 \
    HUD_MEMBER(int, mExecutionMode);                                        \
    HUD_CPP_PTR(const rt::EmbreeAccelerator *, mEmbreeAccel);               \
    HUD_CPP_PTR(const rt::GPUAccelerator *, mGPUAccel);                     \
    HUD_CPP_PTR(const scene_rdl2::rdl2::Layer*, mLayer);                    \
    HUD_MEMBER(float, mTextureBlur);                                        \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Color), mFatalColor);        \
    HUD_MEMBER(bool, mPropagateVisibilityBounceType);                       \
    HUD_PTR(const PathIntegrator *, mIntegrator);                           \
    HUD_CPP_PTR(const Scene *, mScene);                                     \
    HUD_CPP_PTR(const AovSchema *, mAovSchema);                             \
    HUD_CPP_PTR(const MaterialAovs *, mMaterialAovs);                       \
    HUD_CPP_PTR(const LightAovs *, mLightAovs);                             \
    HUD_MEMBER(bool, mRequiresHeatMap);                                     \
    HUD_MEMBER(bool,     mLockFrameNoise);                                  \
    HUD_MEMBER(uint32_t, mShadingWorkloadChunkSize);                        \
    HUD_MEMBER(uint32_t, mFrameNumber);                                     \
    HUD_MEMBER(uint32_t, mInitialSeed);                                     \
    HUD_MEMBER(int, mMaxPresenceDepth);                                     \
    HUD_MEMBER(float, mPresenceThreshold);                                  \
    HUD_PTR(const float *, mSamples1D);                                     \
    HUD_PTR(const Sample2D *, mSamples2D);                                  \
    HUD_MEMBER(HUD_NAMESPACE(shading, ShadowTerminatorFix), mShadowTerminatorFix)

#define FRAME_STATE_VALIDATION                                  \
    HUD_BEGIN_VALIDATION(FrameState);                           \
    HUD_VALIDATE(FrameState, mExecutionMode);                   \
    HUD_VALIDATE(FrameState, mEmbreeAccel);                     \
    HUD_VALIDATE(FrameState, mGPUAccel);                        \
    HUD_VALIDATE(FrameState, mLayer);                           \
    HUD_VALIDATE(FrameState, mTextureBlur);                     \
    HUD_VALIDATE(FrameState, mFatalColor);                      \
    HUD_VALIDATE(FrameState, mPropagateVisibilityBounceType);   \
    HUD_VALIDATE(FrameState, mIntegrator);                      \
    HUD_VALIDATE(FrameState, mScene);                           \
    HUD_VALIDATE(FrameState, mAovSchema);                       \
    HUD_VALIDATE(FrameState, mMaterialAovs);                    \
    HUD_VALIDATE(FrameState, mLightAovs);                       \
    HUD_VALIDATE(FrameState, mRequiresHeatMap);                 \
    HUD_VALIDATE(FrameState, mShadingWorkloadChunkSize);        \
    HUD_VALIDATE(FrameState, mLockFrameNoise);                  \
    HUD_VALIDATE(FrameState, mFrameNumber);                     \
    HUD_VALIDATE(FrameState, mInitialSeed);                     \
    HUD_VALIDATE(FrameState, mMaxPresenceDepth);                \
    HUD_VALIDATE(FrameState, mPresenceThreshold);               \
    HUD_VALIDATE(FrameState, mSamples1D);                       \
    HUD_VALIDATE(FrameState, mSamples2D);                       \
    HUD_VALIDATE(FrameState, mShadowTerminatorFix);             \
    HUD_END_VALIDATION


