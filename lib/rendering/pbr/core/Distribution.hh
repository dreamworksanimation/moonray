// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>


#define TEXTURE_FILTER_ENUM                 \
    TEXTURE_FILTER_NEAREST = 0,             \
    TEXTURE_FILTER_BILINEAR,                \
    TEXTURE_FILTER_NEAREST_MIP_NEAREST,     \
    TEXTURE_FILTER_BILINEAR_MIP_NEAREST,    \
    TEXTURE_FILTER_NUM_TYPES

#define TEXTURE_FILTER_ENUM_VALIDATION                                                              \
    MNRY_ASSERT_REQUIRE(TEXTURE_FILTER_NEAREST == (TextureFilter)ispc::TEXTURE_FILTER_NEAREST);      \
    MNRY_ASSERT_REQUIRE(TEXTURE_FILTER_NUM_TYPES == (TextureFilter)ispc::TEXTURE_FILTER_NUM_TYPES)



//----------------------------------------------------------------------------

#define GUIDE_DISTRIBUTION_1D_MEMBERS   \
    HUD_MEMBER(uint32_t, mSizeCdf);     \
    HUD_MEMBER(uint32_t, mSizeGuide);   \
    HUD_MEMBER(float, mInvSizeCdf);     \
    HUD_MEMBER(float, mInvSizeGuide);   \
    HUD_MEMBER(float, mTotalWeight);    \
    HUD_MEMBER(float, mInvTotalWeight); \
    HUD_MEMBER(float, mThresholdLow);   \
    HUD_MEMBER(float, mThresholdHigh);  \
    HUD_MEMBER(float, mLinearCoeffLow); \
    HUD_MEMBER(float, mLinearCoeffHigh);\
    HUD_PTR(float *, mCdf);             \
    HUD_PTR(uint32_t *, mGuide);        \
    HUD_MEMBER(uint32_t, mOwnsArrays)

#define GUIDE_DISTRIBUTION_1D_VALIDATION                \
    HUD_BEGIN_VALIDATION(GuideDistribution1D);          \
    HUD_VALIDATE(GuideDistribution1D, mSizeCdf);        \
    HUD_VALIDATE(GuideDistribution1D, mSizeGuide);      \
    HUD_VALIDATE(GuideDistribution1D, mInvSizeCdf);     \
    HUD_VALIDATE(GuideDistribution1D, mInvSizeGuide);   \
    HUD_VALIDATE(GuideDistribution1D, mTotalWeight);    \
    HUD_VALIDATE(GuideDistribution1D, mInvTotalWeight); \
    HUD_VALIDATE(GuideDistribution1D, mThresholdLow);   \
    HUD_VALIDATE(GuideDistribution1D, mThresholdHigh);  \
    HUD_VALIDATE(GuideDistribution1D, mLinearCoeffLow); \
    HUD_VALIDATE(GuideDistribution1D, mLinearCoeffHigh);\
    HUD_VALIDATE(GuideDistribution1D, mCdf);            \
    HUD_VALIDATE(GuideDistribution1D, mGuide);          \
    HUD_VALIDATE(GuideDistribution1D, mOwnsArrays);     \
    HUD_END_VALIDATION

#define DISTRIBUTION_2D_MEMBERS                                 \
    HUD_MEMBER(uint32_t, mSizeV);                               \
    HUD_PTR(GuideDistribution1D * HUD_UNIFORM *, mConditional); \
    HUD_PTR(GuideDistribution1D *, mMarginal)

#define DISTRIBUTION_2D_VALIDATION              \
    HUD_BEGIN_VALIDATION(Distribution2D);       \
    HUD_VALIDATE(Distribution2D, mSizeV);       \
    HUD_VALIDATE(Distribution2D, mConditional); \
    HUD_VALIDATE(Distribution2D, mMarginal);    \
    HUD_END_VALIDATION


#define IMAGE_DISTRIBUTION_MEMBERS                                          \
    HUD_PTR(Distribution2D * HUD_UNIFORM *, mDistribution);                 \
    HUD_MEMBER(uint32_t, mWidth);                                           \
    HUD_MEMBER(uint32_t, mHeight);                                          \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Mat3f), mTransformation);    \
    HUD_MEMBER(bool,  mIsTransformed);                                      \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Color), mBorderColor);       \
    HUD_MEMBER(float, mRepsU);                                              \
    HUD_MEMBER(float, mRepsV);                                              \
    HUD_MEMBER(bool, mMirrorU);                                             \
    HUD_MEMBER(bool, mMirrorV);                                             \
    HUD_MEMBER(uint32_t, mNumMipLevels);                                    \
    HUD_PTR(float * HUD_UNIFORM *, mPixelBuffer)

#define IMAGE_DISTRIBUTION_VALIDATION                 \
    HUD_BEGIN_VALIDATION(ImageDistribution);          \
    HUD_VALIDATE(ImageDistribution, mDistribution);   \
    HUD_VALIDATE(ImageDistribution, mWidth);          \
    HUD_VALIDATE(ImageDistribution, mHeight);         \
    HUD_VALIDATE(ImageDistribution, mTransformation); \
    HUD_VALIDATE(ImageDistribution, mIsTransformed);  \
    HUD_VALIDATE(ImageDistribution, mBorderColor);    \
    HUD_VALIDATE(ImageDistribution, mRepsU);          \
    HUD_VALIDATE(ImageDistribution, mRepsV);          \
    HUD_VALIDATE(ImageDistribution, mMirrorU);        \
    HUD_VALIDATE(ImageDistribution, mMirrorV);        \
    HUD_VALIDATE(ImageDistribution, mNumMipLevels);   \
    HUD_VALIDATE(ImageDistribution, mPixelBuffer);    \
    HUD_END_VALIDATION


//----------------------------------------------------------------------------

