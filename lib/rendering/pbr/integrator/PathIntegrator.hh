// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#pragma once
#include <moonray/rendering/mcrt_common/ThreadLocalState.hh>

#define PATH_INTEGRATOR_MEMBERS                            \
    HUD_MEMBER(int, mLightSamples);                        \
    HUD_MEMBER(int, mBsdfSamplesSqrt);                     \
    HUD_MEMBER(int, mBsdfSamples);                         \
    HUD_MEMBER(int, mBssrdfSamples);                       \
    HUD_MEMBER(int, mMaxDepth);                            \
    HUD_MEMBER(int, mMaxDiffuseDepth);                     \
    HUD_MEMBER(int, mMaxGlossyDepth);                      \
    HUD_MEMBER(int, mMaxMirrorDepth);                      \
    HUD_MEMBER(int, mMaxVolumeDepth);                      \
    HUD_MEMBER(int, mMaxPresenceDepth);                    \
    HUD_MEMBER(int, mMaxHairDepth);                        \
    HUD_MEMBER(int, mMaxSubsurfacePerPath);                \
    HUD_MEMBER(float, mTransparencyThreshold);             \
    HUD_MEMBER(float, mPresenceThreshold);                 \
    HUD_MEMBER(float, mRussianRouletteThreshold);          \
    HUD_MEMBER(float, mInvRussianRouletteThreshold);       \
    HUD_MEMBER(float, mSampleClampingValue);               \
    HUD_MEMBER(int, mSampleClampingDepth);                 \
    HUD_MEMBER(float, mRoughnessClampingFactor);           \
    HUD_MEMBER(float, mInvVolumeQuality);                  \
    HUD_MEMBER(float, mInvVolumeShadowQuality);            \
    HUD_MEMBER(int, mVolumeIlluminationSamples);           \
    HUD_MEMBER(float, mVolumeTransmittanceThreshold);      \
    HUD_MEMBER(float, mVolumeAttenuationFactor);           \
    HUD_MEMBER(float, mVolumeContributionFactor);          \
    HUD_MEMBER(float, mVolumePhaseAttenuationFactor);      \
    HUD_MEMBER(VolumeOverlapMode, mVolumeOverlapMode);     \
    HUD_MEMBER(float, mResolution);                        \
    HUD_MEMBER(bool, mEnableSSS);                          \
    HUD_MEMBER(bool, mEnableShadowing);                    \
    HUD_MEMBER(int, mDeepMaxLayers);                       \
    HUD_MEMBER(float, mDeepLayerBias);                     \
    HUD_MEMBER(int, mPad0);                                \
    HUD_CPP_MEMBER(std::vector<int>, mDeepIDAttrIdxs, 24); \
    HUD_MEMBER(int, mCryptoUVAttrIdx);                     \
    HUD_MEMBER(int, mPad1);                                \
    HUD_CPP_MEMBER(PathGuide, mPathGuide, 8)
                

#define PATH_INTEGRATOR_VALIDATION                                 \
    HUD_BEGIN_VALIDATION(PathIntegrator);                          \
    HUD_VALIDATE(PathIntegrator, mLightSamples);                   \
    HUD_VALIDATE(PathIntegrator, mBsdfSamplesSqrt);                \
    HUD_VALIDATE(PathIntegrator, mBsdfSamples);                    \
    HUD_VALIDATE(PathIntegrator, mBssrdfSamples);                  \
    HUD_VALIDATE(PathIntegrator, mMaxDepth);                       \
    HUD_VALIDATE(PathIntegrator, mMaxDiffuseDepth);                \
    HUD_VALIDATE(PathIntegrator, mMaxGlossyDepth);                 \
    HUD_VALIDATE(PathIntegrator, mMaxMirrorDepth);                 \
    HUD_VALIDATE(PathIntegrator, mMaxVolumeDepth);                 \
    HUD_VALIDATE(PathIntegrator, mMaxPresenceDepth);               \
    HUD_VALIDATE(PathIntegrator, mMaxHairDepth);                   \
    HUD_VALIDATE(PathIntegrator, mMaxSubsurfacePerPath);           \
    HUD_VALIDATE(PathIntegrator, mTransparencyThreshold);          \
    HUD_VALIDATE(PathIntegrator, mPresenceThreshold);              \
    HUD_VALIDATE(PathIntegrator, mRussianRouletteThreshold);       \
    HUD_VALIDATE(PathIntegrator, mInvRussianRouletteThreshold);    \
    HUD_VALIDATE(PathIntegrator, mSampleClampingValue);            \
    HUD_VALIDATE(PathIntegrator, mSampleClampingDepth);            \
    HUD_VALIDATE(PathIntegrator, mRoughnessClampingFactor);        \
    HUD_VALIDATE(PathIntegrator, mInvVolumeQuality);               \
    HUD_VALIDATE(PathIntegrator, mInvVolumeShadowQuality);         \
    HUD_VALIDATE(PathIntegrator, mVolumeIlluminationSamples);      \
    HUD_VALIDATE(PathIntegrator, mVolumeTransmittanceThreshold);   \
    HUD_VALIDATE(PathIntegrator, mVolumeAttenuationFactor);        \
    HUD_VALIDATE(PathIntegrator, mVolumeContributionFactor);       \
    HUD_VALIDATE(PathIntegrator, mVolumePhaseAttenuationFactor);   \
    HUD_VALIDATE(PathIntegrator, mVolumeOverlapMode);              \
    HUD_VALIDATE(PathIntegrator, mResolution);                     \
    HUD_VALIDATE(PathIntegrator, mEnableSSS);                      \
    HUD_VALIDATE(PathIntegrator, mEnableShadowing);                \
    HUD_VALIDATE(PathIntegrator, mDeepMaxLayers);                  \
    HUD_VALIDATE(PathIntegrator, mDeepLayerBias);                  \
    HUD_VALIDATE(PathIntegrator, mPad0);                           \
    HUD_VALIDATE(PathIntegrator, mDeepIDAttrIdxs);                 \
    HUD_VALIDATE(PathIntegrator, mCryptoUVAttrIdx);                \
    HUD_VALIDATE(PathIntegrator, mPad1);                           \
    HUD_VALIDATE(PathIntegrator, mPathGuide);                      \
    HUD_END_VALIDATION

