// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PresenZSettings.cc
#include "PresenZSettings.h"

namespace moonray {
namespace mcrt_common {

PresenZSettings::PresenZSettings() :
    mEnabled(true),
    mPhase(PresenZ::Phase::Detect),
    mDetectFile("render.przDetect"),
    mRenderFile("render.przRender"),
    mCamToWorld(getNozM4Identity()),
    mResolution(3000.0f, 2000.0f),
    mRenderScale(1.0f),
    mZOVScale(1.0f, 0.5f, 1.0f),
    mDistanceToGround(1.6f),
    mDraftRendering(false),
    mFroxtrumRendering(false),
    mFroxtrumDepth(6),
    mFroxtrumResolution(8),
    mRenderInsideZOV(false),
    mEnableDeepReflections(true),
    mInterPupillaryDistance(63.5f),
    mZOVOffset(0, 0, 0),
    mSpecularPointOffset(0.0f, 0.0f, 0.0f),
    mEnableClippingSphere(false),
    mClippingSphereRadius(100.0f),
    mClippingSphereCenter(0.0f, 0.0f, 0.0f),
    mClippingSphereRenderInside(true),
    mCurrentFrame(0)
{
}

void
PresenZSettings::setCamToWorld(const scene_rdl2::math::Mat4d& camToWorld) {
    nozMatrix_set(mCamToWorld,
        camToWorld[0][0], camToWorld[0][1], camToWorld[0][2], 0.0,
        camToWorld[1][0], camToWorld[1][1], camToWorld[1][2], 0.0,
        camToWorld[2][0], camToWorld[2][1], camToWorld[2][2], 0.0,
        camToWorld[3][0], camToWorld[3][1], camToWorld[3][2], 1.0);
}

bool
PresenZSettings::phaseBegin(unsigned numThreads) {
    // Initialize the phase
    if (mPhase == PresenZ::Phase::Detect) {
        PzInitPhase(PresenZ::Phase::Detect, PresenZ::Util::RenderingEngineId::PRESENZ_DEVELOP);
        PzSetOutFilePath(mDetectFile.c_str());
    } else if (mPhase == PresenZ::Phase::Render) {
        PzInitPhase(PresenZ::Phase::Render, PresenZ::Util::RenderingEngineId::PRESENZ_DEVELOP);
        PzSetOutFilePath(mRenderFile.c_str());
        PzSetDetectFilePath(mDetectFile.c_str());
        PzSetFroxtrumRayOrigin(PresenZ::Phase::FroxtrumRayOrigin::useCameraNearClip);
    }

    // Draft mode
    PzSetDraft(mDraftRendering);

    // enable/disable the new feature
    PzSetFroxtrum(mFroxtrumRendering);

    // set the volume of the froxtrum. By default, it's set to a
    // conservative value of 6. Depending on your scene, you can turn
    // it up to 7 or 8 to gain more render time, at the cost of some
    // image degradation
    PzSetFroxtrumDepth(mFroxtrumDepth);

    // should not be changed, but turning it to 4 could create more froxtrum
    PzSetFroxtrumResolution(mFroxtrumResolution);

    // Zone of View
    PzSetZovOffset(static_cast<float>(mZOVOffset.x),
                   static_cast<float>(mZOVOffset.y),
                   static_cast<float>(mZOVOffset.z));
    PzSetZovScaling(mZOVScale.x, mZOVScale.y, mZOVScale.z);
    PzSetRenderInsideBox(mRenderInsideZOV);

    // Scene
    PzSetCameraToWorldMatrix(mCamToWorld);
    PzSetRenderScale(mRenderScale);
    PzSetDistanceToGround(mDistanceToGround);
    //PzSetSceneUpVector(NozVector(0.0f, 1.0f, 0.0f));
    PzSetCameraUpVector(NozVector(0.0f, 1.0f, 0.0f));
    PzSetSpecularPointOffset(NozVector(mSpecularPointOffset.x,
                                       mSpecularPointOffset.y,
                                       mSpecularPointOffset.z));

    PzSetCameraSpace(Space::Camera);
    PzSetSampleSpace(Space::Camera);

    // Animation
    PzSetCurrentFrame(mCurrentFrame);
    PzSetMotionVector(true);

    // Transparency and reflections
    PzSetRenderTransparencyMode(PresenZ::Phase::TransparencyRenderingType::PRZ_REGULAR);
    if (mEnableDeepReflections) {
        PzSetDeepReflection(PresenZ::Phase::Eye::RC_LeftAndRight, mInterPupillaryDistance);
    } else {
        PzSetDeepReflection(PresenZ::Phase::Eye::RC_Left, mInterPupillaryDistance);
    }

    // Resolution
    PzSetOutputResolution(mResolution.x, mResolution.y);
    const PzResolutionParam rp = PzGetRenderingResolutionParameters();
    setResolution(rp.resolutionX, rp.resolutionY);
    PzSetRenderingResolutionParameters(rp);

    // Bucketing and threads
    PzSetBucketSize(8, 8);
    PzSetThreadNumber(numThreads);

    // Clipping sphere
    PzSetClippingSphere(mEnableClippingSphere,
                        NozVector(mClippingSphereCenter.x,
                                  mClippingSphereCenter.y,
                                  mClippingSphereCenter.z),
                        mClippingSphereRadius,
                        !mClippingSphereRenderInside);

    //PresenZ::Logger::PzSetConsole(true);
    //PresenZ::Logger::PzSetConsoleLogLevel(PresenZ::Logger::LL_Debug);

    return PzPhaseBegin();
}

void
PresenZSettings::phaseEnd() {
    PzPhaseTerminate();
}


} // namespace mcrt_common
} // namespace moonray

