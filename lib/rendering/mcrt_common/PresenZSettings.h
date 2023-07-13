// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#pragma once

#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec2.h>

// PresenZ includes
#include <API/PzPhaseApi.h>
#include <API/PzAppLogger.h>
#include <common/matrix.h>

namespace moonray {
namespace mcrt_common {

using namespace PresenZ::Phase;

// This class stores the settings and starts/stops PresenZ
class PresenZSettings
{
public:
    PresenZSettings();

    bool phaseBegin(unsigned numThreads);
    void phaseEnd();

    void setEnabled(bool enabled) { mEnabled = enabled; }
    bool getEnabled() const { return mEnabled; }

    void setPhase(int phase) { mPhase = static_cast<Phase>(phase); }
    Phase getPhase() const { return mPhase; }

    void setDetectFile(const std::string& detectFile) { mDetectFile = detectFile; }
    const std::string& getDetectFile() { return mDetectFile; }

    void setRenderFile(const std::string& detectFile) { mRenderFile = detectFile; }
    const std::string& getRenderFile() { return mRenderFile; }

    void setCamToWorld(const scene_rdl2::math::Mat4d& camToWorld);

    void setResolution(unsigned width, unsigned height) {
        mResolution = scene_rdl2::math::Vec2f(static_cast<float>(width), static_cast<float>(height));
    }
    const scene_rdl2::math::Vec2f& getResolution() const { return mResolution; }

    void setRenderScale(float renderScale) { mRenderScale = renderScale; }
    float getRenderScale() const { return mRenderScale; }

    void setZOVScale(const scene_rdl2::math::Vec3f& zovScale) { mZOVScale = zovScale; }
    const scene_rdl2::math::Vec3f& getZOVScale() const { return mZOVScale; }

    void setDistanceToGround(float distanceToGround) { mDistanceToGround = distanceToGround; }
    float getDistanceToGround() const { return mDistanceToGround; }

    void setDraftRendering(bool draftRendering) { mDraftRendering = draftRendering; }
    bool getDraftRendering() const { return mDraftRendering; }

    void setRenderInsideZOV(bool renderInsideZOV) { mRenderInsideZOV = renderInsideZOV; }
    bool getRenderInsideZOV() const { return mRenderInsideZOV; }

    void setEnableDeepReflections(bool enableDeepReflections) { mEnableDeepReflections = enableDeepReflections; }
    bool getEnableDeepReflections() const { return mEnableDeepReflections; }

    void setInterpupillaryDistance(float interpupillaryDistance) { mInterPupillaryDistance = interpupillaryDistance; }
    float getInterpupillaryDistance() const { return mInterPupillaryDistance; }

    void setZOVOffset(int offsetX, int offsetY, int offsetZ) { mZOVOffset = scene_rdl2::math::Vec3i(offsetX, offsetY, offsetZ); }
    const scene_rdl2::math::Vec3i& getZOVOffset() const { return mZOVOffset; }

    void setSpecularPointOffset(const scene_rdl2::math::Vec3f& specularPointOffset) { mSpecularPointOffset = specularPointOffset; }
    const scene_rdl2::math::Vec3f& getSpecularPointOffset() const { return mSpecularPointOffset; }

    void setEnableClippingSphere(bool enableClippingSphere) { mEnableClippingSphere = enableClippingSphere; }
    bool getEnableClippingSphere() const { return mEnableClippingSphere; }

    void setClippingSphereRadius(float clippingSphereRadius) { mClippingSphereRadius = clippingSphereRadius; }
    float getClippingSphereRadius() const { return mClippingSphereRadius; }

    void setClippingSphereCenter(const scene_rdl2::math::Vec3f& clippingSphereCenter) { mClippingSphereCenter = clippingSphereCenter; }
    const scene_rdl2::math::Vec3f& getClippingSphereCenter() const { return mClippingSphereCenter; }

    void setClippingSphereRenderInside(bool clippingSphereRenderInside) { mClippingSphereRenderInside = clippingSphereRenderInside; }
    bool getClippingSphereRenderInside() const { return mClippingSphereRenderInside; }

    void setCurrentFrame(int currentFrame) { mCurrentFrame = currentFrame; }
    int getCurrentFrame() { return mCurrentFrame; }

private:
    bool mEnabled;
    Phase mPhase;
    std::string mDetectFile;
    std::string mRenderFile;
    NozMatrix mCamToWorld;
    scene_rdl2::math::Vec2f mResolution;
    float mRenderScale;
    scene_rdl2::math::Vec3f mZOVScale;
    float mDistanceToGround;
    bool mDraftRendering;
    bool mRenderInsideZOV;
    bool mEnableDeepReflections;
    float mInterPupillaryDistance;
    scene_rdl2::math::Vec3i mZOVOffset;
    scene_rdl2::math::Vec3f mSpecularPointOffset;
    bool mEnableClippingSphere;
    float mClippingSphereRadius;
    scene_rdl2::math::Vec3f mClippingSphereCenter;
    bool mClippingSphereRenderInside;
    int mCurrentFrame;
};

} // namespace mcrt_common
} // namespace moonray

// TM and (c) 2019 DreamWorks Animation LLC.  All Rights Reserved.
// Reproduction in whole or in part without prior written permission of a
// duly authorized representative is prohibited.
