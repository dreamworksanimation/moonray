// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

// @file PresenZCamera.h

#pragma once

#include "Camera.h"

// Forward declarations
namespace moonray {
namespace mcrt_common {
    class PresenZSettings;
} // namespace rndr
} // namespace moonray

namespace moonray {
namespace pbr {

class PresenZCamera : public Camera
{
public:
    /// Constructor
    explicit PresenZCamera(const scene_rdl2::rdl2::Camera* rdlCamera);

    mcrt_common::PresenZSettings* getPresenZSettings() { return mPresenZSettings.get(); }

private:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass& sceneClass);

    bool getIsDofEnabledImpl() const override;

    void updateImpl(const scene_rdl2::math::Mat4d& world2render) override;

    void createRayImpl(mcrt_common::RayDifferential* dstRay,
                       float x,
                       float y,
                       float time,
                       float lensU,
                       float lensV) const override;

    scene_rdl2::math::Vec3f createDirection(float x, float y) const;

    static bool                               sAttributeKeyInitialized;

    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>     sPresenZEnabledKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>      sPhaseKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::String>   sDetectFileKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::String>   sRenderFileKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>    sRenderScaleKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Vec3f>    sZOVScaleKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>    sDistanceToGroundKey;

    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>     sDraftRenderingKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>     sFroxtrumRenderingKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>      sFroxtrumDepthKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>      sFroxtrumResolutionKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>     sRenderInsideZOVKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>     sEnableDeepReflectionsKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>    sInterPupillaryDistanceKey;

    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>      sZOVOffsetXKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>      sZOVOffsetYKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>      sZOVOffsetZKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Vec3f>    sSpecularPointOffsetKey;

    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>     sEnableClippingSphereKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float>    sClippingSphereRadiusKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Vec3f>    sClippingSphereCenterKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>     sClippingSphereRenderInsideKey;

    std::unique_ptr<mcrt_common::PresenZSettings> mPresenZSettings;
};

} // namespace pbr
} // namespace moonray
