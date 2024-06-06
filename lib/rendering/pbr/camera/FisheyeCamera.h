// Copyright 2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Camera.h"

namespace moonray {
namespace pbr {

class FisheyeCamera : public Camera
{
public:
    /// Constructor
    explicit FisheyeCamera(const scene_rdl2::rdl2::Camera* rdlCamera);

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

    static bool sAttributeKeyInitialized;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>   sMappingKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>   sFormatKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sZoomKey;

    int mMapping;
    float mRadialScale;
};

} // namespace pbr
} // namespace moonray


